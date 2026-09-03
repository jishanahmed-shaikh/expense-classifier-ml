"""
src/predict.py

Standalone inference script for the Expense Classifier ML model.

Usage:
    python src/predict.py

Or import the function directly:
    from src.predict import predict_account
"""

import json
import re
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — resolve relative to this file so the script works from any cwd
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
MODEL_BUNDLE_PATH = PROJECT_ROOT / "expense_classifier_model.pkl"
DATA_FILE_PATH    = PROJECT_ROOT / "data" / "accounts-bills.json"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: clean a raw expense text string
# Same logic as used during training — must stay in sync with the notebook
# ─────────────────────────────────────────────────────────────────────────────
def clean_expense_text(ITEM_NAME: str, ITEM_DESCRIPTION: str) -> str:
    """Normalise item name + description into a single vectorisable string."""
    ITEM_NAME        = str(ITEM_NAME).strip()
    ITEM_DESCRIPTION = str(ITEM_DESCRIPTION).strip()

    # Strip leading 4-digit booking-period codes (e.g. "0226 ", "1225-")
    ITEM_NAME        = re.sub(r'^\d{4}[-\s]+', '', ITEM_NAME)
    ITEM_DESCRIPTION = re.sub(r'^\d{4}[-\s]+', '', ITEM_DESCRIPTION)

    ITEM_NAME        = ITEM_NAME.lower()
    ITEM_DESCRIPTION = ITEM_DESCRIPTION.lower()

    # Avoid duplicating text when both fields are identical
    if ITEM_NAME == ITEM_DESCRIPTION or not ITEM_DESCRIPTION:
        COMBINED = ITEM_NAME
    else:
        COMBINED = ITEM_NAME + ' | ' + ITEM_DESCRIPTION

    COMBINED = re.sub(r'[^a-z0-9 |]', ' ', COMBINED)
    COMBINED = re.sub(r'\s+', ' ', COMBINED).strip()
    return COMBINED


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build a sparse vendor one-hot row for a single vendor ID
# ─────────────────────────────────────────────────────────────────────────────
def build_vendor_onehot_row(VENDOR_ID: str, TOP_VENDOR_LIST: list) -> csr_matrix:
    """Return a (1, N) sparse one-hot row for the given vendor."""
    VENDOR_INDEX_MAP = {VID: IDX for IDX, VID in enumerate(TOP_VENDOR_LIST)}
    ROW              = np.zeros((1, len(TOP_VENDOR_LIST)), dtype=np.float32)
    if VENDOR_ID in VENDOR_INDEX_MAP:
        ROW[0, VENDOR_INDEX_MAP[VENDOR_ID]] = 1.0
    return csr_matrix(ROW)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD VENDOR FREQUENCY MAP  — needed to build the vendor_log_freq feature
# Builds once at module level so repeated calls to predict_account are fast
# ─────────────────────────────────────────────────────────────────────────────
def _load_vendor_freq_map() -> dict:
    """Read the dataset and return a {vendor_id: count} frequency dict."""
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as FILE_HANDLE:
        RAW_RECORDS = json.load(FILE_HANDLE)
    VENDOR_COUNTS = {}
    for RECORD in RAW_RECORDS:
        VID = RECORD.get('vendorId', 'unknown')
        VENDOR_COUNTS[VID] = VENDOR_COUNTS.get(VID, 0) + 1
    return VENDOR_COUNTS


VENDOR_FREQ_MAP     = _load_vendor_freq_map()

# Rebuild a LabelEncoder for vendor IDs (needed to produce the vendor_encoded feature)
_ALL_VENDORS        = sorted(VENDOR_FREQ_MAP.keys())
VENDOR_LABEL_ENC    = LabelEncoder()
VENDOR_LABEL_ENC.fit(_ALL_VENDORS)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_account(
    ITEM_NAME: str,
    ITEM_DESCRIPTION: str = '',
    VENDOR_ID: str = '',
    AMOUNT: float = 0.0,
    TOP_K: int = 3
) -> list:
    """Predict the accountName for a new expense line item.

    Parameters
    ----------
    ITEM_NAME        : expense item name
    ITEM_DESCRIPTION : expense item description (may equal ITEM_NAME)
    VENDOR_ID        : vendor identifier string
    AMOUNT           : transaction amount (negative values allowed)
    TOP_K            : number of top predictions to return

    Returns
    -------
    List of (account_name, probability_or_None) tuples sorted by confidence.
    LinearSVC does not produce probabilities — returns None for that field.
    """
    # Load the saved model bundle
    BUNDLE           = joblib.load(MODEL_BUNDLE_PATH)
    SAVED_MODEL      = BUNDLE['model']
    SAVED_TFIDF_WORD = BUNDLE['tfidf_word']
    SAVED_TFIDF_CHAR = BUNDLE['tfidf_char']
    SAVED_SCALER     = BUNDLE['numeric_scaler']
    SAVED_LABEL_ENC  = BUNDLE['label_encoder']
    SAVED_VENDOR_IDS = BUNDLE['top_vendor_ids']

    # ── Text features ─────────────────────────────────────────────────────────
    CLEANED_TEXT  = clean_expense_text(ITEM_NAME, ITEM_DESCRIPTION)
    TEXT_WORD_VEC = SAVED_TFIDF_WORD.transform([CLEANED_TEXT])
    TEXT_CHAR_VEC = SAVED_TFIDF_CHAR.transform([CLEANED_TEXT])
    TEXT_VEC      = hstack([TEXT_WORD_VEC, TEXT_CHAR_VEC])

    # ── Numeric features ──────────────────────────────────────────────────────
    ABS_AMOUNT = max(abs(AMOUNT), 0.01)
    LOG_AMT    = np.log1p(ABS_AMOUNT)
    IS_NEG     = float(AMOUNT < 0)
    AMT_BKT    = 5.0    # default to median bucket when amount context is unknown

    V_LOG_FREQ = np.log1p(VENDOR_FREQ_MAP.get(VENDOR_ID, 1))
    try:
        V_ENCODED = float(VENDOR_LABEL_ENC.transform([VENDOR_ID])[0])
    except ValueError:
        V_ENCODED = 0.0   # unseen vendor — use fallback

    NUM_RAW    = np.array([[V_LOG_FREQ, V_ENCODED, IS_NEG, LOG_AMT, AMT_BKT]])
    NUM_SCALED = csr_matrix(SAVED_SCALER.transform(NUM_RAW))

    # ── Vendor one-hot ────────────────────────────────────────────────────────
    VENDOR_OH  = build_vendor_onehot_row(VENDOR_ID, SAVED_VENDOR_IDS)

    # ── Stack all features ────────────────────────────────────────────────────
    X_INFER    = hstack([TEXT_VEC, NUM_SCALED, VENDOR_OH])

    # ── Predict ───────────────────────────────────────────────────────────────
    if hasattr(SAVED_MODEL, 'predict_proba'):
        PROBA_ARRAY = SAVED_MODEL.predict_proba(X_INFER)[0]
        TOP_IDX     = PROBA_ARRAY.argsort()[-TOP_K:][::-1]
        return [(SAVED_LABEL_ENC.classes_[I], round(float(PROBA_ARRAY[I]), 4)) for I in TOP_IDX]
    else:
        # LinearSVC — single best prediction, no probability score
        PRED_LABEL = SAVED_MODEL.predict(X_INFER)[0]
        PRED_NAME  = SAVED_LABEL_ENC.inverse_transform([PRED_LABEL])[0]
        return [(PRED_NAME, None)]


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  — run a few example predictions when the script is executed directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DEMO_RECORDS = [
        ('monthly subscription fee',     'slack monthly plan',      '',   99.0),
        ('legal services retainer',      'legal services retainer', '',  5000.0),
        ('recruiter commission',         'hiring agency fee',       '',  3500.0),
        ('office supplies',              '',                         '',   250.0),
        ('cloud infrastructure monthly', 'aws usage',               '',  8000.0),
    ]

    print('Expense Classifier ML — Demo Predictions')
    print('=' * 60)
    for DEMO_NAME, DEMO_DESC, DEMO_VENDOR, DEMO_AMT in DEMO_RECORDS:
        RESULTS = predict_account(DEMO_NAME, DEMO_DESC, DEMO_VENDOR, DEMO_AMT)
        print(f'\n  Item   : {DEMO_NAME}')
        for ACCOUNT_NAME, PROB in RESULTS:
            PROB_STR = f'{PROB:.2%}' if PROB is not None else 'n/a'
            print(f'    {PROB_STR:>8}  {ACCOUNT_NAME}')
