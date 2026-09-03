"""Expense line-item classifier.

Predicts the general-ledger account (``accountName``) an expense line item should be
booked to, from its text, vendor, and amount.

The pipeline is deliberately small:

* TF-IDF on the item text  — word 1-2 grams + character 2-5 grams
* one-hot of the vendor id
* log-scaled amount, a sign flag, and a "booking period is a date range" flag
* a linear Support Vector classifier on top

Everything is wrapped in :class:`ExpenseClassifier` so the notebook, the tests, and
``predict.py`` all use the exact same code path.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "accounts-bills.json"
MODEL_PATH = REPO_ROOT / "models" / "expense_classifier.joblib"

RANDOM_SEED = 42
TARGET_COLUMN = "accountName"
TEXT_COLUMNS = ("itemName", "itemDescription")
VENDOR_COLUMN = "vendorId"
AMOUNT_COLUMN = "itemTotalAmount"

# ``accountId`` is excluded on purpose: every ``accountName`` maps to one ``accountId``
# and guessing the account from it alone already scores ~96%. It is assigned while the
# expense is coded, so it is not available at prediction time -- using it leaks the target.
# ``_id`` is a bare row key with no predictive meaning.
LEAKAGE_COLUMNS = ("accountId", "_id")

_BOOKING_RANGE = re.compile(r"^\d{4}[-/]\d{4}")           # e.g. "0126-0127 ..."
_NON_ALNUM = re.compile(r"[^a-z0-9 |]")
_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Data loading and text cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw JSON file into a tidy DataFrame.

    Flattens the Mongo-style ``_id`` field, coerces the amount to a float, and
    fills the handful of missing text / vendor values.
    """
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    frame = pd.DataFrame(records)
    if "_id" in frame.columns:
        frame["_id"] = frame["_id"].apply(
            lambda value: value.get("$oid", "") if isinstance(value, dict) else value
        )

    for column in TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    frame[VENDOR_COLUMN] = frame[VENDOR_COLUMN].fillna("").astype(str).replace("", "unknown")
    frame[AMOUNT_COLUMN] = pd.to_numeric(frame[AMOUNT_COLUMN], errors="coerce").fillna(0.0)
    return frame


def clean_item_text(item_name: str, item_description: str) -> str:
    """Combine name + description into one lowercase string.

    The leading booking-period code (``0725``, ``1225-0227``) is kept for the
    character n-grams but we drop punctuation and collapse whitespace. When the
    two fields are identical (83% of rows) only one copy is used so a term is not
    counted twice.
    """
    name = str(item_name).strip().lower()
    description = str(item_description).strip().lower()
    combined = name if (name == description or not description) else f"{name} | {description}"
    return _WHITESPACE.sub(" ", _NON_ALNUM.sub(" ", combined)).strip()


def _numeric_frame(frame: pd.DataFrame) -> np.ndarray:
    amount = frame[AMOUNT_COLUMN].to_numpy(dtype=float)
    log_amount = np.log1p(np.abs(amount).clip(min=0.01))
    is_negative = (amount < 0).astype(float)
    is_booking_range = (
        frame["itemName"].str.match(_BOOKING_RANGE).fillna(False).astype(float).to_numpy()
    )
    return np.column_stack([log_amount, is_negative, is_booking_range])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ExpenseClassifier:
    """TF-IDF + vendor + amount features feeding a LinearSVC.

    Fit it on a DataFrame that has the raw columns; call :meth:`predict` (or
    :meth:`predict_one`) to get account names back.
    """

    word_ngram: tuple[int, int] = (1, 2)
    word_max_features: int = 20_000
    char_ngram: tuple[int, int] = (2, 5)
    char_max_features: int = 25_000
    char_min_df: int = 2
    numeric_weight: float = 2.0
    C: float = 1.0
    random_state: int = RANDOM_SEED

    # fitted state (populated by ``fit``)
    word_vectorizer: TfidfVectorizer = field(default=None, repr=False)
    char_vectorizer: TfidfVectorizer = field(default=None, repr=False)
    scaler: StandardScaler = field(default=None, repr=False)
    label_encoder: LabelEncoder = field(default=None, repr=False)
    vendor_vocab: list[str] = field(default_factory=list, repr=False)
    model: LinearSVC = field(default=None, repr=False)

    # -- feature construction ------------------------------------------------ #
    def _item_text(self, frame: pd.DataFrame) -> list[str]:
        return [
            clean_item_text(name, description)
            for name, description in zip(frame["itemName"], frame["itemDescription"])
        ]

    def _vendor_onehot(self, vendor_ids) -> csr_matrix:
        index = {vendor: position for position, vendor in enumerate(self.vendor_vocab)}
        matrix = np.zeros((len(vendor_ids), len(self.vendor_vocab)), dtype=np.float32)
        for row, vendor in enumerate(vendor_ids):
            position = index.get(vendor)
            if position is not None:
                matrix[row, position] = 1.0
        return csr_matrix(matrix)

    def _transform(self, frame: pd.DataFrame) -> csr_matrix:
        text = self._item_text(frame)
        numeric = self.scaler.transform(_numeric_frame(frame)) * self.numeric_weight
        return hstack(
            [
                self.word_vectorizer.transform(text),
                self.char_vectorizer.transform(text),
                csr_matrix(numeric),
                self._vendor_onehot(frame[VENDOR_COLUMN].to_numpy()),
            ]
        ).tocsr()

    # -- public API -------------------------------------------------------- #
    def fit(self, frame: pd.DataFrame) -> "ExpenseClassifier":
        text = self._item_text(frame)

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=self.word_ngram, max_features=self.word_max_features,
            sublinear_tf=True, min_df=1, token_pattern=r"(?u)\b\w+\b",
        ).fit(text)
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=self.char_ngram, max_features=self.char_max_features,
            sublinear_tf=True, min_df=self.char_min_df, analyzer="char_wb",
        ).fit(text)
        self.scaler = StandardScaler().fit(_numeric_frame(frame))
        # every vendor seen in training gets its own column
        self.vendor_vocab = frame[VENDOR_COLUMN].value_counts().index.tolist()
        self.label_encoder = LabelEncoder().fit(frame[TARGET_COLUMN])

        self.model = LinearSVC(C=self.C, random_state=self.random_state, max_iter=20000)
        self.model.fit(self._transform(frame), self.label_encoder.transform(frame[TARGET_COLUMN]))
        return self

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(self._transform(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        codes = self.model.predict(self._transform(frame))
        return self.label_encoder.inverse_transform(codes)

    def predict_one(
        self,
        item_name: str,
        item_description: str = "",
        vendor_id: str = "",
        amount: float = 0.0,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Classify a single line item.

        Returns the ``top_k`` most likely accounts as ``(account_name, score)`` pairs,
        highest first. ``score`` is a softmax over the SVM margins: it is a *relative*
        confidence in [0, 1] (how far the winner is ahead of the field across 103
        classes), not a calibrated probability. See "future work" in the report for
        proper calibration.
        """
        row = pd.DataFrame(
            [{
                "itemName": item_name,
                "itemDescription": item_description or item_name,
                VENDOR_COLUMN: vendor_id or "unknown",
                AMOUNT_COLUMN: float(amount),
            }]
        )
        scores = self.decision_function(row)[0]
        softmax = np.exp(scores - scores.max())
        softmax /= softmax.sum()
        order = np.argsort(scores)[::-1][:top_k]
        codes = self.model.classes_[order].astype(int)
        names = self.label_encoder.inverse_transform(codes)
        return [(name, float(softmax[i])) for name, i in zip(names, order)]

    # -- persistence ----------------------------------------------------- #
    def save(self, path: str | Path = MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "ExpenseClassifier":
        return joblib.load(path)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def _stratify_labels(labels: pd.Series) -> np.ndarray:
    """Group the single-sample classes together so StratifiedKFold accepts them."""
    counts = labels.value_counts()
    return np.where(labels.map(counts).to_numpy() < 2, "__rare__", labels.to_numpy())


def cross_val_predict(
    frame: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    dedup: bool = False,
    **model_kwargs,
) -> pd.DataFrame:
    """Out-of-fold predictions with every transformer refit inside each fold.

    Returns a DataFrame (indexed like ``frame``) with ``true``, ``predicted`` and
    ``fold`` columns. Set ``dedup=True`` to collapse identical
    ``(vendorId, itemName, itemTotalAmount, accountName)`` rows before splitting,
    which removes the "same recurring invoice in train and test" effect and gives a
    conservative lower bound.
    """
    working = frame
    if dedup:
        working = frame.drop_duplicates(
            subset=["vendorId", "itemName", AMOUNT_COLUMN, TARGET_COLUMN]
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strat = _stratify_labels(working[TARGET_COLUMN])
    result = pd.DataFrame(index=working.index, columns=["true", "predicted", "fold"])

    for fold, (train_pos, test_pos) in enumerate(splitter.split(working, strat)):
        train_df = working.iloc[train_pos]
        test_df = working.iloc[test_pos]
        model = ExpenseClassifier(random_state=random_state, **model_kwargs).fit(train_df)
        result.iloc[test_pos, result.columns.get_loc("predicted")] = model.predict(test_df)
        result.iloc[test_pos, result.columns.get_loc("true")] = test_df[TARGET_COLUMN].to_numpy()
        result.iloc[test_pos, result.columns.get_loc("fold")] = fold

    return result


def fold_accuracies(cv_result: pd.DataFrame) -> np.ndarray:
    """Per-fold accuracy from a :func:`cross_val_predict` result."""
    return (
        cv_result.assign(correct=cv_result["true"] == cv_result["predicted"])
        .groupby("fold")["correct"]
        .mean()
        .to_numpy(dtype=float)
    )
