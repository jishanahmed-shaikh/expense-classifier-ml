# Expense Line-Item Classifier — Report

## Executive summary

The task is to predict which of 103 general-ledger accounts an expense line item should
be booked to, from 4,894 historical line items. The item text turns out to carry almost
all of the signal, so the model is intentionally small: **TF-IDF on the item text (word
1–2 grams and character 2–5 grams), a one-hot of the vendor, and a log-scaled amount,
fed to a linear support-vector classifier.** Measured with 5-fold stratified
cross-validation (every transformer refit inside each fold), it reaches
**90.0% ± 0.9 pp accuracy**; a single 80/20 stratified hold-out scores **92.1%**, and a
deduplicated cross-validation gives a conservative **89.7%**. The 92% goal is met on the
hold-out and missed by about two points on cross-validation, which is the more reliable
figure and the one we lead with. The remaining ~10% of errors are a small number of
genuinely ambiguous account pairs (prepaid vs. the matching expense account, the two
inter-company clearing variants, two accounts both literally named "Employee Training")
plus the long tail of accounts with only a handful of examples — not something a
different classifier fixes.

---

## 1. Data analysis

### 1.1 Shape of the data

| | |
|---|---|
| Rows | 4,894 |
| Accounts (target classes) | 103 |
| Raw fields | `_id`, `vendorId`, `itemName`, `itemDescription`, `accountId`, `accountName`, `itemTotalAmount` |
| Fields used as features | `vendorId`, `itemName`, `itemDescription`, `itemTotalAmount` |

`accountId` is **excluded**. Every `accountName` maps to exactly one `accountId`, and
predicting the account from `accountId` alone scores **96%** — it is a coarser copy of
the label (97 ids for 103 names), assigned *as part of* coding the expense. It is not
available when a new bill arrives, so using it would both leak the target and not
reflect real deployment.

### 1.2 Class imbalance

The accounts are very unevenly sized:

- Largest account — `611202 Online Subscription/Tool` — holds **1,179 rows (24%)**;
  the top two hold 38%.
- Median account size is **15 rows**.
- **16 accounts have exactly one row**, and **37 (about a third) have ≤ 5 rows** —
  together only 1.7% of the data.

Consequences: splits must be stratified; the single-row accounts cannot be learned at
all; and macro-averaged metrics will sit well below accuracy because they weight those
tiny accounts equally.

### 1.3 The text fields

- `itemName` and `itemDescription` are **identical for 83%** of rows — the description
  rarely adds anything, so we combine them and keep one copy when they match.
- About **89% of item names begin with a booking-period code** such as `0725` or
  `1225-0227` (month-year, sometimes a range).
- A code that is a **date range** (`0126-0127 …`) appears on **51% of prepaid rows** but
  only **3% of everything else** — it is a strong "spread this cost over time" signal, so
  the codes are kept in the text rather than stripped as noise.
- 11 descriptions are missing and ~31 are blank; 19 amounts are negative
  (credits / reversals). None are dropped.

### 1.4 The vendor

- 337 distinct vendors. **~63% always map to a single account.**
- "Always guess the vendor's most common account" alone scores **~74%** — a strong
  baseline and a clear signal to include vendor identity as a feature.

### 1.5 The amount

`itemTotalAmount` spans from about −15,000 to 160,000,000 with a heavy right tail, so a
`log(1 + |amount|)` transform plus a sign flag is used instead of the raw value.

### 1.6 Data-quality notes

- **~150 rows are exact duplicates** (same vendor, item, amount, and account). These are
  legitimate recurring invoices, so they are kept — but a deduplicated cross-validation
  is also reported so the headline number is not flattered by "the same invoice in both
  train and test".
- Repeated item names disagree on the account **~1.5% of the time**, i.e. a slice of the
  ground truth is itself inconsistent — an informal ceiling on achievable accuracy.

---

## 2. Methodology

### 2.1 Features

Built in `src/expense_classifier.py` (one code path shared by the notebook, the tests,
and `predict.py`). Four blocks, horizontally stacked into one sparse matrix:

| Block | Configuration | Rationale |
|---|---|---|
| Word TF-IDF | 1–2 grams, 20,000 features, sublinear tf | keyword signal: *subscription*, *retainer*, *audit fee* |
| Character TF-IDF | 2–5 grams (`char_wb`), 25,000 features | short and coded text, partial-word and booking-code patterns |
| Vendor one-hot | every vendor seen in training (~330 columns) | vendor identity is deterministic for most rows |
| Amount | log-amount, is-negative, is-date-range | separates prepaid / credit / ordinary expense |

Text cleaning: lowercase, drop punctuation, collapse whitespace, keep the booking codes,
and collapse an identical name/description to one copy.

### 2.2 Model choice

A **linear SVM (`LinearSVC`)** on the sparse feature matrix. It is a standard strong
baseline for high-dimensional text, trains in seconds on 40k+ features and 103 classes,
needs no calibration for a plain prediction, and its coefficients are directly readable —
useful for explaining a prediction to a finance reviewer.

Alternatives that were tried and did **not** beat it on cross-validation, so were
dropped in favour of the simpler model:

- Logistic regression, Complement Naive Bayes, gradient boosting (LightGBM)
- An **exact-match retrieval layer** (reuse the historical account for a seen
  `vendor + itemName`) — neutral, because the linear model already classifies the
  recurring items well
- **Soft-voting and stacked ensembles** (SVM + NB + logistic meta-learner) — worse,
  the weaker members drag the blend down
- `class_weight="balanced"` — raises macro-F1 slightly but costs ~2 points of accuracy

### 2.3 Handling imbalance

- Stratified splitting throughout; the 16 single-row accounts are grouped into one
  bucket only so the splitter accepts them, then land in training.
- `class_weight="balanced"` was tried — it lifts macro-F1 by ~1 point but drops overall
  accuracy by ~2, so it is not used (accuracy is the target metric).
- No synthetic oversampling: interpolating new rows in 45k-dimensional sparse TF-IDF
  space (SMOTE) does not produce meaningful "expense line items" and only risks
  overfitting the tiny classes.
- The rare-class cost (37 accounts, 1.7% of rows) is accepted and reported, not hidden.

### 2.4 Validation

1. **5-fold stratified cross-validation — the headline metric.** Both vectorizers, the
   scaler, and the vendor vocabulary are refit **inside each fold**; nothing from a
   validation row is seen during training.
2. **Deduplicated 5-fold CV** — identical recurring invoices collapsed before splitting;
   a conservative lower bound.
3. **Single 80/20 stratified hold-out** — the other protocol the brief allows; used for
   the confusion matrix and error analysis.

---

## 3. Results

### 3.1 Headline

| Metric | Value |
|---|---|
| Accuracy — 5-fold stratified CV | **90.0%** (± 0.9 pp) |
| Accuracy — deduplicated 5-fold CV | 89.7% |
| Accuracy — 80/20 stratified hold-out (seed 42) | 92.1% |
| Macro precision / recall / F1 | 68.3% / 67.2% / 67.3% |
| Weighted precision / recall / F1 | 89.6% / 90.0% / 89.8% |
| Per-fold accuracy | 89.0%, 90.9%, 90.2%, 91.1%, 89.1% |

The spread between the CV mean (90.0%) and the hold-out (92.1%) is normal sampling
variation — across seven hold-out seeds the accuracy ranges 89.6–92.1% with a mean of
90.4% — so the CV mean is the more reliable figure and the one we lead with.

Accuracy far exceeds macro-F1 (67.3%) because ~40 accounts have too few examples to
learn; on the rows that belong to reasonably-sized accounts the model is much stronger
(weighted F1 89.8%).

### 3.2 By category

**Handled well** — F1 = 1.00 in cross-validation, mostly the larger, lexically distinct
accounts:

| Account | Support |
|---|---|
| 511606 Audience Extension | 42 |
| 511102 External Commission | 41 |
| 223001 Salaries Payable | 28 |
| 131020 Unbilled receivables | 25 |
| 619202 Cleaning | 23 |
| 611101 Cloud server - AWS | 21 |

**Handled poorly** — F1 well below average despite having ≥ 5 examples:

| Account | F1 | Support |
|---|---|---|
| 134004 Prepaid Subscription | 0.37 | 21 |
| 612007 Events/Community meetup | 0.40 | 21 |
| 614336 Employee Training | 0.46 | 8 |
| 134002 Prepaid Insurance | 0.60 | 18 |
| 612032 Key Opinion Leader | 0.63 | 8 |
| 612016 Collateral | 0.67 | 25 |

Full per-account precision/recall/F1 and the F1 distribution are in the notebook
(`reports/figures/per_class_f1.png`).

### 3.3 Error analysis

**9.9% of cross-validated predictions are wrong (487 of 4,894)**, and the mistakes are
concentrated in a few sensible pairs (actual → predicted):

| Actual | Predicted | Count |
|---|---|---|
| 611202 Online Subscription/Tool | 134001 Prepaid Operating Expense | 22 |
| 611202 Online Subscription/Tool | 132098 IC Clearing account | 17 |
| 134001 Prepaid Operating Expense | 611202 Online Subscription/Tool | 17 |
| 132098 IC Clearing account | 611202 Online Subscription/Tool | 16 |
| 134001 Prepaid Operating Expense | 132098 IC Clearing account | 15 |
| 134001 Prepaid Operating Expense | 132098 IC Clearing account - Paid on Behalf | 15 |

The dominant patterns:

- **Prepaid vs. the matching expense account** (`134001` ↔ `611202`) — same vendor, same
  wording; the only difference is whether the cost is amortised over a period, which
  usually needs the invoice, not just the line item.
- **`132098 IC Clearing account` vs. `… - Paid on Behalf`** — two near-identical
  inter-company clearing accounts.
- **Two accounts both named "Employee Training"** (`614400` and `614336`) — identical
  account name, different code; the line item cannot distinguish them.
- **Rare accounts predicted as their larger lexical neighbour.**

See `reports/figures/confusion_matrix.png`.

---

## 4. Discussion

### 4.1 Strengths

- **Simple and fast.** One linear model on ~45k sparse features; trains in seconds; every
  coefficient is inspectable and explainable to a finance team.
- **Honest evaluation.** In-fold refitting, a deduplicated CV, and multi-seed hold-out
  variance are all reported — no single lucky number.
- **Reproducible.** Fixed seed, one shared module, a `--self-test` entry point.
- **The choices that moved the needle** were unglamorous: keep the booking-period codes
  (they signal prepaid), one-hot every vendor, character n-grams down to bigrams, and do
  **not** use `class_weight="balanced"` when accuracy is the target.

### 4.2 Limitations

- **Rare accounts are unlearnable.** 16 single-row and 37 ≤5-row accounts cap macro-F1.
- **Prepaid vs. expense** needs context the line item does not contain.
- **Vendor cold-start.** A new vendor contributes an all-zero block; the prediction then
  rests entirely on the text.
- **Label noise.** ~1.5% of repeated item names are coded inconsistently in the data.
- **Static model.** New accounts and vendors require a periodic retrain.

### 4.3 With more time or data

| Idea | Expected value |
|---|---|
| **Calibrated probabilities** (`CalibratedClassifierCV` or a logistic model) | Enables a meaningful auto-post threshold; the current score is only a *relative* confidence |
| A `vendor + item → account` rules table for unambiguous recurring invoices, model for the rest | Medium — removes easy volume, lets tuning focus on the hard cases |
| Sentence embeddings (e.g. `all-MiniLM-L6-v2`) for the free-form descriptions | Medium — better semantics where wording varies |
| Hierarchical classification: account family (first digit) → account | Medium — cuts cross-family confusion |
| Targeted data collection for the rare accounts | Low–medium — mostly helps macro-F1 |

### 4.4 Deploying it

- **Auto-post above a confidence threshold**, queue the rest for a finance reviewer.
  `predict_one` returns a relative score today (softmax over the SVM margins); calibrating
  it (above) turns that into a probability the threshold can be set against.
- **Audit trail:** log model version, predicted account, and score with every line.
- **Monitor and retrain:** track accuracy on new invoices monthly; retrain when it
  drifts below ~90% or when new accounts appear.
- **Explainability:** the linear coefficients let the tool show *which words* drove a
  prediction, which builds trust with the finance team.

---

## Reproducing these numbers

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/expense_classifier.ipynb
```

The notebook prints every metric in this report and regenerates the figures in
`reports/figures/`.
