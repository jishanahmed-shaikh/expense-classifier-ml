# Expense Line-Item Classifier

Predicts which **general-ledger account** an expense line item should be booked to, from
its description, vendor, and amount.

- **Data:** 4,894 line items, 103 accounts
- **Model:** TF-IDF (word + character n-grams) + vendor one-hot + log-amount → linear SVM
- **Accuracy:** **90.0%** (5-fold stratified cross-validation), 92.1% on an 80/20 hold-out

Full write-up with data analysis, methodology, and error analysis:
[`reports/report.md`](reports/report.md).

---

## Why it's a hard problem

| Challenge | Detail |
|---|---|
| Many, imbalanced classes | 103 accounts; the largest holds 24% of rows, 16 appear only once |
| Terse text | `itemName` and `itemDescription` are identical for 83% of rows and often just a booking code plus a few words |
| A long tail | 37 accounts have ≤ 5 examples — too few to learn |
| Ambiguous pairs | e.g. a prepaid account vs. the matching expense account differ only by whether the cost is amortised |

## Approach

The item text nearly determines the account, so the model stays small and legible:

| Feature block | Detail |
|---|---|
| Word TF-IDF | 1–2 grams |
| Character TF-IDF | 2–5 grams (`char_wb`) — handles coded / short text |
| Vendor one-hot | every vendor seen in training |
| Amount | `log(1+|amount|)`, sign flag, "booking period is a date range" flag |

`accountId` is deliberately **not** used — it is a coded copy of the target.

Heavier options (gradient boosting, an exact-match retrieval layer, stacked ensembles)
were tried and did not beat the linear SVM on cross-validation, so the simple model wins.

---

## Repository

```
notebooks/expense_classifier.ipynb   analysis end to end: EDA → features → model → evaluation
src/expense_classifier.py            the pipeline (shared by the notebook, tests, and predict.py)
src/predict.py                       classify a line item from the command line
data/accounts-bills.json             raw dataset (read only)
reports/report.md                    written report
reports/figures/                     figures used by the report
models/                              saved model (.joblib) — git-ignored, regenerate by running the notebook
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# reproduce everything: trains the model, writes reports/figures/, prints every metric
jupyter nbconvert --to notebook --execute --inplace notebooks/expense_classifier.ipynb

# classify one line item (prints the top account + 2 runners-up)
python src/predict.py --item-name "Zoom annual subscription" --amount 1200
#  611202 Online Subscription/Tool   (score 0.03)
#    next: 134001 Prepaid Operating Expense (0.02), 131011 Advance to Supplier - Companies (0.01)

# smoke test
python src/predict.py --self-test
```

```python
import sys; sys.path.append("src")
from expense_classifier import ExpenseClassifier, load_dataset

model = ExpenseClassifier().fit(load_dataset())
model.predict_one(item_name="AWS October usage", amount=8000)
# [('611101 Cloud server - AWS', 0.017), ('619502 Bank Charges', 0.014), ('612031 Paid Content', 0.012)]
```

> The score is a *relative* confidence (softmax over the SVM margins across 103
> classes), not a calibrated probability — the winner sitting clearly above the
> runners-up is the signal, not the absolute number.

## How it's validated

- **5-fold stratified cross-validation** (the headline metric) — every transformer is
  refit inside each fold, so nothing leaks from validation rows into training.
- **Deduplicated cross-validation** — identical recurring invoices collapsed first;
  reported as a conservative lower bound (89.7%).
- **Single 80/20 stratified hold-out** — drives the confusion matrix and error analysis.

## Reproducibility

All randomness is seeded (`RANDOM_SEED = 42`); re-running the notebook gives the same
numbers. Trained artifacts go to `models/` and are not committed.
