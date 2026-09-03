"""Command-line inference for the expense classifier.

Examples
--------
Train (if no saved model yet) and classify one line item::

    python src/predict.py --item-name "Zoom annual subscription" --vendor V1234 --amount 1200

Run the built-in smoke test (used by CI / pre-commit checks)::

    python src/predict.py --self-test
"""
from __future__ import annotations

import argparse
import sys

from expense_classifier import (
    DATA_PATH,
    MODEL_PATH,
    ExpenseClassifier,
    load_dataset,
)

# A few representative line items for the smoke test / demo.
DEMO_ITEMS = [
    {"item_name": "Zoom Workplace Pro monthly", "amount": 99.0},
    {"item_name": "Legal opinion on shareholder matter", "amount": 5000.0},
    {"item_name": "Recruiter placement fee - senior engineer", "amount": 3500.0},
    {"item_name": "Office pantry and stationery", "amount": 250.0},
    {"item_name": "AWS October usage", "amount": 8000.0},
]


def get_model(retrain: bool = False) -> ExpenseClassifier:
    """Load the saved model, training and saving one first if needed."""
    if MODEL_PATH.exists() and not retrain:
        return ExpenseClassifier.load(MODEL_PATH)

    print(f"No saved model at {MODEL_PATH} - training on {DATA_PATH.name} ...", file=sys.stderr)
    model = ExpenseClassifier().fit(load_dataset())
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}", file=sys.stderr)
    return model


def _run_self_test() -> int:
    model = get_model()
    failures = 0
    for item in DEMO_ITEMS:
        ranked = model.predict_one(**item)
        account, score = ranked[0]
        print(f"  {item['item_name']:<45} -> {account}  (score {score:.2f})")
        if not account:
            failures += 1
    print("self-test: OK" if failures == 0 else f"self-test: {failures} failure(s)")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify an expense line item into an account.")
    parser.add_argument("--item-name", help="item name / short description")
    parser.add_argument("--item-description", default="", help="longer description (optional)")
    parser.add_argument("--vendor", default="", help="vendor id (optional)")
    parser.add_argument("--amount", type=float, default=0.0, help="transaction amount")
    parser.add_argument("--retrain", action="store_true", help="retrain the model before predicting")
    parser.add_argument("--self-test", action="store_true", help="run the built-in smoke test and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    if not args.item_name:
        parser.error("provide --item-name, or use --self-test")

    model = get_model(retrain=args.retrain)
    ranked = model.predict_one(
        item_name=args.item_name,
        item_description=args.item_description,
        vendor_id=args.vendor,
        amount=args.amount,
    )
    account, score = ranked[0]
    print(f"{account}   (score {score:.2f})")
    runners_up = ", ".join(f"{name} ({s:.2f})" for name, s in ranked[1:])
    if runners_up:
        print(f"  next: {runners_up}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
