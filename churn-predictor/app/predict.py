"""
predict.py — run inference with a saved LightGBM churn model.

Usage
-----
# Predict on a CSV file and write results to stdout / output file:
    python predict.py --input data/test.csv
    python predict.py --input data/test.csv --output predictions.csv
    python predict.py --input data/test.csv --threshold 0.4 --proba

# Use a different model:
    python predict.py --input data/test.csv --model models/lgb_model_v1.joblib
"""

import argparse
import os
import sys

import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_model_v1.joblib")

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_bucket",
]


# ---------------------------------------------------------------------------
# Preprocessing (mirrors train.py)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ADDON_COLS:
        df[col] = df[col].replace("No internet service", "No").map({"Yes": 1, "No": 0})

    df["num_addons"] = df[ADDON_COLS].sum(axis=1)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["new", "mid", "long"],
    )

    df["has_family"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    df["is_auto_payment"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    drop_cols = [c for c in ["id", "Churn"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return cast_categoricals(df)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run train.py first to generate a model."
        )
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    df: pd.DataFrame,
    model,
    threshold: float = 0.5,
    return_proba: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with columns [id (if present), churn_proba, Churn]."""
    has_id = "id" in df.columns
    ids = df["id"].reset_index(drop=True) if has_id else None

    X = preprocess(df)
    proba = model.predict(X)

    result = pd.DataFrame({"churn_proba": proba})
    if not return_proba:
        result["Churn"] = (proba >= threshold).astype(int)

    if ids is not None:
        result.insert(0, "id", ids)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn model inference")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--output", default=None,
        help="Path to write predictions CSV (default: print to stdout)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"Path to model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for binary label (default: 0.5)",
    )
    parser.add_argument(
        "--proba", action="store_true",
        help="Output churn probability only, skip binary label column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from '{args.model}' ...")
    model = load_model(args.model)

    print(f"Reading input from '{args.input}' ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows loaded")

    results = predict(df, model, threshold=args.threshold, return_proba=args.proba)

    if not args.proba:
        churn_rate = results["Churn"].mean()
        print(f"Predicted churn rate: {churn_rate:.3f} (threshold={args.threshold})")

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to '{args.output}'")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
