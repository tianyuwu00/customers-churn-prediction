import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

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
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    train["Churn"] = train["Churn"].map({"Yes": 1, "No": 0})
    return train, test


# ---------------------------------------------------------------------------
# Feature engineering
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def split_data(
    train: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = train.drop(columns=["Churn", "id"])
    y = train["Churn"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"Churn rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")
    return X_train, X_val, y_train, y_val


def tune_hyperparams(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    n_trials: int = 30,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "seed": SEED,
            "feature_pre_filter": False,
            "learning_rate": 0.05,
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(-1)],
        )

        trial.set_user_attr("best_iteration", model.best_iteration)
        return model.best_score["valid_0"]["auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


def train_final_model(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    best_params: dict,
    num_boost_round: int,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "seed": SEED,
        "learning_rate": 0.05,
        **best_params,
    }
    lgb_full = lgb.Dataset(X_full, label=y_full)
    model = lgb.train(params, lgb_full, num_boost_round=num_boost_round)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
) -> float:
    val_preds = model.predict(X_val)
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc:.4f}")

    cm = confusion_matrix(y_val, (val_preds >= threshold).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot()

    import matplotlib.pyplot as plt
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=100)
    plt.close()
    print("Confusion matrix saved.")

    return auc


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_model(model: lgb.Booster, best_params: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lgb_model_v1.joblib")
    params_path = os.path.join(MODEL_DIR, "best_params_lgb_v1.json")

    joblib.dump(model, model_path)
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Params saved to {params_path}")




# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Loading data ===")
    train_df, test_df = load_data()

    print("\n=== Feature engineering ===")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    print("\n=== Splitting data ===")
    X_train, X_val, y_train, y_val = split_data(train_df)
    X_train = cast_categoricals(X_train)
    X_val = cast_categoricals(X_val)

    print("\n=== Hyperparameter tuning ===")
    study = tune_hyperparams(X_train, X_val, y_train, y_val, n_trials=30)

    print("\n=== Evaluating best trial model ===")
    # Train a model with best params for evaluation
    eval_params = {
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "seed": SEED,
        "learning_rate": 0.05,
        **study.best_params,
    }
    lgb_train_ds = lgb.Dataset(X_train, label=y_train)
    lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_train_ds)
    eval_model = lgb.train(
        eval_params,
        lgb_train_ds,
        num_boost_round=study.best_trial.user_attrs["best_iteration"],
        valid_sets=[lgb_val_ds],
        callbacks=[lgb.log_evaluation(100)],
    )
    evaluate(eval_model, X_val, y_val)

    print("\n=== Retraining on full data ===")
    X_full = cast_categoricals(
        pd.concat([X_train, X_val]).reset_index(drop=True)
    )
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    final_model = train_final_model(
        X_full, y_full,
        best_params=study.best_params,
        num_boost_round=study.best_trial.user_attrs["best_iteration"],
    )

    print("\n=== Saving model ===")
    save_model(final_model, study.best_params)

    print("\nDone.")


if __name__ == "__main__":
    main()
