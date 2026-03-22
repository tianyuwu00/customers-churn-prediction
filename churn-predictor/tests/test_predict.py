"""
Tests for predict.py

Run with:
    pytest tests/test_predict.py -v
"""

import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from app import predict


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RAW_COLUMNS = {
    "id": [0, 1, 2, 3, 4],
    "gender": ["Male", "Female", "Male", "Female", "Male"],
    "SeniorCitizen": [0, 1, 0, 0, 1],
    "Partner": ["Yes", "No", "Yes", "No", "Yes"],
    "Dependents": ["No", "No", "Yes", "No", "No"],
    "tenure": [1, 24, 50, 12, 72],
    "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
    "OnlineSecurity": ["Yes", "No", "No internet service", "No internet service", "No"],
    "OnlineBackup": ["No", "Yes", "No internet service", "No internet service", "Yes"],
    "DeviceProtection": ["Yes", "No", "No internet service", "No internet service", "No"],
    "TechSupport": ["No", "No", "No internet service", "No internet service", "Yes"],
    "StreamingTV": ["No", "Yes", "No internet service", "No internet service", "Yes"],
    "StreamingMovies": ["Yes", "No", "No internet service", "No internet service", "No"],
    "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
    "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Mailed check",
        "Electronic check",
    ],
    "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
    "TotalCharges": [29.85, 1889.50, 108.15, 1840.75, 151.65],
}


def make_raw_df(with_id: bool = True, with_churn: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(RAW_COLUMNS)
    if not with_id:
        df = df.drop(columns=["id"])
    if with_churn:
        df["Churn"] = [0, 1, 0, 1, 0]
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def tiny_model(tmp_path):
    """Train a minimal LightGBM model on synthetic data and save it."""
    df = make_raw_df(with_churn=True)
    # Duplicate rows so LightGBM has enough data to train
    df = pd.concat([df] * 20, ignore_index=True)
    df["Churn"] = np.random.default_rng(42).integers(0, 2, len(df))

    X = predict.preprocess(df)
    y = df["Churn"].reset_index(drop=True)

    lgb_ds = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 4}
    model = lgb.train(params, lgb_ds, num_boost_round=10)

    model_path = tmp_path / "lgb_model_test.joblib"
    joblib.dump(model, model_path)
    return model, str(model_path)


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_addon_cols_encoded(self, raw_df):
        out = predict.engineer_features(raw_df)
        for col in predict.ADDON_COLS:
            assert out[col].isin([0, 1]).all(), f"{col} should be 0/1"

    def test_no_internet_service_becomes_zero(self, raw_df):
        out = predict.engineer_features(raw_df)
        # row 2 and 3 had "No internet service" for all addon cols
        assert out.loc[2, "OnlineSecurity"] == 0
        assert out.loc[3, "StreamingTV"] == 0

    def test_num_addons_range(self, raw_df):
        out = predict.engineer_features(raw_df)
        assert out["num_addons"].between(0, len(predict.ADDON_COLS)).all()

    def test_tenure_bucket_labels(self, raw_df):
        out = predict.engineer_features(raw_df)
        assert set(out["tenure_bucket"].dropna().astype(str)).issubset({"new", "mid", "long"})
        # tenure=1 → new, tenure=24 → mid, tenure=72 → long
        assert str(out.loc[0, "tenure_bucket"]) == "new"
        assert str(out.loc[1, "tenure_bucket"]) == "mid"
        assert str(out.loc[4, "tenure_bucket"]) == "long"

    def test_has_family(self, raw_df):
        out = predict.engineer_features(raw_df)
        # row 0: Partner=Yes → 1
        assert out.loc[0, "has_family"] == 1
        # row 1: Partner=No, Dependents=No → 0
        assert out.loc[1, "has_family"] == 0
        # row 2: Partner=Yes OR Dependents=Yes → 1
        assert out.loc[2, "has_family"] == 1

    def test_is_auto_payment(self, raw_df):
        out = predict.engineer_features(raw_df)
        # row 1: Bank transfer (automatic) → 1
        assert out.loc[1, "is_auto_payment"] == 1
        # row 2: Credit card (automatic) → 1
        assert out.loc[2, "is_auto_payment"] == 1
        # row 0: Electronic check → 0
        assert out.loc[0, "is_auto_payment"] == 0

    def test_original_df_not_mutated(self, raw_df):
        original_cols = raw_df.columns.tolist()
        predict.engineer_features(raw_df)
        assert raw_df.columns.tolist() == original_cols
        assert raw_df["OnlineSecurity"].iloc[0] != 0  # still raw string


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_drops_id_and_churn(self):
        df = make_raw_df(with_id=True, with_churn=True)
        out = predict.preprocess(df)
        assert "id" not in out.columns
        assert "Churn" not in out.columns

    def test_cat_cols_are_category_dtype(self, raw_df):
        out = predict.preprocess(raw_df)
        for col in predict.CAT_COLS:
            assert out[col].dtype.name == "category", f"{col} should be category"

    def test_output_shape(self, raw_df):
        out = predict.preprocess(raw_df)
        # original 20 cols - id + 5 engineered - 6 addon (now numeric, kept) = 23
        assert out.shape[0] == len(raw_df)
        assert out.shape[1] > 0


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run train.py"):
            predict.load_model(str(tmp_path / "nonexistent.joblib"))

    def test_loads_saved_model(self, tiny_model):
        _, model_path = tiny_model
        model = predict.load_model(model_path)
        assert model is not None


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_columns_default(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model)
        assert list(result.columns) == ["id", "churn_proba", "Churn"]

    def test_output_columns_proba_only(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model, return_proba=True)
        assert "Churn" not in result.columns
        assert "churn_proba" in result.columns

    def test_output_rows_match_input(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model)
        assert len(result) == len(raw_df)

    def test_proba_in_unit_interval(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model)
        assert result["churn_proba"].between(0.0, 1.0).all()

    def test_binary_label_is_0_or_1(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model)
        assert result["Churn"].isin([0, 1]).all()

    def test_threshold_all_churn(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model, threshold=0.0)
        assert result["Churn"].eq(1).all()

    def test_threshold_no_churn(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model, threshold=1.0)
        assert result["Churn"].eq(0).all()

    def test_no_id_column(self, tiny_model):
        model, _ = tiny_model
        df = make_raw_df(with_id=False)
        result = predict.predict(df, model)
        assert "id" not in result.columns

    def test_id_values_preserved(self, raw_df, tiny_model):
        model, _ = tiny_model
        result = predict.predict(raw_df, model)
        assert list(result["id"]) == list(raw_df["id"])


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_predict_to_output_file(self, raw_df, tiny_model, tmp_path):
        _, model_path = tiny_model
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        raw_df.to_csv(input_path, index=False)

        sys.argv = [
            "predict.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", model_path,
        ]
        predict.main()

        assert output_path.exists()
        out = pd.read_csv(output_path)
        assert "Churn" in out.columns
        assert len(out) == len(raw_df)

    def test_proba_flag(self, raw_df, tiny_model, tmp_path):
        _, model_path = tiny_model
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        raw_df.to_csv(input_path, index=False)

        sys.argv = [
            "predict.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", model_path,
            "--proba",
        ]
        predict.main()

        out = pd.read_csv(output_path)
        assert "Churn" not in out.columns
        assert "churn_proba" in out.columns

    def test_custom_threshold(self, raw_df, tiny_model, tmp_path):
        _, model_path = tiny_model
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        raw_df.to_csv(input_path, index=False)

        sys.argv = [
            "predict.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", model_path,
            "--threshold", "0.0",
        ]
        predict.main()

        out = pd.read_csv(output_path)
        assert out["Churn"].eq(1).all()
