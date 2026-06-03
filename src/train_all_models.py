"""Train XGBoost models for diabetes, heart disease, and COPD."""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RAW_PATH = "data/brfss_survey_data_2024.csv"
REFUSED_CODES = {7, 9, 77, 99, 777, 999}

NUMERIC_FEATURES = [
    "_BMI5", "POORHLTH", "MENTHLTH", "PHYSHLTH", "ALCDAY4",
    "_AGEG5YR", "CHILDREN", "INCOME3", "EDUCA",
]
CATEGORICAL_FEATURES = [
    "_BMI5CAT", "EXERANY2", "SMOKDAY2", "GENHLTH", "CVDCRHD4",
    "CHCOCNC1", "CHCCOPD3", "CHCKDNY2", "ADDEPEV3", "HAVARTH4",
    "BLIND", "DECIDE", "DIFFWALK", "_RFHLTH", "_TOTINDA", "_RFSMOK3", "_RFBING6",
]

TARGETS = {
    "diabetes": {
        "col": "DIABETE4",
        # 1=Yes, 2=Yes(pregnant only), 3=No, 4=No(pre-diabetes)
        "at_risk": {1},
        "not_risk": {2, 3, 4},
        "model_path": "models/xgb_diabetes.pkl",
        "shap_path": "models/shap_diabetes.png",
    },
    "heart_disease": {
        "col": "CVDCRHD4",
        # 1=Yes, 2=No
        "at_risk": {1},
        "not_risk": {2},
        "model_path": "models/xgb_heart.pkl",
        "shap_path": "models/shap_heart.png",
    },
    "copd": {
        "col": "CHCCOPD3",
        # 1=Yes, 2=No
        "at_risk": {1},
        "not_risk": {2},
        "model_path": "models/xgb_copd.pkl",
        "shap_path": "models/shap_copd.png",
    },
}


def clean_refused(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(~s.isin(REFUSED_CODES), other=np.nan)


def encode_target(series: pd.Series, at_risk: set, not_risk: set) -> pd.Series:
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = int(val)
        if v in at_risk:
            return 1
        if v in not_risk:
            return 0
        return np.nan
    return series.map(_map)


def build_transformer(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])


def train_and_evaluate(name: str, cfg: dict, df_raw: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Training: {name.upper()}")

    target_col = cfg["col"]

    # Build feature columns excluding current target to avoid leakage
    cat_cols = [c for c in CATEGORICAL_FEATURES
                if c in df_raw.columns and c != target_col]
    num_cols = [c for c in NUMERIC_FEATURES if c in df_raw.columns]

    use_cols = num_cols + cat_cols + [target_col]
    df = df_raw[use_cols].copy()

    # Encode target
    y_raw = encode_target(df[target_col], cfg["at_risk"], cfg["not_risk"])
    mask = y_raw.notna()
    df = df[mask].copy()
    y = y_raw[mask].astype(int)
    print(f"Rows: {len(y):,}  |  at-risk: {y.sum():,} ({y.mean()*100:.1f}%)")

    df = df.drop(columns=[target_col])

    # Clean refused codes
    for col in num_cols + cat_cols:
        if col in df.columns:
            df[col] = clean_refused(df[col])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    transformer = build_transformer(num_cols, cat_cols)
    X = transformer.fit_transform(df[num_cols + cat_cols])

    cat_names = transformer.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_names

    # Cross-validation
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_res = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "recall"], n_jobs=1)
    print(f"CV ROC-AUC: {cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}")
    print(f"CV Recall : {cv_res['test_recall'].mean():.4f} ± {cv_res['test_recall'].std():.4f}")

    # Full fit
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"Train ROC-AUC: {roc_auc_score(y, y_prob):.4f}")

    # SHAP summary
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(3000, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X[sample_idx])
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X[sample_idx], feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(cfg["shap_path"], dpi=120, bbox_inches="tight")
    plt.close()

    # Save model + transformer
    joblib.dump(model, cfg["model_path"])
    joblib.dump({
        "transformer": transformer,
        "feature_names": feature_names,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }, cfg["model_path"].replace(".pkl", "_transformer.pkl"))
    print(f"Saved: {cfg['model_path']}")


def build_dashboard_csv(df_raw: pd.DataFrame) -> None:
    """Save slim CSV with all columns needed for dashboard."""
    RACE_MAP = {
        1: "White", 2: "Black/African American", 3: "American Indian/Alaska Native",
        4: "Asian", 5: "Native Hawaiian/Pacific Islander", 6: "Other",
        7: "Multiracial", 8: "Hispanic", 9: np.nan,
    }
    SEX_MAP = {1: "Male", 2: "Female"}
    INCOME_MAP = {
        1: "<$15k", 2: "$15k-$20k", 3: "$20k-$25k", 4: "$25k-$35k",
        5: "$35k-$50k", 6: "$50k-$75k", 7: "$75k-$100k", 8: "$100k-$150k",
        9: "$150k-$200k", 10: "$200k+", 11: "Don't know/Refused",
        77: np.nan, 99: np.nan,
    }
    EDUCA_MAP = {
        1: "No school", 2: "Elementary", 3: "Some high school",
        4: "High school graduate", 5: "Some college", 6: "College graduate",
        9: np.nan,
    }

    keep = ["DIABETE4", "CVDCRHD4", "CHCCOPD3", "_STATE",
            "_AGEG5YR", "_BMI5CAT", "EXERANY2", "SEXVAR", "_RACE", "INCOME3", "EDUCA"]
    df = df_raw[[c for c in keep if c in df_raw.columns]].copy()

    df["SEXVAR"]   = pd.to_numeric(df["SEXVAR"],   errors="coerce").map(SEX_MAP)
    df["_RACE"]    = pd.to_numeric(df["_RACE"],    errors="coerce").map(RACE_MAP)
    df["INCOME3"]  = pd.to_numeric(df["INCOME3"],  errors="coerce").map(INCOME_MAP)
    df["EDUCA"]    = pd.to_numeric(df["EDUCA"],    errors="coerce").map(EDUCA_MAP)

    # Decode disease targets
    df["CVDCRHD4"] = pd.to_numeric(df["CVDCRHD4"], errors="coerce").map(
        {1: "Yes", 2: "No", 7: np.nan, 9: np.nan})
    df["CHCCOPD3"] = pd.to_numeric(df["CHCCOPD3"], errors="coerce").map(
        {1: "Yes", 2: "No", 7: np.nan, 9: np.nan})
    df["DIABETE4"] = pd.to_numeric(df["DIABETE4"], errors="coerce").map(
        {1: "Yes", 2: "Yes_pregnant_only", 3: "No", 4: "No_prediabetes", 7: np.nan, 9: np.nan})

    df.to_csv("data/dashboard.csv", index=False)
    print(f"\nRebuilt dashboard.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Size: {__import__('os').path.getsize('data/dashboard.csv')/1e6:.1f} MB")


def main():
    print("Loading raw data in chunks…")
    all_cols = list(set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES +
        [cfg["col"] for cfg in TARGETS.values()] +
        ["SEXVAR", "_RACE", "INCOME3", "EDUCA", "_STATE", "_AGEG5YR", "_BMI5CAT", "EXERANY2"]
    ))
    chunks = []
    for chunk in pd.read_csv(RAW_PATH, usecols=all_cols, low_memory=False, chunksize=50000):
        chunks.append(chunk)
    df_raw = pd.concat(chunks, ignore_index=True)
    print(f"Loaded: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    for name, cfg in TARGETS.items():
        train_and_evaluate(name, cfg, df_raw)

    build_dashboard_csv(df_raw)
    print("\nAll done.")


if __name__ == "__main__":
    main()
