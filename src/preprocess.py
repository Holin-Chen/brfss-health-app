"""Phase 2 — Feature engineering for BRFSS 2024 diabetes risk model."""

import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = "data/brfss_survey_data_2024.csv"
FEATURES_PATH = "data/features.pkl"

# --- Refused/unknown sentinel values → NaN ---
REFUSED_CODES = {7, 9, 77, 99, 777, 999}

# --- 25 selected features ---
# Numeric/ordinal features (kept as-is after NaN handling)
NUMERIC_FEATURES = [
    "_BMI5",        # continuous BMI value
    "POORHLTH",     # days poor health limited activities (past 30)
    "MENTHLTH",     # days mental health not good (past 30)
    "PHYSHLTH",     # days physical health not good (past 30)
    "ALCDAY4",      # days had alcohol in past 30
    "_AGEG5YR",     # age group (1-13 ordinal)
    "CHILDREN",     # number of children in household
    "INCOME3",      # income level (ordinal)
    "EDUCA",        # education level (ordinal)
]

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = [
    "_BMI5CAT",     # BMI category
    "EXERANY2",     # any exercise past 30 days
    "SMOKDAY2",     # smoking frequency
    "GENHLTH",      # general health rating
    "CVDCRHD4",     # coronary heart disease
    "CHCOCNC1",     # any cancer
    "CHCCOPD3",     # COPD/emphysema
    "CHCKDNY2",     # kidney disease
    "ADDEPEV3",     # depression
    "HAVARTH4",     # arthritis
    "BLIND",        # blind or serious difficulty seeing
    "DECIDE",       # difficulty concentrating/deciding
    "DIFFWALK",     # difficulty walking
    "_RFHLTH",      # good or better health (computed)
    "_TOTINDA",     # no leisure-time physical activity
    "_RFSMOK3",     # current smoker (computed)
    "_RFBING6",     # binge drinker (computed)
]

TARGET_COL = "DIABETE4"

# 2024 BRFSS DIABETE4: 1=Yes, 2=Yes(pregnant only), 3=No, 4=No(pre-diabetes)
# at-risk = 1 only; not-at-risk = 2,3,4; 7,9 = NaN
TARGET_AT_RISK = {1}
TARGET_NOT_RISK = {2, 3, 4}


def encode_target(series: pd.Series) -> pd.Series:
    """Binary encode DIABETE4: 1=at-risk, 0=not-at-risk, NaN=excluded."""
    def _map(val):
        if pd.isna(val):
            return np.nan
        val = int(val)
        if val in TARGET_AT_RISK:
            return 1
        if val in TARGET_NOT_RISK:
            return 0
        return np.nan  # 7, 9
    return series.map(_map)


def clean_refused(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace refused/unknown sentinel codes with NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(~df[col].isin(REFUSED_CODES), other=np.nan)
    return df


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]):
    """Build sklearn ColumnTransformer for imputation + encoding."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return transformer


def main() -> None:
    print("Loading raw data...")
    all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    df = pd.read_csv(RAW_PATH, usecols=[c for c in all_cols if c != TARGET_COL] + [TARGET_COL],
                     low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows")

    # Drop rows with missing target
    y_raw = encode_target(df[TARGET_COL])
    mask = y_raw.notna()
    df = df[mask].copy()
    y = y_raw[mask].astype(int)
    print(f"After dropping missing target: {df.shape[0]:,} rows")
    print(f"Target distribution — at-risk (1): {y.sum():,} ({y.mean()*100:.1f}%)  "
          f"not-at-risk (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

    # Drop target from features
    df = df.drop(columns=[TARGET_COL])

    # Resolve actual present columns
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        print(f"Warning — columns not found, skipping: {missing}")

    # Clean refused/unknown codes
    df = clean_refused(df, num_cols + cat_cols)

    # Convert categoricals to string for OneHotEncoder
    for col in cat_cols:
        df[col] = df[col].astype("object")

    print(f"\nFeatures selected: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # Fit transform
    transformer = build_pipeline(num_cols, cat_cols)
    X = transformer.fit_transform(df[num_cols + cat_cols])

    # Build feature names
    cat_feature_names = transformer.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names

    print(f"Feature matrix shape after encoding: {X.shape}")

    # Save
    joblib.dump({
        "X": X,
        "y": y.values,
        "feature_names": feature_names,
        "transformer": transformer,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }, FEATURES_PATH)
    print(f"\nSaved features to {FEATURES_PATH}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    main()
