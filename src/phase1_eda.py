"""Phase 1 — EDA & data cleaning for BRFSS 2024."""

import pandas as pd
import numpy as np

RAW_PATH = "data/brfss_survey_data_2024.csv"
CLEAN_PATH = "data/brfss_clean.csv"

# BRFSS codebook decode maps
DECODE_MAPS = {
    # 2024 BRFSS DIABETE4: 1=Yes, 2=Yes(pregnant only), 3=No, 4=No(pre-diabetes/borderline)
    "DIABETE4": {
        1: "Yes", 2: "Yes_pregnant_only", 3: "No", 4: "No_prediabetes",
        7: np.nan, 9: np.nan,
    },
    # BPHIGH6 renamed to BPHIGH4 in 2024 dataset
    "BPHIGH4": {
        1: "Yes", 2: "Yes_pregnant", 3: "No",
        7: np.nan, 9: np.nan,
    },
    "CVDCRHD4": {
        1: "Yes", 2: "No",
        7: np.nan, 9: np.nan,
    },
    "SMOKDAY2": {
        1: "Every_day", 2: "Some_days", 3: "Not_at_all",
        7: np.nan, 9: np.nan,
    },
    "_AGEG5YR": {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
        6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
        11: "70-74", 12: "75-79", 13: "80+", 14: np.nan,
    },
    "_BMI5CAT": {
        1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese",
    },
    "EXERANY2": {
        1: "Yes", 2: "No",
        7: np.nan, 9: np.nan,
    },
    "_STATE": None,  # keep numeric FIPS as-is for choropleth
}


def load_and_inspect(path: str) -> pd.DataFrame:
    """Load CSV and print basic shape and missing value info."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.5]
    print(f"\nColumns with >50% missing: {len(high_missing)}")
    print(high_missing.head(10).to_string())

    key_cols = list(DECODE_MAPS.keys())
    present = [c for c in key_cols if c in df.columns]
    missing_cols = [c for c in key_cols if c not in df.columns]
    print(f"\nKey columns present: {present}")
    if missing_cols:
        print(f"Key columns NOT found: {missing_cols}")

    return df


def decode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode BRFSS numeric codes to human-readable labels."""
    df = df.copy()
    for col, mapping in DECODE_MAPS.items():
        if col not in df.columns:
            continue
        if mapping is None:
            continue
        df[col] = df[col].map(mapping)
        print(f"  Decoded {col}: {df[col].value_counts(dropna=False).to_dict()}")
    return df


def main() -> None:
    df = load_and_inspect(RAW_PATH)

    print("\n--- Decoding columns ---")
    df_clean = decode_columns(df)

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned data to {CLEAN_PATH}")
    print(f"Final shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

    # Quick summary of target variable
    if "DIABETE4" in df_clean.columns:
        print("\nDIABETE4 distribution:")
        print(df_clean["DIABETE4"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
