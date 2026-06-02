"""Download BRFSS 2024 data from Kaggle if not already present."""

import os
import json
import subprocess
import sys
from pathlib import Path


DATASET = "rudritarahman/cdc-brfss-survey-data-2024"
DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "brfss_survey_data_2024.csv"


def setup_kaggle_credentials() -> bool:
    """Write kaggle.json from Streamlit secrets if running in the cloud."""
    try:
        import streamlit as st
        username = st.secrets["kaggle"]["username"]
        key = st.secrets["kaggle"]["key"]
    except Exception:
        return False  # running locally — credentials already on disk

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds_path = kaggle_dir / "kaggle.json"
    creds_path.write_text(json.dumps({"username": username, "key": key}))
    creds_path.chmod(0o600)
    return True


def download_if_missing() -> None:
    """Download and unzip dataset from Kaggle if CSV is not present."""
    if RAW_CSV.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    setup_kaggle_credentials()

    print("Downloading BRFSS 2024 dataset from Kaggle…")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         DATASET, "-p", str(DATA_DIR), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")
    print("Download complete.")


if __name__ == "__main__":
    download_if_missing()
