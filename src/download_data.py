"""Download BRFSS 2024 data from Kaggle if not already present."""

import json
import os
from pathlib import Path


DATASET_OWNER = "rudritarahman"
DATASET_NAME = "cdc-brfss-survey-data-2024"
DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "brfss_survey_data_2024.csv"


def setup_kaggle_credentials() -> None:
    """Write kaggle.json from Streamlit secrets when running in the cloud."""
    kaggle_dir = Path.home() / ".kaggle"
    creds_path = kaggle_dir / "kaggle.json"
    if creds_path.exists():
        return  # already configured (local dev)

    try:
        import streamlit as st
        username = st.secrets["kaggle"]["username"]
        key = st.secrets["kaggle"]["key"]
    except Exception as e:
        raise RuntimeError(
            "Kaggle credentials not found. Add [kaggle] username/key to Streamlit secrets."
        ) from e

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps({"username": username, "key": key}))
    creds_path.chmod(0o600)
    # Tell the kaggle library where to find credentials
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)


def download_if_missing() -> None:
    """Download and unzip dataset from Kaggle if CSV is not present."""
    if RAW_CSV.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    setup_kaggle_credentials()

    # Import after credentials are written so the library picks them up
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()
    api.authenticate()

    print(f"Downloading {DATASET_OWNER}/{DATASET_NAME} from Kaggle…")
    api.dataset_download_files(
        f"{DATASET_OWNER}/{DATASET_NAME}",
        path=str(DATA_DIR),
        unzip=True,
        quiet=False,
    )
    print("Download complete.")


if __name__ == "__main__":
    download_if_missing()
