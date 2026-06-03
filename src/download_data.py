"""Download pre-processed app files from Hugging Face Hub."""

import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO = "holinchen/brfss-health-app-data"

FILES = {
    "data/dashboard.csv":                    "dashboard.csv",
    "models/model_comparison.json":          "model_comparison.json",
    "models/xgb_diabetes.pkl":               "xgb_diabetes.pkl",
    "models/xgb_diabetes_transformer.pkl":   "xgb_diabetes_transformer.pkl",
    "models/xgb_heart.pkl":                  "xgb_heart.pkl",
    "models/xgb_heart_transformer.pkl":      "xgb_heart_transformer.pkl",
    "models/xgb_copd.pkl":                   "xgb_copd.pkl",
    "models/xgb_copd_transformer.pkl":       "xgb_copd_transformer.pkl",
}


def download_all() -> None:
    """Download all files from HF Hub, using HF's ETag cache to avoid re-downloading unchanged files."""
    for local_path, repo_filename in FILES.items():
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Checking {repo_filename}…")
        # hf_hub_download uses ETags: only fetches if remote file changed
        cached = hf_hub_download(
            repo_id=HF_REPO,
            filename=repo_filename,
            repo_type="dataset",
        )
        # Copy from HF cache to expected local path
        if not path.exists() or path.stat().st_size != Path(cached).stat().st_size:
            shutil.copy2(cached, path)
            print(f"  Updated {local_path}")
        else:
            print(f"  Up to date")


# Keep old name so app.py import still works
download_if_missing = download_all


if __name__ == "__main__":
    download_all()
