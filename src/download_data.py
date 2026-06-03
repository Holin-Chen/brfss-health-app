"""Download pre-processed app files from Hugging Face Hub if not present."""

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


def download_if_missing() -> None:
    for local_path, repo_filename in FILES.items():
        path = Path(local_path)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_filename} from Hugging Face…")
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=repo_filename,
            repo_type="dataset",
            local_dir=str(path.parent),
        )
        print(f"Saved to {downloaded}")


if __name__ == "__main__":
    download_if_missing()
