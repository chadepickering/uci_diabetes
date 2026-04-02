"""
gcs_upload.py — Upload trained artifacts from ml/data/ to GCS.

Usage:
    python deploy/gcs_upload.py [--version v1]

Uploads to gs://{GCS_BUCKET}/artifacts/{VERSION}/:
  xgb_model.joblib
  preprocessor.joblib
  feature_names.txt
  xgb_metrics.json
  xgb_shap_importance.csv

Environment variables (from .env):
  GCS_BUCKET  — target bucket name (e.g. my-project-uci-diabetes-ml)
"""

import argparse
import os
import pathlib

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "ml" / "data"

ARTIFACTS = [
    "xgb_model.joblib",
    "preprocessor.joblib",
    "feature_names.txt",
    "xgb_metrics.json",
    "xgb_shap_importance.csv",
]


def upload(version: str = "v1") -> dict[str, str]:
    bucket_name = os.environ["GCS_BUCKET"]
    client  = storage.Client()
    bucket  = client.bucket(bucket_name)
    prefix  = f"artifacts/{version}"
    uris: dict[str, str] = {}

    for filename in ARTIFACTS:
        local_path = DATA_DIR / filename
        if not local_path.exists():
            raise FileNotFoundError(f"Missing artifact: {local_path}")
        blob_name = f"{prefix}/{filename}"
        bucket.blob(blob_name).upload_from_filename(str(local_path))
        uri = f"gs://{bucket_name}/{blob_name}"
        print(f"Uploaded {filename:<35} → {uri}")
        uris[filename] = uri

    print(f"\nAll artifacts at: gs://{bucket_name}/{prefix}/")
    return uris


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", help="Artifact version tag (default: v1)")
    args = parser.parse_args()
    upload(args.version)
