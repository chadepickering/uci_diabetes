"""
vertex_batch_predict.py — Batch prediction on holdout set via Vertex AI.

Simulates production traffic by feeding holdout encounters in temporal order
(sorted by encounter_id) through the deployed model in N batches, representing
quarterly scoring periods.

Workflow
--------
1. Load raw_features.parquet; filter to holdout; sort by encounter_id
2. Divide into N temporal batches
3. Upload each batch as JSONL to GCS  (gs://BUCKET/data/VERSION/batch_NN/input.jsonl)
4. Submit a Vertex AI BatchPredictionJob for each batch
5. Print output URIs (consumed by drift monitoring)

Usage:
    python deploy/vertex_batch_predict.py [--n-batches 4] [--version v1]

Environment variables (from .env):
  GCP_PROJECT_ID
  GCP_REGION
  GCS_BUCKET
  INFERENCE_IMAGE  — container image used by the batch prediction job

The batch job uses the same inference container as the online endpoint, so
predictions are identical in both serving paths.
"""

import argparse
import os
import pathlib

import pandas as pd
from dotenv import load_dotenv
from google.cloud import aiplatform, storage

load_dotenv()

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "ml" / "data"

# Columns not sent to the inference API (same set as preprocess.py DROP + NZV)
_DROP_COLS: set[str] = {
    "encounter_id", "patient_nbr", "readmitted_raw", "readmitted_any",
    "split_group", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "readmitted_30day",
    "glimepiride_pioglitazone", "metformin_pioglitazone",
    "metformin_rosiglitazone", "glipizide_metformin", "tolbutamide",
    "miglitol", "tolazamide", "chlorpropamide", "acarbose",
    "glyburide_metformin", "nateglinide",
}


def _upload_jsonl(df: pd.DataFrame, gcs_uri: str, client: storage.Client) -> None:
    bucket_name, _, blob_path = gcs_uri.replace("gs://", "").partition("/")
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(
        df.to_json(orient="records", lines=True),
        content_type="application/jsonl",
    )
    print(f"  Uploaded {len(df):,} rows → {gcs_uri}")


def run_batch_predict(n_batches: int = 4, version: str = "v1") -> list[str]:
    project     = os.environ["GCP_PROJECT_ID"]
    region      = os.environ["GCP_REGION"]
    bucket_name = os.environ["GCS_BUCKET"]
    image_uri   = os.environ["INFERENCE_IMAGE"]

    # Load model resource name written by vertex_register.py
    resource_file = pathlib.Path(__file__).parent / "model_resource_name.txt"
    if not resource_file.exists():
        raise FileNotFoundError(
            "deploy/model_resource_name.txt not found. "
            "Run vertex_register.py first."
        )
    model_resource_name = resource_file.read_text().strip()

    aiplatform.init(project=project, location=region)
    gcs_client = storage.Client()

    # Load raw holdout features, sorted by encounter_id for temporal ordering
    raw     = pd.read_parquet(DATA_DIR / "raw_features.parquet")
    holdout = (
        raw[raw["split_group"] == "holdout"]
        .sort_values("encounter_id")
        .reset_index(drop=True)
    )
    feature_cols = [c for c in holdout.columns if c not in _DROP_COLS]

    print(f"Holdout: {len(holdout):,} rows  |  Batches: {n_batches}")

    batch_size   = len(holdout) // n_batches
    output_uris: list[str] = []

    for i in range(n_batches):
        start     = i * batch_size
        end       = (i + 1) * batch_size if i < n_batches - 1 else len(holdout)
        batch_df  = holdout.iloc[start:end][feature_cols]
        label     = f"batch_{i+1:02d}"
        input_uri = f"gs://{bucket_name}/data/{version}/{label}/input.jsonl"
        out_uri   = f"gs://{bucket_name}/predictions/{version}/{label}/"

        print(f"\nBatch {i+1}/{n_batches}: encounters {start}–{end-1}")
        _upload_jsonl(batch_df, input_uri, gcs_client)

        job = aiplatform.BatchPredictionJob.create(
            job_display_name=f"uci-diabetes-{version}-{label}",
            model_name=model_resource_name,
            instances_format="jsonl",
            predictions_format="jsonl",
            gcs_source=input_uri,
            gcs_destination_prefix=out_uri,
            machine_type="n1-standard-2",
        )
        print(f"  Job: {job.resource_name}")
        output_uris.append(out_uri)

    print(f"\nAll {n_batches} batch jobs submitted.")
    print("Output URIs:")
    for uri in output_uris:
        print(f"  {uri}")

    return output_uris


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--version",   default="v1")
    args = parser.parse_args()
    run_batch_predict(n_batches=args.n_batches, version=args.version)
