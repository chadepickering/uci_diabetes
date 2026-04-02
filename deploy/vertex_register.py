"""
vertex_register.py — Register the XGBoost model in Vertex AI Model Registry.

Usage:
    python deploy/vertex_register.py [--version v1] [--display-name "XGBoost v1"]

The inference container image must already exist in Artifact Registry.
See docker/ for build instructions; push with:
    docker tag  uci-diabetes-inference:latest ${INFERENCE_IMAGE}
    docker push ${INFERENCE_IMAGE}

Environment variables (from .env):
  GCP_PROJECT_ID   — GCP project ID
  GCP_REGION       — Vertex AI region (e.g. us-central1)
  GCS_BUCKET       — bucket holding the uploaded artifacts
  INFERENCE_IMAGE  — Artifact Registry image URI
                     (e.g. us-central1-docker.pkg.dev/PROJECT/repo/uci-diabetes-inference:v1)

Outputs:
  Prints "Registered model: <resource_name>" (parsed by the Airflow DAG via XCom).
  Writes resource name to deploy/model_resource_name.txt for ad-hoc use.
"""

import argparse
import json
import os
import pathlib

from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

REPO_ROOT     = pathlib.Path(__file__).resolve().parent.parent
_RESOURCE_OUT = pathlib.Path(__file__).parent / "model_resource_name.txt"


def register(version: str = "v1", display_name: str | None = None) -> str:
    project      = os.environ["GCP_PROJECT_ID"]
    region       = os.environ["GCP_REGION"]
    bucket       = os.environ["GCS_BUCKET"]
    image_uri    = os.environ["INFERENCE_IMAGE"]
    artifact_uri = f"gs://{bucket}/artifacts/{version}"
    display      = display_name or f"uci-diabetes-xgboost-{version}"

    # Attach validation AUC as a label for easy filtering in the registry
    metrics_path = REPO_ROOT / "ml" / "data" / "xgb_metrics.json"
    metrics      = json.loads(metrics_path.read_text())
    val_auc      = next(m["auc_roc"] for m in metrics if m["split"] == "validation")

    aiplatform.init(project=project, location=region)

    model = aiplatform.Model.upload(
        display_name=display,
        artifact_uri=artifact_uri,
        serving_container_image_uri=image_uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/ping",
        serving_container_environment_variables={
            "MODEL_DIR": "/app/model",
            "TAU": "0.12",
        },
        labels={
            "framework":    "xgboost",
            "version":      version,
            # Label values must be lowercase letters, digits, or hyphens
            "val_auc_roc":  f"{val_auc:.4f}".replace(".", "-"),
        },
    )

    resource_name = model.resource_name
    _RESOURCE_OUT.write_text(resource_name)

    print(f"Registered model: {resource_name}")
    print(f"  display_name : {display}")
    print(f"  artifact_uri : {artifact_uri}")
    print(f"  val AUC-ROC  : {val_auc:.4f}")
    return resource_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",      default="v1")
    parser.add_argument("--display-name", default=None)
    args = parser.parse_args()
    register(version=args.version, display_name=args.display_name)
