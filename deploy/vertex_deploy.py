"""
vertex_deploy.py — Deploy a registered Vertex AI model to an endpoint.

Idempotent: reuses an existing endpoint with the same display name if found.

Usage:
    python deploy/vertex_deploy.py --model-name RESOURCE_NAME
    python deploy/vertex_deploy.py --model-name "$(cat deploy/model_resource_name.txt)"

RESOURCE_NAME: full Vertex AI model resource name from vertex_register.py
               (projects/PROJECT/locations/REGION/models/MODEL_ID)

Environment variables (from .env):
  GCP_PROJECT_ID
  GCP_REGION

Outputs:
  Prints endpoint resource name.
  Writes deploy/endpoint.json with endpoint + model resource names.
"""

import argparse
import json
import os
import pathlib

from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

_ENDPOINT_OUT = pathlib.Path(__file__).parent / "endpoint.json"

_ENDPOINT_DISPLAY_NAME = "uci-diabetes-endpoint"
_MACHINE_TYPE          = "n1-standard-2"


def deploy(model_resource_name: str) -> str:
    project = os.environ["GCP_PROJECT_ID"]
    region  = os.environ["GCP_REGION"]

    aiplatform.init(project=project, location=region)

    # Reuse existing endpoint or create a new one
    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{_ENDPOINT_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if existing:
        endpoint = existing[0]
        print(f"Reusing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=_ENDPOINT_DISPLAY_NAME)
        print(f"Created endpoint: {endpoint.resource_name}")

    model = aiplatform.Model(model_name=model_resource_name)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="xgboost-tau012",
        machine_type=_MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
    )

    info = {
        "endpoint_resource_name": endpoint.resource_name,
        "model_resource_name":    model_resource_name,
    }
    _ENDPOINT_OUT.write_text(json.dumps(info, indent=2))
    print(f"Deployed. Endpoint info → {_ENDPOINT_OUT}")
    return endpoint.resource_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", required=True,
        help="Vertex AI model resource name (from vertex_register.py)",
    )
    args = parser.parse_args()
    deploy(model_resource_name=args.model_name)
