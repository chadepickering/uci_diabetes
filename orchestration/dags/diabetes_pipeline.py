"""
diabetes_pipeline.py — Full UCI Diabetes MLOps pipeline DAG.

Stages
------
1. preprocess      ml/sklearn/preprocess.py --dry-run
2. train_xgb       ml/sklearn/train_xgb.py  (XGB_N_TRIALS controls Optuna trials)
3. upload_gcs      deploy/gcs_upload.py
4. vertex_register deploy/vertex_register.py  (writes model resource name to XCom)
5. vertex_deploy   deploy/vertex_deploy.py    (reads resource name from XCom)
6. drift_monitor   monitoring/drift.py

Scheduling
----------
@monthly, catchup=False.  Trigger manually in the UI or via:
    airflow dags trigger diabetes_pipeline

Local setup
-----------
    cd orchestration
    echo "AIRFLOW_UID=$(id -u)" >> .env
    docker compose up airflow-init
    docker compose up -d
    # UI at http://localhost:8081  (admin / admin)

Environment variables forwarded from the host .env:
    GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET, INFERENCE_IMAGE,
    SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, XGB_N_TRIALS
"""

import os
import pathlib
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Repo root is bind-mounted at /opt/uci_diabetes in the Airflow container
REPO_ROOT = pathlib.Path("/opt/uci_diabetes")
VENV_PY   = REPO_ROOT / "venv_ml" / "bin" / "python"
PYTHON    = str(VENV_PY) if VENV_PY.exists() else sys.executable


def _run(script_path: str, *args: str, capture: bool = False) -> str | None:
    """Run a project script with the venv Python; returns stdout if capture=True."""
    cmd    = [PYTHON, str(REPO_ROOT / script_path)] + list(args)
    result = subprocess.run(cmd, capture_output=capture, text=True, check=True)
    return result.stdout if capture else None


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _preprocess() -> None:
    _run("ml/sklearn/preprocess.py", "--dry-run")


def _train_xgb() -> None:
    n_trials = os.environ.get("XGB_N_TRIALS", "3")  # 3 for local smoke; raise for prod
    _run("ml/sklearn/train_xgb.py", "--n-trials", n_trials)


def _upload_gcs() -> None:
    _run("deploy/gcs_upload.py", "--version", "v1")


def _vertex_register(**context) -> str:
    """Returns model resource name pushed to XCom automatically."""
    stdout = _run("deploy/vertex_register.py", "--version", "v1", capture=True)
    for line in (stdout or "").splitlines():
        if line.startswith("Registered model: "):
            return line.split("Registered model: ", 1)[1].strip()
    raise RuntimeError(
        f"Could not parse model resource name from vertex_register.py output:\n{stdout}"
    )


def _vertex_deploy(**context) -> None:
    model_name = context["ti"].xcom_pull(task_ids="vertex_register")
    _run("deploy/vertex_deploy.py", "--model-name", model_name)


def _drift_monitor() -> None:
    _run("monitoring/drift.py", "--n-batches", "4")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner":            "chad",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="diabetes_pipeline",
    description="UCI Diabetes readmission — end-to-end MLOps pipeline",
    schedule_interval="@monthly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["uci-diabetes", "mlops", "xgboost"],
) as dag:

    t_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=_preprocess,
        doc_md="Preprocess raw features (--dry-run). Writes train/val/holdout parquets.",
    )

    t_train = PythonOperator(
        task_id="train_xgb",
        python_callable=_train_xgb,
        doc_md="Train XGBoost with Optuna. XGB_N_TRIALS env var sets trial count.",
    )

    t_upload = PythonOperator(
        task_id="upload_gcs",
        python_callable=_upload_gcs,
        doc_md="Upload model artifacts to gs://GCS_BUCKET/artifacts/v1/.",
    )

    t_register = PythonOperator(
        task_id="vertex_register",
        python_callable=_vertex_register,
        doc_md="Register model in Vertex AI Model Registry. Pushes resource name to XCom.",
    )

    t_deploy = PythonOperator(
        task_id="vertex_deploy",
        python_callable=_vertex_deploy,
        doc_md="Deploy registered model to Vertex AI endpoint (idempotent).",
    )

    t_drift = PythonOperator(
        task_id="drift_monitor",
        python_callable=_drift_monitor,
        doc_md="PSI/KS/AUC drift monitoring on 4 temporal holdout batches.",
    )

    t_preprocess >> t_train >> t_upload >> t_register >> t_deploy >> t_drift
