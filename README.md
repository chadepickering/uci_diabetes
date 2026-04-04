# Predicting Diabetes-Related Hospital Readmission — End-to-End ML Pipeline

Predicting 30-day hospital readmission in diabetic inpatients using the UCI 130-US Hospitals dataset (1999–2008). This project covers the full lifecycle from raw data ingestion to a deployed REST inference endpoint with drift monitoring — built as a comprehensive demonstration of a production-grade ML stack.

---

## Objective

Unplanned 30-day readmissions in diabetic patients are costly for hospitals and harmful for patients. The objective is to flag high-risk encounters at discharge so care teams can intervene; not to predict with certainty, but to rank risk well enough to allocate limited resources wisely. This translates to an AUC-optimised binary classifier with a deployment threshold calibrated for operational flag rates.

---

## Dataset

**Source**: UCI 130-US Hospitals Diabetes dataset — 101,766 encounters across 130 US hospitals, 1999–2008.

**After deduplication and exclusions**: 69,581 encounters (one per patient, keeping the highest-acuity encounter).

**Temporal split by `encounter_id` percentile** (no explicit date column; encounter ID is a proxy for time):

| Split | N | Positive rate |
|---|---|---|
| Train | 48,706 (70%) | 7.1% |
| Validation | 6,958 (10%) | 5.6% |
| Holdout | 13,917 (20%) | 5.6% |

The positive rate drop from train to validation/holdout is a documented temporal artifact: earlier encounters had more aggressive readmission coding. All drift analysis uses the training distribution as reference.

**Note:** PySpark was used for data ingestion rather than pandas for architectural reasons: the pipeline is designed to scale to hospital-system-sized EHR data where distributed processing would be necessary. For datasets of this size, `snowflake-connector-python` with pandas (`write_pandas()`) is a simpler and equally valid alternative. See `ingestion/ingest_to_snowflake.py` for the PySpark implementation.

---

## Repository Layout

```
├── ingestion/          PySpark ingestion → Snowflake
├── dbt/                dbt transformations (staging and marts layer)
├── ml/
│   ├── sklearn/        Logistic regression baseline, XGBoost (ultimately chosen), LightGBM
│       ├── preprocess.py       Snowflake → sklearn ColumnTransformer → parquet splits
│   └── tensorflow/     Feedforward NN (MLP)
├── monitoring/
│   └── drift.py        PSI / KS / AUC drift monitoring on holdout batches
├── deploy/
│   ├── gcs_upload.py         Upload model artifacts to GCS
│   ├── vertex_register.py    Register model in Vertex AI Model Registry
│   ├── vertex_deploy.py      Deploy to Vertex AI endpoint
│   ├── vertex_batch_predict.py  Batch inference on holdout batches
│   └── DEPLOYMENT_LESSONS.md   Apple Silicon → Vertex AI pitfalls
├── docker/
│   ├── base/           Shared base image (scikit-learn, XGBoost, pandas)
│   ├── train/          Training container (+ LightGBM, TF, Optuna, SHAP)
│   └── inference/      FastAPI serving container (+ google-cloud-storage)
├── orchestration/
│   ├── dags/           Airflow DAG: preprocess → train → upload → register → deploy → monitor
│   └── docker-compose.yml  Local Airflow via Docker Compose
├── reports/
│   ├── part1/          R/Quarto statistical analysis book
│   └── part2/          Python/Quarto ML evaluation/monitoring book
└── docs/               Rendered HTML output (GitHub Pages)
```

---

## Part 1 — Statistical Analysis

Rendered book: `docs/part1/`
Stack: **R 4.3 · Quarto · RJDBC (Snowflake)**

### Chapter 1 — Exploratory Data Analysis
Demographics, utilisation patterns, missingness, near-zero-variance features, and correlation structure across 55 candidate features. Identified 11 NZV medication columns for removal and documented the expected temporal positive-rate shift between training and holdout.

### Chapter 2 — Causal Inference
Three complementary estimators — IPTW, AIPW, and Propensity Score Matching — to estimate the causal effect of `number_inpatient` visits, insulin adjustment, and HbA1c testing on 30-day readmission risk. Key findings:

- Prior inpatient visits: AIPW risk ratio ~2.0 (strong positive causal effect)
- HbA1c testing: IPTW OR ~0.918 (modest protective effect)
- Insulin adjustment (Up/Down): PSM OR ~1.17 (positive association, likely confounded by severity)

These directional expectations serve as a SHAP consistency check in Part 2.

### Chapter 3 — Experimental Design
Power analysis and sequential trial design for a prospective A/B test of the deployed model's intervention effect. Designed for an average of 2,505 total encounters (half per arm) using O'Brien-Fleming group sequential boundaries at 3 interim analyses (3,048 using fixed design), targeting 80% power to detect a 3pp reduction in readmission rate.

---

## Part 2 — ML Pipeline and Deployment

Rendered book: `docs/part2/`
Stack: **Python · Logistic regression · XGBoost · LightGBM · Multi-layer perceptron · scikit-learn · TensorFlow · Vertex AI · Docker · Airflow**

### Preprocessing

`ml/preprocess.py` connects to Snowflake, applies a `ColumnTransformer`, and writes three parquet splits plus `preprocessor.joblib` and `raw_features.parquet` (pre-transform, for drift monitoring).

Feature treatment:
- **Drop**: identifiers, leakage columns, 11 NZV medication columns
- **Ordinal**: 8 medication dosage columns (`No < Steady < Down < Up`), HbA1c result, glucose serum
- **One-hot**: race, gender, age band, admission type, specialty, 3 diagnosis groups, change, diabetes_med
- **Standard scale**: 12 continuous utilisation and demographic features

### Chapter 1 — Model Evaluation

XGBoost tuned via Optuna (100 trials on validation AUC). Calibrated with Platt scaling (5-fold CV). Final validation metrics:

| Metric | Validation |
|---|---|
| AUC-ROC | 0.687 |
| AUC-PR | 0.113 |
| Brier score | 0.052 |
| Flag rate at τ=0.12 | 16.1% |
| Precision at τ=0.12 | 12.4% |
| Recall at τ=0.12 | 35.4% |

SHAP analysis confirmed directional consistency with Part 1 causal estimates: `number_inpatient`, `num_medications`, and `time_in_hospital` are the top contributors, all with expected sign.

### Chapter 2 — Deployment Drift Monitoring

The holdout set (13,917 encounters, row order = temporal by `encounter_id`) is divided into four semi-annual batches (~3,479 each, representing ~6-month windows across 2007–2008). Each batch is scored against the deployed model with the **training distribution as reference** for all metrics.

| Period | AUC-ROC | Flag Rate | KS Stat | Mean PSI | Alert |
|---|---|---|---|---|---|
| P1 (early 2007) | 0.699 | 14.3% | 0.031 | 0.041 | — |
| P2 | 0.660 | 13.9% | 0.032 | 0.050 | — |
| P3 | 0.664 | 14.3% | 0.052 | 0.060 | — |
| P4 (late 2008) | 0.671 | 15.0% | 0.046 | 0.063 | — |

No drift alerts triggered (thresholds: KS > 0.10, Mean PSI > 0.20). `number_diagnoses` showed individual PSI of 0.318 across the full holdout period — attributable to ICD-9 coding practice expansion driven by DRG reimbursement incentives, not genuine patient acuity change. AUC remains within ±0.04 of the validation reference across all periods.

---

## Deployment Architecture

```
[Snowflake Mart] → preprocess.py → raw_features.parquet
                                 ↓
                          xgb_model.joblib
                          preprocessor.joblib
                                 ↓
                        deploy/gcs_upload.py → gs://uci-diabetes-ml-cep/artifacts/v1/
                                 ↓
                     vertex_register.py → Vertex AI Model Registry
                                 ↓
                      vertex_deploy.py → Vertex AI Endpoint (n1-standard-2)
                                 ↓
                  FastAPI container (inference/app.py)
                  - Downloads artifacts from AIP_STORAGE_URI at startup
                  - POST /predict → { probability, flag, threshold }
```

The inference container uses `AIP_STORAGE_URI` (set by Vertex AI at runtime) to download artifacts from GCS, with `MODEL_DIR` fallback for local `docker-compose` testing.

See [deploy/DEPLOYMENT_LESSONS.md](deploy/DEPLOYMENT_LESSONS.md) for six deployment pitfalls encountered when building for Vertex AI from Apple Silicon, including the `buildx` attestation manifest issue and the ARM64 architecture rejection.

---

## Orchestration

Local Airflow (2.10, LocalExecutor) via `orchestration/docker-compose.yml`. The DAG would run monthly and covers:

```
preprocess → train_xgb → upload_gcs → vertex_register → vertex_deploy → drift_monitor
```

Model resource names are passed between tasks via Airflow XCom. The repo root is bind-mounted read-only at `/opt/uci_diabetes` inside the Airflow container.

---

## Running Locally

**Prerequisites**: Python 3.11+, Docker, `gcloud` CLI authenticated, Snowflake credentials in `.env`.

```bash
# 1. Install ML dependencies
python -m venv venv_ml && source venv_ml/bin/activate
pip install -r ml/requirements.txt

# 2. Preprocess (pulls from Snowflake)
python ml/preprocess.py

# 3. Train XGBoost
python ml/sklearn/train_xgb.py

# 4. Run drift monitoring
python monitoring/drift.py

# 5. Render reports (requires Quarto + R for Part 1, venv_ml kernel for Part 2)
cd reports/part2 && quarto render

# 6. Local inference smoke test
cd docker && docker compose up
```

**Deploy to Vertex AI**:
```bash
pip install -r deploy/requirements.txt

# Build and push inference image (Apple Silicon — note the flags)
docker buildx build --platform linux/amd64 --provenance=false \
  -t IMAGE:TAG -f docker/inference/Dockerfile --push .

python deploy/gcs_upload.py --version v1
python deploy/vertex_register.py --version v1
python deploy/vertex_deploy.py
```

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Data warehouse | Snowflake (GCP us-central1) |
| Ingestion | PySpark 3.5 + Snowflake Spark connector |
| Transformation | dbt 1.11 |
| Statistical analysis | R 4.3 · Quarto |
| ML training | Python · XGBoost · scikit-learn · Optuna · SHAP |
| Serving | FastAPI · Docker · Vertex AI custom container |
| Orchestration | Apache Airflow 2.10 (LocalExecutor, Docker Compose) |
| Monitoring | PSI · KS test · AUC tracking (Python) |
| Reporting | Quarto books (HTML) |
