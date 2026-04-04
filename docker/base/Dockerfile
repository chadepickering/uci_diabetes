# docker/base/Dockerfile
# Shared base image — core ML stack common to both training and inference.
#
# Build: docker build --platform linux/amd64 -t uci-diabetes-base:latest docker/base/
FROM --platform=linux/amd64 python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Core ML stack pinned to match venv_ml (ml/requirements.txt)
RUN pip install --no-cache-dir \
    scikit-learn==1.8.0 \
    xgboost==3.2.0 \
    pandas==3.0.1 \
    numpy==2.4.4 \
    pyarrow==23.0.1 \
    joblib==1.5.3 \
    python-dotenv==1.2.2
