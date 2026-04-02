# docker/inference/Dockerfile
# Lean inference container — FastAPI REST server wrapping the XGBoost model.
# Implements the Vertex AI custom container prediction interface.
#
# Build (from repo root): docker build -t uci-diabetes-inference:latest -f docker/inference/Dockerfile .
# Run locally:
#   docker run -p 8080:8080 -v $(pwd)/ml/data:/app/model:ro uci-diabetes-inference
FROM uci-diabetes-base:latest

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.0"

# Application code
COPY docker/inference/app.py /app/app.py

# Model artifacts are mounted at runtime — no baked-in weights
# Mount: -v /path/to/ml/data:/app/model
ENV MODEL_DIR=/app/model
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
