# docker/train/Dockerfile
# Training container — preprocessing + model training for all four models.
# Includes Snowflake connector for live data pulls; use --dry-run to skip.
#
# Build (from repo root): docker build --platform linux/amd64 -t uci-diabetes-train:latest -f docker/train/Dockerfile .
# Run (dry-run, XGBoost only):
#   docker run --platform linux/amd64 -v $(pwd)/ml/data:/app/ml/data uci-diabetes-train all --dry-run
FROM --platform=linux/amd64 uci-diabetes-base:latest

RUN pip install --no-cache-dir \
    lightgbm==4.6.0 \
    tensorflow==2.20.0 \
    optuna==4.8.0 \
    shap==0.47.2 \
    imbalanced-learn==0.14.1 \
    snowflake-connector-python==4.4.0

# Copy ML source
COPY ml/ /app/ml/

COPY docker/train/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Model artifacts written to /app/ml/data — mount a volume here
VOLUME ["/app/ml/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
# Default: full pipeline in dry-run mode
CMD ["all", "--dry-run"]
