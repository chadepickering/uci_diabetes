# Vertex AI Custom Container Deployment — Lessons Learned

Issues encountered deploying an XGBoost model to Vertex AI from an Apple Silicon Mac.

---

## 1. Install GCP SDK into the project venv before running deploy scripts

`google-cloud-aiplatform` and `google-cloud-storage` are not part of the ML training requirements. They need to be installed separately before any deploy script is run.

```bash
pip install -r deploy/requirements.txt
```

**Avoid**: assuming the ML venv already has GCP dependencies just because `gcloud` CLI is installed on the host.

---

## 2. Apple Silicon → Vertex AI requires explicit `linux/amd64` cross-compilation

Vertex AI prediction nodes are x86_64 only. Docker images built on Apple Silicon (M1/M2/M3) default to `arm64` and will be rejected at deploy time with:

> `Unsupported container image architecture. Please rebuild your image on x86.`

**Fix**: add `--platform=linux/amd64` to both the `FROM` line and the build command:

```dockerfile
FROM --platform=linux/amd64 python:3.13-slim
```

```bash
docker build --platform linux/amd64 -t myimage:latest .
```

---

## 3. `docker buildx` adds an attestation manifest that breaks Vertex AI

Even with `--platform linux/amd64`, `docker buildx build` by default appends a provenance/attestation manifest (`unknown/unknown` platform entry) to the registry manifest list. Vertex AI rejects any image whose manifest list contains non-`linux/amd64` entries.

**Symptom**: `docker manifest inspect` shows `{'architecture': 'unknown', 'os': 'unknown'}` alongside the amd64 entry.

**Fix**: always pass `--provenance=false` when building for Vertex AI:

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -t IMAGE:TAG \
  -f path/to/Dockerfile \
  --push \
  .
```

Verify the result before deploying:

```bash
docker manifest inspect --verbose IMAGE:TAG
# Should show: {'architecture': 'amd64', 'os': 'linux'} — nothing else
```

---

## 4. Vertex AI custom containers do not auto-mount model artifacts

The model container starts with an empty filesystem. The `MODEL_DIR` volume mount used in local `docker-compose` testing does not exist on Vertex AI.

**How Vertex AI provides artifacts**: it sets the `AIP_STORAGE_URI` environment variable at container startup, pointing to the GCS path registered via `aiplatform.Model.upload(artifact_uri=...)`.

**Fix**: in `app.py` startup, check for `AIP_STORAGE_URI` and download artifacts from GCS before loading them:

```python
_AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "").rstrip("/")

def _load_artifacts():
    if _AIP_STORAGE_URI:
        local_dir = pathlib.Path(tempfile.mkdtemp(prefix="model_"))
        _download_from_gcs(_AIP_STORAGE_URI, local_dir)
    else:
        local_dir = pathlib.Path(os.environ.get("MODEL_DIR", "/app/model"))
    ...
```

Also add `google-cloud-storage` to the inference container's pip install — it is not needed locally but is required on Vertex AI.

---

## 5. Registering a new image tag requires re-registering the model

The Vertex AI Model Registry stores the image URI at registration time. Pushing a new image to the same tag does **not** update an already-registered model — the LRO will still spin up the old image digest.

**Fix**: register a new model entry pointing to the corrected image, then deploy that new model to the endpoint. The endpoint deploy step is idempotent (it reuses the existing endpoint), so only `vertex_register.py` needs to be re-run.

---

## 6. GCS artifact path and image tag are independent versioning axes

`vertex_register.py --version v2` sets `artifact_uri=gs://BUCKET/artifacts/v2`, which will fail if those files don't exist. The image tag in `.env` (`INFERENCE_IMAGE`) is a separate axis.

**Pattern that works**:
- Artifacts live at `gs://BUCKET/artifacts/v1` (upload once per training run)
- Image tag bumps independently (`:v1`, `:v2`, `:v3`) as the container is fixed
- Pass `--version v1` (the artifact path) even when registering a new display name for a fixed container
