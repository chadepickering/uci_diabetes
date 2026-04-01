# reports/part2/ — ML Evaluation Report

Quarto book rendered to `docs/part2/`. Parallel structure to `reports/part1/`.

---

## Prerequisites

### 1. ML artifacts

Run the full Part 2 pipeline first (`ml/` directory). The report loads from
pre-computed local files — no Snowflake connection needed.

Required files in `ml/data/`:

```
train.parquet, validation.parquet, holdout.parquet, raw_features.parquet
lr_model.joblib, xgb_model.joblib, lgbm_model.joblib
tf_model.keras, tf_calibrator.joblib
lr_metrics.json, xgb_metrics.json, lgbm_metrics.json, tf_metrics.json
lr_coefficients.csv, xgb_shap_importance.csv, lgbm_shap_importance.csv, tf_ig_importance.csv
feature_names.txt
```

### 2. Python environment

The report runs in `venv_ml`. Two packages not in the original ML requirements
are needed for Quarto rendering:

```bash
source venv_ml/bin/activate
pip install matplotlib pyyaml
```

`pyyaml` is required by Quarto's internal bootstrap script (`jupyter.py`), which
runs under the system Python. Install it there too:

```bash
pip3 install pyyaml nbformat nbclient   # system Python (not venv_ml)
```

### 3. Jupyter kernel registration

Quarto resolves the Python environment via a named Jupyter kernel, not the
activated venv. Register `venv_ml` as a kernel once:

```bash
source venv_ml/bin/activate
python -m ipykernel install --user --name venv_ml --display-name "Python (venv_ml)"
```

The kernel name `venv_ml` is declared in the front matter of each `.qmd` file:

```yaml
---
jupyter: venv_ml
---
```

This must be in the **QMD front matter**, not in `_quarto.yml`. Quarto does not
propagate the `jupyter:` key from the book-level YAML to individual chapters.

---

## Rendering

```bash
# Single chapter
quarto render reports/part2/01_model_eval.qmd

# Full book
cd reports/part2 && quarto render
```

Output is written to `docs/part2/`. The `output-dir: ../../docs/part2` path
in `_quarto.yml` triggers Quarto path warnings ("not a subdirectory of the main
project directory") — these are cosmetic and both `index.html` and chapter HTML
files are created correctly. Part 1 has the same configuration.

---

## Structure

```
reports/part2/
├── _quarto.yml          # book config, output-dir → docs/part2/
├── index.qmd            # overview chapter
├── 01_model_eval.qmd    # main evaluation document
└── py/
    └── load_artifacts.py  # shared loader (parallel to part1/R/connections.R)
```

`load_artifacts.py` is sourced via `exec()` at the top of each `.qmd`. It loads
parquet splits, metrics JSON, feature importance CSVs, and exposes lazy model
loaders (`load_sklearn_model`, `load_tf_model`). It requires `repo_root` to be
defined in the calling namespace before `exec()`.
