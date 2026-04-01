# 03 — ML Jobs Framework

A Kedro/Cookiecutter-inspired boilerplate for building production ML pipelines on Snowflake, using **ML Jobs `submit_directory`** for serverless compute.

## Project Structure

```
03_ml_jobs_framework/
├── main.py                          # CLI entrypoint — run one or all pipelines
├── conda.yml                        # Conda environment (runtime dependencies)
├── pyproject.toml                   # Project metadata, black/ruff/isort config
├── .pre-commit-config.yaml          # Pre-commit hooks (black, ruff, isort, etc.)
├── conf/
│   └── parameters.yml               # Single YAML config for all pipelines
└── src/
    ├── session.py                   # Snowpark session factory (local execution)
    ├── pipelines/
    │   ├── feature_pipeline.py      # Feature Store: load → preprocess → register → dataset
    │   ├── training_pipeline.py     # Submit HPO training job via submit_directory
    │   ├── promotion_pipeline.py    # Explain best model + promote (alias, tags, default)
    │   ├── inference_pipeline.py    # Batch inference via model version
    │   ├── scheduling_pipeline.py   # Scheduled batch inference via stored procedure
    │   └── monitoring_pipeline.py   # Set up ModelMonitor for drift detection
    ├── feature_engineering/
    │   ├── data_loader.py           # Join CUSTOMERS + PURCHASE_BEHAVIOR
    │   ├── preprocessing.py         # Feature derivation (Snowpark DataFrame ops)
    │   └── feature_store.py         # Entity, FeatureView, Dataset registration
    ├── modelling/
    │   ├── pipeline.py              # sklearn Pipeline (ColumnTransformer + XGBRegressor)
    │   ├── splitter.py              # Train/val split, DataConnector utilities
    │   ├── evaluate.py              # MAE, MAPE, R² evaluation
    │   └── train.py                 # ML Job entrypoint — HPO with Tuner (submit_directory)
    ├── ml_engineering/
    │   ├── promotion.py             # Best version selection, alias/tag/default promotion
    │   ├── serving.py               # Inference service deployment, batch predictions
    │   ├── scheduling.py            # Stored procedure + task for scheduled inference
    │   └── monitoring.py            # ModelMonitor setup
    └── utils/
        ├── helpers.py               # Shared utilities (table_exists, etc.)
        └── versioning.py            # Auto-increment version helpers for models & datasets
```

## Snowflake Services Used

| Service | Purpose |
|---|---|
| **ML Jobs (`submit_directory`)** | Builds a clean payload from `src/` + `conf/` and submits to a compute pool; `modelling/train.py` is the entrypoint |
| **Feature Store** | Managed FeatureViews backed by Dynamic Tables with scheduled refresh |
| **Model Registry** | Versioned model storage with aliases, tags, and default versions |
| **Experiment Tracking** | Per-trial parameter/metric/model logging during HPO |
| **HPO Tuner** | RandomSearch over XGBoost hyperparameters across distributed trials |
| **Datasets & DataConnectors** | Immutable, versioned snapshots for reproducible training |
| **Model Monitor** | Continuous drift and performance monitoring |
| **Model Explainability** | SHAP-based feature importance via the built-in `explain` function |

## Quick Start

```bash
# Run the full pipeline end-to-end
python main.py all

# Run individual pipelines
python main.py feature      # Feature engineering + Feature Store
python main.py training     # Submit HPO training ML Job
python main.py promotion    # Promote best model
python main.py inference    # Deploy service + batch predictions
python main.py scheduling   # Create scheduled inference task
python main.py monitoring   # Set up monitoring

# Use custom config
python main.py all --config conf/parameters.yml

# Run a range of pipelines
python main.py --from training --to inference
```

## How `submit_directory` Works

Unlike the `@remote` decorator (used in `02_ml_jobs_notebook`), this framework submits the entire project directory to a Snowflake compute pool:

```python
from snowflake.ml.jobs import submit_directory

job = submit_directory(
    payload_dir,                           # clean payload: src/ contents + conf/
    "CUSTOMER_VALUE_MODEL_POOL_CPU",       # compute pool
    entrypoint="modelling/train.py",       # script to execute (relative to payload root)
    stage_name="payload_stage",
    session=session,
)
job.wait()
```

Inside the container, `Session.builder.getOrCreate()` provides the Snowpark session automatically — no credentials needed. All configuration is read from `conf/parameters.yml` at runtime.

## Configuration

All parameters live in `conf/parameters.yml`. Key sections:

- **snowflake** — connection, database, schema, warehouse
- **feature_store** — entity, feature view, refresh frequency, dataset name
- **model_registry** — schema for versioned models
- **modelling** — model name, feature/target columns, encoders, train/test split
- **hpo** — hyperparameter search space (values wrapped with `tune.choice()`)
- **compute** — pool name, stage, trial count, instance count
- **serving** — inference service config
- **scheduling** — stored procedure + task for scheduled batch inference
- **monitoring** — prediction/baseline tables, refresh intervals

## Environment Setup

```bash
# Create conda environment
conda env create -f conda.yml
conda activate snowflake_ds

# Install dev tools (linting, formatting)
pip install pre-commit black ruff isort

# Install pre-commit hooks
pre-commit install

# Run linters manually
black .
ruff check . --fix
isort .
```

## Dev Tooling

| Tool | Config | Purpose |
|---|---|---|
| **black** | `pyproject.toml` | Code formatting (line-length=120) |
| **ruff** | `pyproject.toml` | Fast linting (pyflakes, pycodestyle, isort, bugbear) |
| **isort** | `pyproject.toml` | Import sorting (black-compatible profile) |
| **pre-commit** | `.pre-commit-config.yaml` | Git hooks: black, ruff, isort, trailing whitespace, YAML check, large file guard |

## Comparison with Other Implementations

| Aspect | 01 (Notebooks) | 02 (ML Jobs Notebook) | **03 (Framework)** |
|---|---|---|---|
| Execution | Interactive cells | `@remote` decorator | `submit_directory` |
| Structure | Single notebook | Notebook + helper `.py` | Modular packages |
| Config | Hardcoded | Hardcoded | `parameters.yml` |
| Reusability | Low | Medium | **High** |
| CI/CD Ready | No | Partial | **Yes** |
| HPO | In-notebook Tuner | In-notebook Tuner | **Entrypoint script** |
| Best For | Exploration | Prototyping | **Production** |

## Related Repos

| Repo | Description |
|------|-------------|
| [snowflake-ds-setup](https://github.com/jar-ry/snowflake-ds-setup) | Environment setup, data generation, and helper utilities (run this first) |
| [snowflake-ds-01-notebooks](https://github.com/jar-ry/snowflake-ds-01-notebooks) | Same pipeline running entirely in Snowflake UI (no local setup) |
| [snowflake-ds-02-ml-jobs-notebook](https://github.com/jar-ry/snowflake-ds-02-ml-jobs-notebook) | Same pipeline run locally with `@remote` decorator for ML Jobs |
| [snowflake-ds-04-feature-store](https://github.com/jar-ry/snowflake-ds-04-feature-store) | Split repo: Feature Store with FeatureViews and Versioned Datasets |
| [snowflake-ds-04-ml-training](https://github.com/jar-ry/snowflake-ds-04-ml-training) | Split repo: ML Training with training, promotion, inference, monitoring |
