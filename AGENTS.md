# ML Jobs Framework — Agent Guide

## What This Repo Is

A production-grade, Kedro/Cookiecutter-style Python framework for ML on Snowflake. The entire project directory is submitted to a compute pool via `submit_directory`. Six pipeline stages, centralised YAML config, pre-commit hooks, modular packages.

**Use case:** Customer value regression (predict `MONTHLY_CUSTOMER_VALUE`).

## Repo Structure

```
├── main.py                          # CLI entrypoint — run one or all pipelines
├── conda.yml                        # Conda environment (runtime deps)
├── pyproject.toml                   # black/ruff/isort config
├── .pre-commit-config.yaml          # Git hooks
├── conf/
│   └── parameters.yml               # All pipeline configuration
├── pipelines/
│   ├── feature_pipeline.py          # Feature Store: load → preprocess → register → dataset
│   ├── training_pipeline.py         # Submit HPO training job via submit_directory
│   ├── promotion_pipeline.py        # Explain best model + promote (alias, tags, default)
│   ├── inference_pipeline.py        # Batch inference via model version
│   ├── scheduling_pipeline.py       # Scheduled batch inference via stored procedure
│   └── monitoring_pipeline.py       # ModelMonitor for drift detection
└── src/
    ├── session.py                   # Snowpark session factory
    ├── features/
    │   ├── data_loader.py           # Join CUSTOMERS + PURCHASE_BEHAVIOR
    │   └── preprocessing.py         # Feature derivation (Snowpark DataFrame ops)
    ├── feature_engineering/
    │   └── feature_store.py         # Entity, FeatureView, Dataset registration
    ├── modelling/
    │   ├── pipeline.py              # sklearn Pipeline (ColumnTransformer + XGBRegressor)
    │   ├── splitter.py              # Train/val split, DataConnector utilities
    │   ├── evaluate.py              # MAE, MAPE, R² evaluation
    │   └── train.py                 # ML Job entrypoint for HPO (submit_directory target)
    ├── ml_engineering/
    │   ├── promotion.py             # Best version selection, alias/tag/default promotion
    │   ├── serving.py               # SPCS service deployment, batch predictions
    │   ├── scheduling.py            # Stored procedure + Task for scheduled inference
    │   └── monitoring.py            # ModelMonitor setup
    └── utils/
        ├── helpers.py               # table_exists, etc.
        └── versioning.py            # Auto-increment version helpers
```

## Environment

```bash
conda env create -f conda.yml
conda activate snowflake_ds
pip install pre-commit && pre-commit install
```

Python 3.10. Key packages: `snowflake-ml-python>=1.30.0`, `xgboost`, `scikit-learn`, `altair`.

## Linting and Formatting

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
black .
isort .
ruff check . --fix
```

Config lives in `pyproject.toml`:
- **black** — line-length=120
- **isort** — black-compatible profile
- **ruff** — pyflakes, pycodestyle, isort, bugbear, pyupgrade, flake8-simplify

## How to Run

```bash
python main.py all                        # Full end-to-end
python main.py feature                    # Just feature engineering
python main.py training                   # Submit HPO job
python main.py promotion                  # Promote best model
python main.py inference                  # Deploy + batch predict
python main.py scheduling                 # Create scheduled task
python main.py monitoring                 # Set up drift monitoring
python main.py --from training --to inference   # Run a range
python main.py all --config conf/parameters.yml # Custom config path
```

## Snowflake Connection

`src/session.py` reads `connection.json` (copy from `connection.json.example`).

Environment variable override: set `SNOWFLAKE_CONNECTION_NAME` to use a named connection.

Inside ML Job containers, `Session.builder.getOrCreate()` provides the session automatically.

## Configuration

All parameters live in `conf/parameters.yml`. No hardcoded values in code. Key sections:

- **snowflake** — database, schema, role, warehouse
- **feature_store** — entity, FeatureView, refresh frequency, dataset name
- **model_registry** — schema for versioned models
- **modelling** — model name, feature/target columns, column types, encoders, train/test split, tuning metric
- **hpo** — hyperparameter search space (values wrapped with `tune.choice()` at runtime)
- **compute** — pool name, stage, target instances, num trials
- **serving** — inference service config
- **scheduling** — stored procedure + Task definition, cron schedule
- **monitoring** — prediction/baseline tables, refresh intervals

## Key Snowflake Objects

- **Database:** `RETAIL_REGRESSION_DEMO`
- **Schemas:** `DS`, `MODELLING`, `FEATURE_STORE`
- **Compute Pool:** `CUSTOMER_VALUE_MODEL_POOL_CPU`
- **Model:** `UC01_SNOWFLAKEML_RF_REGRESSOR_MODEL`
- **Stage:** `payload_stage`

## Architecture Notes

- `submit_directory` ships the entire project to the compute pool. `src/modelling/train.py` is the entrypoint executed inside the container.
- `train.py` has two roles: (1) the `train()` function is the per-trial HPO function run by Ray workers, (2) the `__main__` block sets up the Tuner and launches HPO.
- `train()` reads `conf/parameters.yml` inside the container — the file is available because `submit_directory` uploads the whole project.
- `SnowflakeXgboostCallback` is commented out in `train.py` — it doesn't support `target_platforms` or `enable_explainability`. Models are logged via `exp.log_model()` with `target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]` and `options={"enable_explainability": True}`.
- Before HPO, the `__main__` block pre-creates the model in the Registry with a dummy version to avoid "Object already exists" race conditions from parallel trials.
- `promotion_pipeline.py` runs explainability (SHAP) on the best model before promoting it.

## Common Modifications

- **Change model type:** Edit `src/modelling/pipeline.py` (sklearn Pipeline), update `hpo` section in `parameters.yml`, update `src/modelling/evaluate.py` metrics
- **Change features:** Edit `src/features/data_loader.py` and `src/features/preprocessing.py`, update column lists in `parameters.yml`
- **Change HPO:** Modify `hpo` section in `parameters.yml` (parameter names must match model constructor args)
- **Add a new pipeline stage:** Create `pipelines/new_pipeline.py` with a `run(session, conf)` function, add to `PIPELINE_ORDER` and `PIPELINES` in `main.py`
- **Refactor for a different use case:** Use the `refactor-framework` Cortex Code skill (`.cortex/skills/refactor-framework/SKILL.md`)
