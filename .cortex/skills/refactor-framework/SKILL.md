---
name: refactor-framework
description: "Refactor this ML framework for a new use case. Use when: adapting the pipeline for a different model, dataset, or business problem. Triggers: refactor, adapt, new use case, change model, swap dataset, customise, customize, churn, fraud, forecasting, classification, new project, my own data."
---

# Refactor Framework

You are helping a user adapt this Snowflake ML Jobs framework to their own use case. This framework is a production-grade, config-driven ML pipeline that runs on Snowflake using `submit_directory`. Your job is to gather requirements, then systematically refactor every layer: config, data loading, feature engineering, modelling, and downstream pipelines.

## Framework Architecture

```
main.py                      ← CLI entrypoint — runs one or all pipelines
conf/parameters.yml          ← Single config file driving all pipeline stages
connection.json              ← Snowflake credentials (never committed)
conda.yml                    ← Conda environment for ML Jobs runtime
pipelines/
  feature_pipeline.py        ← Orchestrates: load → preprocess → Feature Store → Dataset
  training_pipeline.py       ← Submits HPO job via submit_directory
  promotion_pipeline.py      ← Promotes best model version
  inference_pipeline.py      ← Deploys SPCS service + batch predictions
  scheduling_pipeline.py     ← Stored procedure + Task for scheduled inference
  monitoring_pipeline.py     ← ModelMonitor for drift detection
src/
  session.py                 ← Snowpark session factory (reads connection.json)
  feature_engineering/
    data_loader.py           ← Joins source tables (currently CUSTOMERS + PURCHASE_BEHAVIOR)
    preprocessing.py         ← Derives features via Snowpark DataFrame ops
    feature_store.py         ← Entity, FeatureView, Dataset registration + spine builder
  modelling/
    pipeline.py              ← sklearn Pipeline (ColumnTransformer + XGBRegressor)
    splitter.py              ← Load Versioned Dataset, train/val split
    train.py                 ← ML Job entrypoint (HPO with Tuner, runs inside container)
    evaluate.py              ← MAE, MAPE, R² metric computation
  ml_engineering/
    promotion.py             ← Best version selection, alias/tag/default
    serving.py               ← SPCS service deployment, batch predictions
    scheduling.py            ← Stored procedure + Task creation
    monitoring.py            ← ModelMonitor setup
  utils/
    helpers.py               ← table_exists utility
    versioning.py            ← Auto-increment version helpers for models & datasets
```

## Step 1: Gather Requirements

**Goal:** Understand the user's new use case before touching any code.

**Ask these questions (use ask_user_question tool):**

1. **Use case**: What are you building? (e.g. churn prediction, fraud detection, demand forecasting, recommendation scoring)
2. **Model type**: Classification or regression? What algorithm? (e.g. XGBClassifier, LightGBM, RandomForest — the current framework uses XGBRegressor)
3. **Source tables**: What Snowflake tables will you use? Get the fully qualified names (DATABASE.SCHEMA.TABLE). Ask the user to list them or describe their data.
4. **Join logic**: How do the tables relate? What columns do they join on? Is it a simple join or something more complex?
5. **Entity**: What is the primary entity for the Feature Store? (e.g. CUSTOMER, PRODUCT, TRANSACTION) What are the join keys?
6. **Target column**: What column are you predicting?
7. **Timestamp column**: Which column tracks when records were updated? (Used for FeatureView `timestamp_col` and spine logic)
8. **Feature columns**: Which columns are features? Which are numerical, categorical, ordinal? (If the user is unsure, offer to inspect the tables and suggest.)
9. **Snowflake objects**: What database, schema, warehouse, compute pool, and role should the pipeline use? (Offer to keep the existing ones if the user is just experimenting.)

**STOP**: Confirm the requirements with the user before proceeding. Summarise what you understood and ask for corrections.

## Step 2: Refactor Configuration

**Goal:** Rewrite `conf/parameters.yml` for the new use case.

**File:** `conf/parameters.yml`

**Changes:**
- `snowflake.*` — Update database, schema, role, warehouse, warehouse_size, connection_file to user's values
- `feature_store.*` — Update entity_name, entity_join_keys, feature_view_name, feature_view_version, refresh_freq, dataset_name
- `model_registry.schema` — Update if needed
- `modelling.*` — New model_name, experiment_name, target_column, feature_columns, numerical_columns, categorical_columns, ordinal_columns, ordinal_categories, test_size, random_state, tuning_metric, tuning_mode
- `hpo.*` — Update hyperparameter search space to match the new algorithm (e.g. XGBClassifier params differ from XGBRegressor). Each key is a list of candidate values that gets wrapped in `tune.choice()`
- `compute.*` — Keep or adjust pool_name, stage_name, target_instances, num_trials
- `serving.*` — Update service_name
- `scheduling.*` — Update task_name, procedure_name, cron, timezone, warehouse
- `monitoring.*` — Update prediction_table, baseline_table, timestamp_column, id_columns, prediction_columns, actual_columns, segment_columns, background_warehouse, refresh_interval, aggregation_window

**Important:** The `tuning_metric` and `tuning_mode` must match the problem type:
- Regression: `mean_absolute_percentage_error` / `min`, or `r2_score` / `max`
- Classification: `f1_score` / `max`, `accuracy` / `max`, `roc_auc` / `max`, `log_loss` / `min`

## Step 3: Refactor Data Loading

**Goal:** Update `src/feature_engineering/data_loader.py` AND `pipelines/feature_pipeline.py`.

### `src/feature_engineering/data_loader.py`
- Replace the `load_data()` function body with the user's join logic
- Update column selections to match the user's schema
- Handle any renaming, deduplication, or filtering the user needs
- Keep the function signature compatible: takes Snowpark DataFrames, returns a Snowpark DataFrame
- Currently the rename dict handles `UPDATED_AT` → `BEHAVIOR_UPDATED_AT` / `CUSTOMER_UPDATED_AT` — update for the user's timestamp columns

### `pipelines/feature_pipeline.py`
- **CRITICAL**: This file hardcodes the table names `CUSTOMERS` and `PURCHASE_BEHAVIOR` on lines 29-30. Update these to the user's table names.
- If the user has a single table, simplify: remove the second `session.table()` call and pass a single DataFrame to `load_data()`
- If the user has more than two tables, add additional `session.table()` calls and update `load_data()` signature accordingly

## Step 4: Refactor Feature Engineering

**Goal:** Update `src/feature_engineering/preprocessing.py` and `src/feature_engineering/feature_store.py`.

### `src/feature_engineering/preprocessing.py`
- Replace the `pre_process()` function with the user's feature derivations
- Use Snowpark DataFrame operations (F.col, F.when, F.datediff, F.iff, F.round, etc.)
- Keep the function signature: takes a Snowpark DataFrame, returns a Snowpark DataFrame
- Make sure all derived column names match what's listed in `parameters.yml` under `feature_columns`

### `src/feature_engineering/feature_store.py`
Update ALL of these:

1. **`register_feature_view()`**:
   - Change the `feature_desc` dict to describe the new features
   - **CRITICAL**: The `timestamp_col` is hardcoded to `"BEHAVIOR_UPDATED_AT"` — update to the user's timestamp column
   - Update the `desc` string

2. **`register_entity()`**: Entity name and join keys come from config — usually no changes needed, but verify

3. **`get_spine_df()`**:
   - **CRITICAL**: Currently hardcodes `group_by("CUSTOMER_ID")` — update to the user's entity join key
   - The `ASOF_DATE` column name is used by `generate_training_dataset()` as `spine_timestamp_col` — keep this consistent or update both

4. **`generate_training_dataset()`**: Usually no changes needed — it reads dataset_name from config

## Step 5: Refactor Modelling

**Goal:** Update the model pipeline for the new algorithm and problem type.

### `src/modelling/pipeline.py`
- Replace `xgb.XGBRegressor` with the user's chosen model (e.g. `xgb.XGBClassifier`, `lightgbm.LGBMClassifier`)
- Update the ColumnTransformer if column types changed (MinMaxScaler for numerical, OneHotEncoder for categorical, OrdinalEncoder for ordinal)
- If the user has no ordinal columns, remove that transformer
- Update imports (`import xgboost as xgb` → whatever library the user needs)

### `src/modelling/evaluate.py`
- Replace regression metrics (MAE, MAPE, R²) with appropriate metrics:
  - Classification: accuracy, precision, recall, f1, roc_auc, log_loss
  - Regression: MAE, MAPE, RMSE, R²
- Update the return dict keys — these must match what `train.py` reports to the tuner and what `promotion.py` uses to find the best version
- The `tuning_metric` in `parameters.yml` must be a key in this dict

### `src/modelling/train.py`
- **NOTE**: The `SnowflakeXgboostCallback` is commented out by default because it does not support `target_platforms` or `enable_explainability` options. If switching away from XGBoost, remove the commented-out callback references entirely. If staying with XGBoost and you don't need explainability/target_platforms, you can uncomment the callback and remove the `exp.log_model()` call.
- Update the `import xgboost` lines if the model library changed
- The HPO search space is built dynamically from `parameters.yml` — verify the parameter names in `hpo.*` match the new model's constructor args (e.g. XGBClassifier uses the same params as XGBRegressor, but LightGBM uses `learning_rate` instead of `eta`, `num_leaves` instead of `max_depth`, etc.)
- If the model requires different fitting logic (e.g. `eval_set` for early stopping), update the `model.fit()` call
- The `train()` function runs inside Ray workers — all imports must happen inside the function body, not at module level

### `src/modelling/splitter.py`
- Usually no changes needed — it's generic (loads a Versioned Dataset via DataConnector)
- If the user needs stratified splitting (common for imbalanced classification), add `stratify=y` to `train_test_split`

## Step 6: Refactor Downstream Pipelines

**Goal:** Update inference, scheduling, and monitoring for the new use case.

### `src/ml_engineering/promotion.py`
- The `get_best_model_version()` function has a default `metric="mean_absolute_percentage_error"` — this should match `tuning_metric` from config. The calling code in `pipelines/promotion_pipeline.py` passes config values, but verify the fallback default is sensible for the new use case.

### `src/ml_engineering/serving.py`
- Usually no changes — it's generic (calls `mv.run()`)
- If the user needs different prediction column naming, update `prediction_column` parameter

### `pipelines/inference_pipeline.py`
- Constructs input table as `{database}.{fs_schema}.{fv_name}${fv_version}` (the Dynamic Table backing the FeatureView). This should work automatically from config, but verify the config keys are set correctly.

### `src/ml_engineering/scheduling.py`
- Reads all config values and embeds them in the stored procedure SQL. Config-driven — should work if `parameters.yml` is correct.

### `src/ml_engineering/monitoring.py`
- Config-driven — should work if `parameters.yml` monitoring section is correct
- Verify `timestamp_column`, `id_columns`, `prediction_columns`, `actual_columns` match the prediction output

## Step 7: Update Dependencies

**Goal:** Ensure `conda.yml` includes any new libraries.

**File:** `conda.yml`

**Changes:**
- If switching from XGBoost to LightGBM, replace `xgboost` with `lightgbm` (or add alongside)
- If adding new sklearn components, verify they're in the base `scikit-learn` package
- For classification metrics that need `sklearn.metrics`, no new deps needed
- Keep `snowflake-ml-python`, `snowflake-snowpark-python`, `pyyaml` as-is

## Step 8: Validate

**Goal:** Verify the refactored framework is consistent end-to-end.

**Checklist:**
- [ ] All column names in `parameters.yml` match the actual table schemas
- [ ] `data_loader.py` returns a DataFrame with all columns referenced by `preprocessing.py`
- [ ] `preprocessing.py` output includes all columns listed in `parameters.yml` `feature_columns`
- [ ] `feature_pipeline.py` table names match the user's actual source tables
- [ ] `feature_store.py` `timestamp_col` in `register_feature_view()` matches the user's timestamp column
- [ ] `feature_store.py` `get_spine_df()` groups by the correct entity join key (not hardcoded `CUSTOMER_ID`)
- [ ] `pipeline.py` model class matches the algorithm in `parameters.yml`
- [ ] `evaluate.py` metrics dict includes a key matching `tuning_metric` in config
- [ ] `train.py` HPO param names match the model's constructor args
- [ ] `train.py` callback (SnowflakeXgboostCallback) is removed if not using XGBoost (it is commented out by default)
- [ ] `promotion.py` default metric matches `tuning_metric` or is overridden by the caller
- [ ] `monitoring.*` columns in config match the actual prediction output schema
- [ ] `conda.yml` includes all required packages for the new model library

**Actions:**
1. Read through each modified file and cross-reference column names against `parameters.yml`
2. If the user has provided table access, run `SELECT * FROM table LIMIT 5` to verify schemas
3. Present a summary of all changes to the user

## Step 9: Test Run

**Goal:** Help the user do a dry run.

**Suggest:**
```bash
# Test feature pipeline first (validates data loading + Feature Store + Dataset)
python main.py feature

# Then test training (validates model + HPO + submit_directory)
python main.py training

# Then run the rest
python main.py --from promotion --to monitoring
```

If errors occur, help debug by reading logs and tracing the issue back to the relevant file. Common issues:
- **Column not found** → mismatch between `parameters.yml` and actual table/DataFrame columns
- **Import error in Ray worker** → missing import inside `train()` function (must be inside, not module-level)
- **HPO param error** → parameter name in `hpo.*` doesn't match model constructor (e.g. `eta` vs `learning_rate`)
- **FeatureView registration fails** → entity join key mismatch or timestamp column doesn't exist

## Important Notes

- **Never hardcode values** — everything goes in `parameters.yml`
- **Keep the pipeline orchestration pattern** — individual pipelines in `pipelines/`, business logic in `src/`
- **Preserve the `submit_directory` pattern** — `train.py` runs inside a Snowflake container, so it must load config from `conf/parameters.yml` at runtime
- **The `train()` function in `train.py` is called by Ray workers** — imports must happen inside the function, not at module level (path issues in distributed execution)
- **Feature descriptions** in `feature_store.py` are optional but good practice — update them to match new features
- **`connection.json` is never committed** — it's in `.gitignore`. The user creates it from `connection.json.example`
- **`src/session.py` is only used for local execution** — inside a Snowflake container (submit_directory), `Session.builder.getOrCreate()` provides the session automatically
