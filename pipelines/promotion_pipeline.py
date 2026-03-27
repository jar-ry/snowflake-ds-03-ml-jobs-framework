from snowflake.ml.registry import Registry

from src.ml_engineering.promotion import get_best_model_version, promote_model


def run(session, conf: dict):
    """Find the best model version by metric and promote it (tag + set default)."""
    print("=" * 60)
    print("PROMOTION PIPELINE")
    print("=" * 60)

    database = conf["snowflake"]["database"]
    mr_schema = conf["model_registry"]["schema"]
    model_name = conf["modelling"]["model_name"]

    mr = Registry(session=session, database_name=database, schema_name=mr_schema)

    print(f"\n[1/2] Finding best version for {model_name}...")
    metric = conf["modelling"].get("tuning_metric", "mean_absolute_percentage_error")
    mode = conf["modelling"].get("tuning_mode", "min")
    best_version, best_score = get_best_model_version(mr, model_name, metric=metric, mode=mode)
    if best_version is None:
        print("No model versions found. Skipping promotion.")
        return None

    version_name = best_version.version_name
    if best_score is not None:
        print(f"  Best: {version_name} ({metric}={best_score:.4f})")
    else:
        print(f"  Best: {version_name} (no metrics, using latest)")

    print("[2/3] Explaining best model (feature importance)...")
    dataset_fqn = f"{database}.{conf['feature_store']['schema']}.{conf['feature_store']['dataset_name']}"
    from snowflake.ml.data.data_connector import DataConnector
    from snowflake.ml.dataset import Dataset, load_dataset

    from modelling.splitter import generate_train_val_set

    ds = Dataset.load(session=session, name=dataset_fqn)
    ds_latest_version = str(ds.list_versions()[-1])
    ds_df = load_dataset(session, dataset_fqn, ds_latest_version)
    dc = DataConnector.from_dataset(ds_df)
    df = dc.to_pandas()
    train_df, _ = generate_train_val_set(
        df,
        feature_columns=conf["modelling"]["feature_columns"],
        target_column=conf["modelling"]["target_column"],
    )
    X_explain = train_df.drop(conf["modelling"]["target_column"], axis=1).head(100)
    explanations = best_version.run(X_explain, function_name="explain")
    print(explanations)

    print("[3/3] Promoting model (tag + default)...")
    mv = promote_model(session, mr, model_name, version_name)

    print(f"\nPromotion complete: {model_name}/{version_name}")
    return mv
