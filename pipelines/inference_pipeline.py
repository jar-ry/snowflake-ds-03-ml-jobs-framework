from snowflake.ml.registry import Registry

from src.ml_engineering.serving import deploy_inference_service, run_batch_predictions
from src.utils.helpers import table_exists


def run(session, conf: dict):
    """Deploy the inference service, run batch predictions, and save a baseline snapshot."""
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)

    database = conf["snowflake"]["database"]
    mr_schema = conf["model_registry"]["schema"]
    model_name = conf["modelling"]["model_name"]
    prediction_table = conf["monitoring"]["prediction_table"]
    baseline_table = conf["monitoring"].get("baseline_table")
    pool_name = conf["compute"]["pool_name"]
    service_name = conf["serving"]["service_name"]

    # Use the feature view dynamic table as input for inference
    fs_schema = conf["feature_store"]["schema"]
    fv_name = conf["feature_store"]["feature_view_name"]
    fv_version = conf["feature_store"]["feature_view_version"]
    input_table = f"{database}.{fs_schema}.{fv_name}${fv_version}"

    mr = Registry(session=session, database_name=database, schema_name=mr_schema)

    # Verify default version exists
    model = mr.get_model(model_name)
    default_version = model.default
    if default_version is None:
        print("No default model version set. Run promotion pipeline first.")
        return None

    version_name = default_version.version_name
    print(f"  Model   : {model_name}")
    print(f"  Version : {version_name} (default)")
    print(f"  Input   : {input_table}")
    print(f"  Output  : {prediction_table}")
    print(f"  Service : {service_name}")
    print(f"  Pool    : {pool_name}")

    print("\n[1/3] Deploying inference service...")
    deploy_inference_service(session, mr, model_name, version_name, pool_name, service_name)

    print("[2/3] Running batch predictions via SPCS...")
    run_batch_predictions(session, mr, model_name, input_table, prediction_table, service_name)

    row_count = session.table(prediction_table).count()
    print(f"  {row_count} predictions written.")

    # Save first prediction run as baseline for drift monitoring
    print("[3/3] Checking baseline table...")
    if baseline_table:
        if not table_exists(session, baseline_table):
            session.table(prediction_table).write.mode("overwrite").save_as_table(baseline_table)
            print(f"  Baseline snapshot saved to {baseline_table} ({row_count} rows)")
        else:
            print(f"  Baseline already exists at {baseline_table}, skipping.")
    else:
        print("  No baseline_table configured, skipping.")

    print("\nInference pipeline complete.")
