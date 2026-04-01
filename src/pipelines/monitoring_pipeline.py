from snowflake.ml.registry import Registry

from src.ml_engineering.monitoring import setup_monitor
from src.ml_engineering.promotion import get_best_model_version
from src.utils.helpers import table_exists


def run(session, conf: dict):
    """Set up a ModelMonitor for drift detection on the prediction table."""
    print("=" * 60)
    print("MONITORING PIPELINE")
    print("=" * 60)

    database = conf["snowflake"]["database"]
    schema = conf["snowflake"]["schema"]
    mr_schema = conf["model_registry"]["schema"]
    model_name = conf["modelling"]["model_name"]
    prediction_table = f"{database}.{schema}.{conf['monitoring']['prediction_table']}"

    # Check prediction table exists before attempting monitor setup
    if not table_exists(session, prediction_table):
        print(f"Prediction table '{prediction_table}' does not exist yet.")
        print("Run batch inference first to create it, then re-run monitoring.")
        print("Skipping monitoring setup.")
        return None

    mr = Registry(session=session, database_name=database, schema_name=mr_schema)

    best_version, _ = get_best_model_version(mr, model_name)
    if best_version is None:
        print("No model versions found. Skipping monitoring setup.")
        return None

    version_name = best_version.version_name
    print(f"Setting up monitor for {model_name}/{version_name}")

    monitor = setup_monitor(session, mr, model_name, version_name, conf)
    print("\nMonitoring pipeline complete.")
    return monitor
