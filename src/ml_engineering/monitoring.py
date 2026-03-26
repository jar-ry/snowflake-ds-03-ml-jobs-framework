from snowflake.ml.monitoring.entities.model_monitor_config import ModelMonitorConfig, ModelMonitorSourceConfig
from snowflake.ml.registry import Registry


def setup_monitor(session, mr: Registry, model_name: str, version_name: str, conf: dict):
    """Create a ModelMonitor for drift detection on a prediction table.

    Configures the monitor source (prediction + optional baseline tables),
    refresh interval, and aggregation window from the monitoring config section.
    """
    mon_conf = conf["monitoring"]
    database, schema = mr.location.split(".")
    src_db = conf["snowflake"]["database"]
    src_schema = conf["snowflake"]["schema"]

    # Set session context to the model registry schema so internal SQL resolves the model
    session.sql(f"USE DATABASE {database}").collect()
    session.sql(f"USE SCHEMA {schema}").collect()

    model = mr.get_model(model_name)
    mv = model.version(version_name)

    prediction_fqn = f"{src_db}.{src_schema}.{mon_conf['prediction_table']}"
    baseline_table = mon_conf.get("baseline_table")
    baseline_fqn = f"{src_db}.{src_schema}.{baseline_table}" if baseline_table else None

    source_config = ModelMonitorSourceConfig(
        source=prediction_fqn,
        timestamp_column=mon_conf["timestamp_column"],
        id_columns=mon_conf["id_columns"],
        prediction_score_columns=mon_conf["prediction_columns"],
        actual_score_columns=mon_conf["actual_columns"],
        baseline=baseline_fqn,
        segment_columns=mon_conf.get("segment_columns"),
    )

    monitor_config = ModelMonitorConfig(
        model_version=mv,
        model_function_name="predict",
        background_compute_warehouse_name=mon_conf["background_warehouse"],
        refresh_interval=mon_conf.get("refresh_interval", "1 hour"),
        aggregation_window=mon_conf.get("aggregation_window", "1 day"),
    )

    monitor = mr.add_monitor(
        name=f"{model_name.split('.')[-1]}_MONITOR",
        source_config=source_config,
        model_monitor_config=monitor_config,
    )
    print(f"Monitor created for {model_name}/{version_name}")
    return monitor
