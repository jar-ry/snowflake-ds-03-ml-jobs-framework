def _quote_id(name: str) -> str:
    """Double-quote a Snowflake identifier to prevent SQL injection."""
    return '"' + name.replace('"', '""') + '"'


def create_inference_procedure(session, conf: dict):
    """Create a stored procedure that runs batch inference via SPCS service."""
    database = conf["snowflake"]["database"]
    mr_schema = conf["model_registry"]["schema"]
    model_name = conf["modelling"]["model_name"]
    fs_schema = conf["feature_store"]["schema"]
    fv_name = conf["feature_store"]["feature_view_name"]
    fv_version = conf["feature_store"]["feature_view_version"]
    prediction_table = conf["monitoring"]["prediction_table"]
    procedure_name = conf["scheduling"]["procedure_name"]
    service_name = conf["serving"]["service_name"]

    input_table = f"{database}.{fs_schema}.{fv_name}${fv_version}"
    fqn_procedure = f"{_quote_id(database)}.{_quote_id(mr_schema)}.{_quote_id(procedure_name)}"

    # Values embedded in the procedure body are string literals (not identifiers),
    # so we escape single quotes to prevent SQL injection inside the $$ block.
    safe_database = database.replace("'", "''")
    safe_mr_schema = mr_schema.replace("'", "''")
    safe_model_name = model_name.replace("'", "''")
    safe_input_table = input_table.replace("'", "''")
    safe_service_name = service_name.replace("'", "''")
    safe_prediction_table = prediction_table.replace("'", "''")

    sp_sql = f"""
    CREATE OR REPLACE PROCEDURE {fqn_procedure}()
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = ('snowflake-ml-python', 'snowflake-snowpark-python')
    HANDLER = 'run_inference'
    AS
    $$
def run_inference(session):
    from snowflake.ml.registry import Registry

    mr = Registry(
        session=session,
        database_name="{safe_database}",
        schema_name="{safe_mr_schema}",
    )
    model = mr.get_model("{safe_model_name}")
    mv = model.default
    if mv is None:
        return "No default model version set. Skipping."

    input_df = session.table("{safe_input_table}")
    predictions = mv.run(
        input_df,
        function_name="predict",
        service_name="{safe_service_name}",
    )

    # Rename generic output column to PREDICTION
    input_cols = input_df.columns
    output_cols = [c for c in predictions.columns if c not in input_cols]
    if output_cols:
        predictions = predictions.with_column_renamed(output_cols[0], "PREDICTION")

    predictions.write.mode("overwrite").save_as_table("{safe_prediction_table}")

    row_count = session.table("{safe_prediction_table}").count()
    return f"Batch inference complete. {{row_count}} predictions written using {{mv.version_name}}."
    $$;
    """
    session.sql(sp_sql).collect()
    print(f"Stored procedure '{fqn_procedure}' created.")
    return fqn_procedure


def create_inference_task(session, conf: dict, procedure_fqn: str):
    """Create a Snowflake Task that runs the inference procedure on a cron schedule (SUSPENDED)."""
    database = conf["snowflake"]["database"]
    mr_schema = conf["model_registry"]["schema"]
    sched = conf["scheduling"]
    task_name = f"{_quote_id(database)}.{_quote_id(mr_schema)}.{_quote_id(sched['task_name'])}"
    warehouse = _quote_id(sched["warehouse"])
    cron = sched["cron"]
    timezone = sched.get("timezone", "UTC")

    task_sql = f"""
    CREATE OR REPLACE TASK {task_name}
        WAREHOUSE = {warehouse}
        SCHEDULE = 'USING CRON {cron} {timezone}'
    AS
        CALL {procedure_fqn}();
    """
    session.sql(task_sql).collect()
    print(f"Task '{task_name}' created (SUSPENDED).")
    print(f"  Schedule : {cron} ({timezone})")
    print(f"  Warehouse: {warehouse}")
    print(f"  To activate: ALTER TASK {task_name} RESUME;")
    return task_name
