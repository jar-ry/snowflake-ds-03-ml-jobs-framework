from snowflake.ml.registry import Registry


def deploy_inference_service(
    session,
    mr: Registry,
    model_name: str,
    version_name: str,
    pool_name: str,
    service_name: str,
    ingress_enabled: bool = True,
):
    """Deploy a model version as a SPCS service for inference.

    Drops any existing service with the same name first so it always
    reflects the current promoted model version.
    """
    model = mr.get_model(model_name)
    mv = model.version(version_name)

    # Build fully qualified service name
    db, schema = mr.location.split(".")
    fqn_service = f"{db}.{schema}.{service_name}"

    # Drop existing service if it exists (from any version)
    session.sql(f"DROP SERVICE IF EXISTS {fqn_service}").collect()

    mv.create_service(
        service_name=service_name,
        service_compute_pool=pool_name,
        ingress_enabled=ingress_enabled,
    )
    print(f"Service '{fqn_service}' deployed for {model_name}/{version_name}")
    return service_name


def run_batch_predictions(
    session,
    mr: Registry,
    model_name: str,
    input_table: str,
    output_table: str,
    service_name: str,
    prediction_column: str = "PREDICTION",
):
    """Run batch predictions via SPCS service."""
    model = mr.get_model(model_name)
    mv = model.default
    input_df = session.table(input_table)
    predictions = mv.run(input_df, function_name="predict", service_name=service_name)
    # Rename the generic output column to a meaningful name
    output_cols = [c for c in predictions.columns if c not in input_df.columns]
    if output_cols:
        predictions = predictions.with_column_renamed(output_cols[0], prediction_column)
    predictions.write.mode("overwrite").save_as_table(output_table)
    print(f"Predictions saved to {output_table}")
    return predictions
