import json
from datetime import datetime

import snowflake.snowpark.functions as F
from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore, FeatureView
from snowflake.ml.registry import Registry


def create_model_registry(session, database: str, schema: str) -> Registry:
    """Create a Model Registry in the given schema, creating the schema if needed."""
    cs = session.get_current_schema()
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}").collect()
    mr = Registry(session=session, database_name=database, schema_name=schema)
    session.sql(f"USE SCHEMA {cs}").collect()
    print(f"Model Registry ({schema}) ready")
    return mr


def create_feature_store(session, database: str, schema: str, warehouse: str) -> FeatureStore:
    """Initialize a Feature Store, creating the schema if it does not exist."""
    fs = FeatureStore(
        session,
        database,
        schema,
        warehouse,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    print(f"Feature Store ({schema}) ready")
    return fs


def register_entity(fs: FeatureStore, conf: dict) -> Entity:
    """Register the primary entity (e.g. CUSTOMER) in the Feature Store if not already present."""
    entity_name = conf["feature_store"]["entity_name"]
    join_keys = conf["feature_store"]["entity_join_keys"]
    entities = json.loads(fs.list_entities().select(F.to_json(F.array_agg("NAME", True))).collect()[0][0])
    if entity_name not in entities:
        entity = Entity(name=entity_name, join_keys=join_keys, desc=f"Primary Key for {entity_name}")
        fs.register_entity(entity)
    else:
        entity = fs.get_entity(entity_name)
    return entity


def register_feature_view(fs: FeatureStore, entity: Entity, feature_df, conf: dict) -> FeatureView:
    """Register (or overwrite) a FeatureView backed by a Dynamic Table with scheduled refresh."""
    fv_name = conf["feature_store"]["feature_view_name"]
    fv_version = conf["feature_store"]["feature_view_version"]
    refresh_freq = conf["feature_store"]["refresh_freq"]

    feature_desc = {
        "AVERAGE_ORDER_PER_MONTH": "Average number of orders per month",
        "DAYS_SINCE_LAST_PURCHASE": "Days since last purchase",
        "DAYS_SINCE_SIGNUP": "Days since signup",
        "EXPECTED_DAYS_BETWEEN_PURCHASES": "Expected days between purchases",
        "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE": "Days since expected last purchase date",
    }

    fv_instance = FeatureView(
        name=fv_name,
        entities=[entity],
        feature_df=feature_df,
        timestamp_col="BEHAVIOR_UPDATED_AT",
        refresh_freq=refresh_freq,
        desc="Customer Modelling Features",
    ).attach_feature_desc(feature_desc)

    fv = fs.register_feature_view(
        feature_view=fv_instance,
        version=fv_version,
        block=True,
        overwrite=True,
    )
    return fv


def generate_training_dataset(session, fs: FeatureStore, feature_view: FeatureView, conf: dict):
    """Generate a versioned training dataset from a FeatureView using a spine DataFrame.

    Returns:
        Tuple of (training Snowpark DataFrame, Feature Store schema name).
    """
    from src.utils.versioning import dataset_check_and_update

    dataset_name = conf["feature_store"]["dataset_name"]
    schema_name = fs.list_feature_views().to_pandas()["SCHEMA_NAME"][0]
    dataset_version = dataset_check_and_update(session, dataset_name, schema_name=schema_name)

    spine_sdf = get_spine_df(feature_view)

    training_dataset = fs.generate_dataset(
        name=dataset_name,
        version=dataset_version,
        spine_df=spine_sdf,
        features=[feature_view],
        spine_timestamp_col="ASOF_DATE",
    )
    training_dataset_sdf = training_dataset.read.to_snowpark_dataframe()
    return training_dataset_sdf, schema_name


def get_spine_df(feature_view):
    """Build a spine DataFrame with one row per CUSTOMER_ID and today's date as ASOF_DATE."""
    asof_date = datetime.now()
    spine_sdf = feature_view.feature_df.group_by("CUSTOMER_ID").agg(
        F.lit(asof_date.strftime("%Y-%m-%d")).as_("ASOF_DATE")
    )
    return spine_sdf
