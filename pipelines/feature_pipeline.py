from src.feature_engineering.feature_store import (
    create_feature_store,
    create_model_registry,
    generate_training_dataset,
    register_entity,
    register_feature_view,
)
from src.features.data_loader import load_data
from src.features.preprocessing import pre_process
from src.session import create_session


def run(conf: dict):
    """Run the feature pipeline: load data, preprocess, register FeatureView, generate dataset."""
    print("=" * 60)
    print("FEATURE PIPELINE")
    print("=" * 60)

    session, database, schema, warehouse = create_session(conf)

    fs_schema = conf["feature_store"]["schema"]
    mr_schema = conf["model_registry"]["schema"]

    mr = create_model_registry(session, database, mr_schema)
    fs = create_feature_store(session, database, fs_schema, warehouse)

    print("\n[1/4] Loading raw data...")
    fqn_prefix = f"{database}.{schema}"
    customer_data = session.table(f"{fqn_prefix}.CUSTOMERS")
    behavior_data = session.table(f"{fqn_prefix}.PURCHASE_BEHAVIOR")
    raw_data = load_data(customer_data, behavior_data)

    print("[2/4] Preprocessing...")
    processed_data = pre_process(raw_data)

    print("[3/4] Registering feature view...")
    entity = register_entity(fs, conf)
    feature_view = register_feature_view(fs, entity, processed_data, conf)

    print("[4/4] Generating training dataset...")
    training_sdf, fs_schema_name = generate_training_dataset(session, fs, feature_view, conf)

    row_count = training_sdf.count()
    print(f"\nFeature pipeline complete. Training dataset: {row_count} rows")
    return session, mr, fs, feature_view
