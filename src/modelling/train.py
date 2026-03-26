"""
ML Job entrypoint for HPO training via submit_directory.

Usage (submitted by pipelines/training_pipeline.py):
    submit_directory("./", pool, entrypoint="src/modelling/train.py")

The entire project directory is uploaded, so conf/parameters.yml is available
inside the container. All column lists, hyperparameters, and training config
are read from that YAML — nothing is hardcoded here.
"""

import os
import sys

import yaml


def _ensure_src_on_path():
    """Ensure src/ is on sys.path — needed in both main process and Ray workers."""
    src_via_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_via_cwd = os.path.join(os.getcwd(), "src")
    for p in (src_via_file, src_via_cwd):
        if p not in sys.path:
            sys.path.insert(0, p)


# Set up path in main process for module-level usage
_ensure_src_on_path()

from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.experiment.callback.xgboost import SnowflakeXgboostCallback
from snowflake.ml.model.model_signature import infer_signature
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from modelling.evaluate import evaluate_model
from modelling.pipeline import build_pipeline
from modelling.splitter import create_data_connector, generate_train_val_set


def _load_conf() -> dict:
    """Load parameters.yml from the project root (available via submit_directory)."""
    # Try __file__-based resolution first, then fall back to cwd-based
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "conf", "parameters.yml"
        ),
        os.path.join(os.getcwd(), "conf", "parameters.yml"),
    ]
    for conf_path in candidates:
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"parameters.yml not found. Tried: {candidates}")


def train():
    """HPO trial function executed by each Ray worker.

    Loads config from parameters.yml, retrieves hyperparameters from the tuner context,
    trains an XGBRegressor pipeline, evaluates it, logs metrics and the model to
    Experiment Tracking and the Model Registry, then reports results back to the tuner.
    """
    from snowflake.ml.modeling import tune

    # Ensure path is set in Ray worker processes too
    _ensure_src_on_path()

    conf = _load_conf()
    modelling = conf["modelling"]
    target_column = modelling["target_column"]

    session = Session.builder.getOrCreate()
    tuner_context = tune.get_tuner_context()
    params = tuner_context.get_hyper_params()
    dm = tuner_context.get_dataset_map()
    model_name = params.pop("model_name")
    mr_schema_name = params.pop("mr_schema_name")
    experiment_name = params.pop("experiment_name")

    exp = ExperimentTracking(session=session, schema_name=mr_schema_name)
    exp.set_experiment(experiment_name)

    with exp.start_run() as run:
        train_data = dm["train"].to_pandas()
        val_data = dm["val"].to_pandas()

        X_train = train_data.drop(target_column, axis=1)
        y_train = train_data[target_column]
        X_val = val_data.drop(target_column, axis=1)
        y_val = val_data[target_column]

        sig = infer_signature(X_train, y_train)
        callback = SnowflakeXgboostCallback(
            exp,
            model_name=model_name,
            model_signature=sig,
        )
        params["callbacks"] = [callback]

        model = build_pipeline(
            model_params=params,
            numerical_columns=modelling["numerical_columns"],
            categorical_columns=modelling["categorical_columns"],
            ordinal_columns=modelling["ordinal_columns"],
            ordinal_categories=modelling["ordinal_categories"],
        )
        exp.log_params(params)

        print("Training model...", end="")
        model.fit(X_train, y_train)

        print("Evaluating model...", end="")
        metrics = evaluate_model(model, X_val, y_val)

        print("Log metrics...", end="")
        exp.log_metrics(metrics)
        metrics["run_name"] = run.name

        exp.log_model(
            model=model,
            model_name=model_name,
            version_name=run.name,
            sample_input_data=X_train,
        )

        # Set metrics on the model version in the registry so
        # promotion pipeline can find them via mv.show_metrics()
        mr = Registry(
            session=session,
            database_name=session.get_current_database(),
            schema_name=mr_schema_name,
        )
        mv = mr.get_model(model_name).version(run.name)
        for metric_name, metric_value in metrics.items():
            if metric_name == "run_name":
                continue
            mv.set_metric(metric_name, metric_value)
        print(f"Metrics set on model version {run.name}")

        tuner_context.report(metrics=metrics, model="model")


if __name__ == "__main__":
    from snowflake.ml.modeling import tune
    from snowflake.ml.modeling.tune.search import RandomSearch

    conf = _load_conf()
    modelling = conf["modelling"]
    compute = conf["compute"]
    fs = conf["feature_store"]
    database = conf["snowflake"]["database"]

    session = Session.builder.getOrCreate()

    # Build dataset FQN from config
    dataset_fqn = f"{database}.{fs['schema']}.{fs['dataset_name']}"

    print(f"Loading data from {dataset_fqn}...", end="", flush=True)
    dc = create_data_connector(session, dataset_name=dataset_fqn)
    df = dc.to_pandas()

    print("Building train/val data")
    train_df, val_df = generate_train_val_set(
        df,
        feature_columns=modelling["feature_columns"],
        target_column=modelling["target_column"],
        test_size=modelling.get("test_size", 0.2),
        random_state=modelling.get("random_state", 42),
    )

    dataset_map = {
        "train": DataConnector.from_dataframe(session.create_dataframe(train_df)),
        "val": DataConnector.from_dataframe(session.create_dataframe(val_df)),
    }

    # Build HPO search space from config — each key maps to a list of
    # candidate values that get wrapped in tune.choice()
    hpo_conf = conf.get("hpo", {})
    search_space = {
        "mr_schema_name": conf["model_registry"]["schema"],
        "model_name": modelling["model_name"],
        "experiment_name": modelling["experiment_name"],
    }
    for param_name, candidates in hpo_conf.items():
        search_space[param_name] = tune.choice(candidates)

    tuner_config = tune.TunerConfig(
        metric=modelling.get("tuning_metric", "mean_absolute_percentage_error"),
        mode=modelling.get("tuning_mode", "min"),
        search_alg=RandomSearch(),
        num_trials=compute.get("num_trials", 10),
    )

    tuner = tune.Tuner(
        train_func=train,
        search_space=search_space,
        tuner_config=tuner_config,
    )

    print("HPO starting")
    results = tuner.run(dataset_map=dataset_map)
    print("HPO DONE")
    print(results.results)
