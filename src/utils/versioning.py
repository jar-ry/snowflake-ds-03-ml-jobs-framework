import ast

from snowflake.ml._internal.exceptions import dataset_errors
from snowflake.ml.dataset import Dataset


def check_and_update(df, model_name: str) -> str:
    """Return the next version string (e.g. V_2) for a model based on existing registry versions."""
    if "." in model_name:
        model_name = model_name.split(".")[-1]
    if df.empty or df[df["name"] == model_name].empty:
        return "V_1"
    list_of_lists = df["versions"].apply(ast.literal_eval)
    all_versions = [v for sublist in list_of_lists for v in sublist]
    nums = sorted(int(v.rsplit("_", 1)[-1]) for v in all_versions)
    return f"V_{nums[-1] + 1}"


def dataset_check_and_update(session, dataset_name: str, schema_name: str = None) -> str:
    """Return the next version string for a Dataset, or V_1 if it does not exist yet."""
    if schema_name is None:
        schema_name = session.get_current_schema()
    full_name = f"{session.get_current_database()}.{schema_name}.{dataset_name}"
    try:
        ds = Dataset.load(session=session, name=full_name)
        versions = ds.list_versions()
    except dataset_errors.DatasetNotExistError:
        return "V_1"
    if len(versions) == 0:
        return "V_1"
    nums = sorted(int(v.rsplit("_", 1)[-1]) for v in versions)
    return f"V_{nums[-1] + 1}"


def get_latest(df, model_name: str) -> str:
    """Return the next incremented version string based on the latest existing version."""
    if df.empty or df[df["name"] == model_name].empty:
        return "V_1"
    raw_versions = ast.literal_eval(df["versions"][0])
    lst = sorted(raw_versions, key=lambda x: int(x.rsplit("_", 1)[-1]))
    prefix, num = lst[-1].rsplit("_", 1)
    return f"{prefix}_{int(num) + 1}"
