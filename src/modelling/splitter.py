import pandas as pd
from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.dataset import Dataset, load_dataset


def create_data_connector(session, dataset_name: str) -> DataConnector:
    """Load the latest version of a Dataset and return a DataConnector for it."""
    ds = Dataset.load(session=session, name=dataset_name)
    ds_latest_version = str(ds.list_versions()[-1])
    ds_df = load_dataset(session, dataset_name, ds_latest_version)
    return DataConnector.from_dataset(ds_df)


def generate_train_val_set(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and validation sets.

    Returns:
        Tuple of (train_df, val_df) each containing feature and target columns.
    """
    from sklearn.model_selection import train_test_split

    X = dataframe[feature_columns]
    y = dataframe[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_test, y_test], axis=1)
    return train_df, val_df
