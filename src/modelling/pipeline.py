import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def build_pipeline(
    model_params: dict,
    numerical_columns: list[str],
    categorical_columns: list[str],
    ordinal_columns: list[str],
    ordinal_categories: dict[str, list[str]],
) -> Pipeline:
    """Build a sklearn Pipeline with a ColumnTransformer preprocessor and XGBRegressor.

    Applies MinMaxScaler to numerical columns, OneHotEncoder to categorical columns,
    and OrdinalEncoder (with explicit category ordering) to ordinal columns.
    """
    explicit_categories = [ordinal_categories[col] for col in ordinal_columns]
    ordinal_encoder = OrdinalEncoder(categories=explicit_categories, dtype=int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("NUM", MinMaxScaler(), numerical_columns),
            ("CAT", OneHotEncoder(), categorical_columns),
            ("ORD", ordinal_encoder, ordinal_columns),
        ],
        remainder="passthrough",
    )

    model = xgb.XGBRegressor(**model_params)
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])
