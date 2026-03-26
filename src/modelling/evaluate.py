import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Compute MAE, MAPE, and R² for a trained model on a validation set."""
    y_pred = model.predict(X_test)
    return {
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }
