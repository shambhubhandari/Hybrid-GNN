"""Regression metrics: R2 / MAE / MSE."""
from typing import Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return R2, MAE, and MSE for the predictions."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }
