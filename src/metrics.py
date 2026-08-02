# Copied from src/evaluation/metrics.py — the paper's reported metric triple,
# computed one way so every model is scored identically.
from typing import Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """R2 / MAE / MSE, matching the paper's reported triple."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }
