"""The 70/15/15 stratified holdout shared by every stage."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logging_util import get_logger

logger = get_logger(__name__)


def stratified_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    quantiles: int = 10,
    random_state: int = 42,
):
    """70/15/15 split, stratified on qcut bins of the target."""
    y_bins = pd.qcut(y, q=quantiles, duplicates="drop", labels=False)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state, stratify=y_bins
    )
    rel_val = val_size / (1.0 - test_size)
    y_bins_tv = pd.qcut(y_tv, q=quantiles, duplicates="drop", labels=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, shuffle=True, random_state=random_state, stratify=y_bins_tv
    )
    logger.info("Holdout split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
