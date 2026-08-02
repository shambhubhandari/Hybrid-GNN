# Copied from src/models/gbdt.py — train_xgboost (the only regressor the ablation
# retrains live). Same hyper-parameters as the paper's pipeline (src.config).
from src.logging_util import get_logger

logger = get_logger(__name__)


def train_xgboost(X_train, y_train, X_val, y_val, params: dict):
    """Best individual model on the MEGNet feature set (paper: R2 = 0.921)."""
    import xgboost as xgb

    # Pull training-control params before passing the rest to xgb.train.
    num_boost_round = int(params.pop("num_boost_round", 1000))
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]

    logger.info("Training XGBoost: num_boost_round=%d, early_stopping_rounds=%d",
                num_boost_round, early_stopping_rounds)
    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=num_boost_round,
        evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False,
    )
    logger.info("XGBoost finished: best_iteration=%d, best_score=%.4f",
                booster.best_iteration, booster.best_score)
    return booster
