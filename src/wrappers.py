"""Adapter that gives a raw xgboost Booster a sklearn-style .predict(X)."""


class XGBWrapper:
    """Wrap a Booster so .predict(X) accepts a numpy array or DataFrame."""

    def __init__(self, booster):
        self._b = booster

    def predict(self, X):
        import xgboost as _xgb
        return self._b.predict(_xgb.DMatrix(X))
