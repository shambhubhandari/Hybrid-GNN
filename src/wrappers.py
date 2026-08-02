# Copied verbatim from src/models/wrappers.py.
class XGBWrapper:
    """Gives a raw xgboost Booster a .predict(X) that accepts numpy/DataFrame.

    train_xgboost() returns a Booster, whose predict() wants a DMatrix; this
    adapts it to the sklearn-style .predict(X) the harness and SHAP expect.
    """

    def __init__(self, booster):
        self._b = booster

    def predict(self, X):
        import xgboost as _xgb
        return self._b.predict(_xgb.DMatrix(X))
