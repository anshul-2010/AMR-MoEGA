"""
Simple regression expert placeholder.
"""
from sklearn.ensemble import GradientBoostingRegressor


class ExpertRegressor:
    def __init__(self, params: dict = None):
        self.model = GradientBoostingRegressor(**(params or {}))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
