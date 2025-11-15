"""
A simple expert classifier wrapper (scikit-learn compatible).
"""
from sklearn.ensemble import GradientBoostingClassifier
from typing import Any


class ExpertClassifier:
    def __init__(self, params: dict = None):
        self.params = params or {}
        self.model = GradientBoostingClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
