"""
Gating network skeleton. In practice this can be a small NN in PyTorch/TF
or a logistic regression on features producing mixture weights.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class GatingNetwork:
    def __init__(self, n_experts):
        self.n_experts = n_experts
        # For skeleton, use one-vs-rest logistic regressors to predict expert responsibilities
        self.model = LogisticRegression(multi_class="multinomial", max_iter=200)

    def fit(self, X, responsibilities):
        # responsibilities: shape (n_samples,) integer labels for chosen expert (or soft labels)
        self.model.fit(X, responsibilities)

    def predict_weights(self, X):
        # returns probabilities per expert (n_samples, n_experts)
        return self.model.predict_proba(X)
