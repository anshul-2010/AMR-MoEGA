"""
Common scoring helpers for evaluating architectures and models.
"""
from utils.stats_utils import binary_metrics


def score_classification(y_true, y_pred, y_prob=None):
    return binary_metrics(y_true, y_pred, y_prob)
