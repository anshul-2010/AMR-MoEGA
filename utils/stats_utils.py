"""
Statistical utilities and common metrics wrappers.
"""
from typing import Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)


def binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    res = {}
    res["accuracy"] = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    res["precision"] = p
    res["recall"] = r
    res["f1"] = f1
    if y_prob is not None:
        try:
            res["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            res["roc_auc"] = None
    res["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return res


def train_test_split_stratified(X, y, test_size=0.2, random_state=0):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
