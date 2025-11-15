"""
Cross validation utilities and wrappers for reproducible evaluation.
"""
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils.logger import get_logger
from utils.stats_utils import binary_metrics
from sklearn.ensemble import RandomForestClassifier

logger = get_logger("cv")


def evaluate_cv(features_csv, labels_csv, n_splits=5):
    X = pd.read_csv(features_csv)
    y = pd.read_csv(labels_csv).iloc[:, 0]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    metrics = []
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        metrics.append(binary_metrics(y.iloc[test_idx], preds))
    return metrics


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res = evaluate_cv(args.features, args.labels)
    with open(args.out, "w") as f:
        import json

        json.dump(res, f, indent=2)
    logger.info(f"Wrote CV results to {args.out}")
