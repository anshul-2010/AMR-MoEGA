"""
High-level orchestration for training static ML baselines with CV and logging.
"""
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils.logger import get_logger
from utils.stats_utils import binary_metrics
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

logger = get_logger("static_ml")


def train_cv(features_csv: str, labels_csv: str, out_dir: str, n_splits=5):
    X = pd.read_csv(features_csv)
    y = pd.read_csv(labels_csv).iloc[:, 0]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold = 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[val_idx])
        metrics = binary_metrics(y.iloc[val_idx], preds)
        logger.info(f"Fold {fold} metrics: {metrics}")
        joblib.dump(clf, f"{out_dir}/model_fold{fold}.joblib")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_splits", type=int, default=5)
    args = p.parse_args()
    train_cv(args.features, args.labels, args.out_dir, args.n_splits)
