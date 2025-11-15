"""
Baseline ML models (sklearn-based).
Provides wrappers for training and prediction.
"""
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from utils.logger import get_logger
from utils.stats_utils import binary_metrics
from pathlib import Path
import joblib

logger = get_logger("baseline_models")

MODEL_MAP = {
    "rf": RandomForestClassifier,
    "lr": LogisticRegression,
}


def train_baseline(features_csv: str, labels_csv: str, out_model: str, model_type="rf"):
    X = pd.read_csv(features_csv)
    y = pd.read_csv(labels_csv).iloc[:, 0]
    Model = MODEL_MAP.get(model_type, RandomForestClassifier)
    model = Model()
    logger.info(f"Training baseline model {model_type}")
    model.fit(X, y)
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)
    logger.info(f"Saved model to {out_model}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_model", required=True)
    p.add_argument("--model_type", default="rf")
    args = p.parse_args()
    train_baseline(args.features, args.labels, args.out_model, args.model_type)
