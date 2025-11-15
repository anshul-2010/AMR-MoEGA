#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-experiments/model.joblib}
FEATURES=${2:-data/processed/train_test_split/X_test.csv}
LABELS=${3:-data/processed/train_test_split/y_test.csv}
python - <<PY
import joblib, pandas as pd
from utils.stats_utils import binary_metrics
model = joblib.load("${MODEL}")
X = pd.read_csv("${FEATURES}")
y = pd.read_csv("${LABELS}").iloc[:,0]
preds = model.predict(X)
probs = None
try:
    probs = model.predict_proba(X)[:,1]
except Exception:
    pass
print(binary_metrics(y, preds, probs))
PY
