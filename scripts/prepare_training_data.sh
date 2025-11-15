#!/usr/bin/env bash
set -euo pipefail
# Convert processed features into train/test splits and place into experiments/
FEATURE_CSV=${1:-data/processed/features/features.csv}
LABELS=${2:-data/processed/labels/labels.csv}
OUT_DIR=${3:-data/processed/train_test_split}
mkdir -p ${OUT_DIR}
python - <<PY
from sklearn.model_selection import train_test_split
import pandas as pd
X = pd.read_csv("${FEATURE_CSV}")
y = pd.read_csv("${LABELS}").iloc[:,0]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
Xtr.to_csv("${OUT_DIR}/X_train.csv", index=False)
Xte.to_csv("${OUT_DIR}/X_test.csv", index=False)
ytr.to_csv("${OUT_DIR}/y_train.csv", index=False)
yte.to_csv("${OUT_DIR}/y_test.csv", index=False)
print("Prepared train/test split")
PY
echo "Train/test split files saved to ${OUT_DIR}"