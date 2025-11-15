#!/usr/bin/env bash
set -euo pipefail
# Example wrapper to run the entire pipeline in order
# Usage: ./run_pipeline.sh config.yaml

CONFIG=${1:-config.yaml}

echo "Using config: $CONFIG"
# Example steps (customize paths in config)
python pipeline/00_download/download_microbigg_data.py --out_dir data/raw
# Preprocessing (loop over samples in real script)
# Align / Trim placeholders
# Variants
# Feature engineering
# Modeling
# Evaluation
echo "Pipeline finished (skeleton). Customize run_pipeline.sh to your needs."