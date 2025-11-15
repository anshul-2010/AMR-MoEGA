#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/convert_vcf_to_matrix.sh config/bioinfo_config.yaml

CONFIG=${1:-config/bioinfo_config.yaml}
FILTERED_DIR=$(python3 - <<PY
import yaml
cfg=yaml.safe_load(open("$CONFIG"))
print(cfg["data"]["filtered_vcf_dir"])
PY
)
OUT_MATRIX=$(python3 - <<PY
import yaml
cfg=yaml.safe_load(open("$CONFIG"))
print(cfg["data"]["snp_matrix_csv"])
PY
)

echo "Building SNP matrix from VCFs in: $FILTERED_DIR"
python3 pipeline/03_feature_engineering/build_snp_matrix_ml_iamr.py --config "$CONFIG" --vcf_dir "$FILTERED_DIR" --out_csv "$OUT_MATRIX"
echo "SNP matrix written to: $OUT_MATRIX"