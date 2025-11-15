#!/usr/bin/env bash
set -euo pipefail

# Usage: ./call_variants.sh config/bioinfo_config.yaml /path/to/bam_dir /path/to/out_vcf_dir

CONFIG=${1:-config/bioinfo_config.yaml}
BAM_DIR=${2:-data/intermediate/aligned_reads}
OUT_VCF_DIR=${3:-data/intermediate/variants}

if [[ ! -f "$CONFIG" ]]; then
  echo "Config YAML not found: $CONFIG"
  exit 1
fi

# parse simple values from config using python
REF_FA=$(python3 - <<PY
import yaml
print(yaml.safe_load(open("$CONFIG"))["reference"]["fasta"])
PY
)

THREADS=$(python3 - <<PY
import yaml
print(yaml.safe_load(open("$CONFIG"))["pipeline"]["threads"])
PY
)

mkdir -p "$OUT_VCF_DIR"

echo "Reference: $REF_FA"
echo "Threads: $THREADS"
echo "BAM dir: $BAM_DIR"
echo "VCF out dir: $OUT_VCF_DIR"

for bam in "${BAM_DIR}"/*.bam; do
  if [[ ! -f "$bam" ]]; then
    echo "No BAM files found in $BAM_DIR"
    break
  fi
  sample=$(basename "$bam" .bam)
  out_vcf="${OUT_VCF_DIR}/${sample}.vcf.gz"
  echo "Calling variants for sample: $sample"
  # mpileup then call (SNP+indel). Adjust bcftools options if needed.
  bcftools mpileup -f "$REF_FA" "$bam" -Ou -a FORMAT/DP | \
    bcftools call -mv -Oz -o "$out_vcf" --threads "$THREADS"
  bcftools index "$out_vcf"
done

echo "Variant calling finished. Output in: $OUT_VCF_DIR"