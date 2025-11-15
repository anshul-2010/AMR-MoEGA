#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1   # path to config YAML
SAMPLES_DIR=$2  # directory with raw fastq (paired)
OUT_BAM_DIR=$3

# read config
REF=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG}"))
print(cfg["data"]["reference"])
PY
)

THREADS=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG}"))
print(cfg["pipeline"]["threads"])
PY
)

for fq1 in "${SAMPLES_DIR}"/*_R1*.fastq.gz; do
  base=$(basename "${fq1}" "_R1.fastq.gz")
  fq2="${SAMPLES_DIR}/${base}_R2.fastq.gz"
  out1="${OUT_BAM_DIR}/${base}_trim_R1.fastq.gz"
  out2="${OUT_BAM_DIR}/${base}_trim_R2.fastq.gz"
  bam_out="${OUT_BAM_DIR}/${base}.bam"

  echo "Trimming sample $base"
  fastp -i "${fq1}" -I "${fq2}" -o "${out1}" -O "${out2}" -w "${THREADS}" --detect_adapter_for_pe

  echo "Aligning sample $base"
  bwa mem -t "${THREADS}" "${REF}" "${out1}" "${out2}" \
    | samtools view -bS - \
    | samtools sort -@ "${THREADS}" -o "${bam_out}"
  samtools index "${bam_out}"
done