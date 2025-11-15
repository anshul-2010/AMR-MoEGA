#!/usr/bin/env bash
set -euo pipefail

# Usage: ./filter_and_annotate.sh config/bioinfo_config.yaml /path/to/vcf_dir /path/to/filtered_dir /path/to/annotated_dir

CONFIG=${1:-config/bioinfo_config.yaml}
VCF_DIR=${2:-data/intermediate/variants}
FILTERED_DIR=${3:-data/intermediate/variants_filtered}
ANNOTATED_DIR=${4:-data/intermediate/annotated_variants}

if [[ ! -f "$CONFIG" ]]; then
  echo "Config YAML not found: $CONFIG"
  exit 1
fi

DP_TH=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["variant_calling"]["min_dp"])
PY
)

QUAL_TH=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["variant_calling"]["min_qual"])
PY
)

SNPEFF_DB=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["reference"]["snpeff_db"])
PY
)

mkdir -p "$FILTERED_DIR" "$ANNOTATED_DIR"

echo "Filtering VCFs in $VCF_DIR with DP >= $DP_TH and QUAL >= $QUAL_TH"
echo "Annotating using snpEff DB: $SNPEFF_DB"

for v in "${VCF_DIR}"/*.vcf.gz; do
  if [[ ! -f "$v" ]]; then
    echo "No VCFs found in $VCF_DIR"
    break
  fi
  base=$(basename "$v" .vcf.gz)
  filtered="${FILTERED_DIR}/${base}.filtered.vcf.gz"
  annotated="${ANNOTATED_DIR}/${base}.annotated.vcf.gz"

  echo "Processing $base"
  bcftools view "$v" -Ou | \
    bcftools filter -e "INFO/DP<${DP_TH} || QUAL<${QUAL_TH}" -Oz -o "$filtered"
  bcftools index "$filtered"

  # snpEff: output gz VCF
  # snpEff can read gz input; using java -jar if snpEff binary not in PATH.
  if command -v snpEff >/dev/null 2>&1; then
    snpEff -v "$SNPEFF_DB" "$filtered" | bgzip -c > "$annotated"
  else
    # try typical jar location (user may need to adjust)
    if [[ -f "/usr/share/java/snpEff.jar" ]]; then
      java -Xmx4g -jar /usr/share/java/snpEff.jar -v "$SNPEFF_DB" "$filtered" | bgzip -c > "$annotated"
    else
      echo "snpEff not found in PATH and default jar location not present. Please install or adjust script."
      exit 1
    fi
  fi
  bcftools index "$annotated"
done

echo "Filtering and annotation completed. Filtered: $FILTERED_DIR, Annotated: $ANNOTATED_DIR"