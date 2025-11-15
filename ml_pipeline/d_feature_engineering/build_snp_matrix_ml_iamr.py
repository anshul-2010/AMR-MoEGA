#!/usr/bin/env python3
"""
Given a directory with per-sample annotated VCFs (gzipped), builds a SNP matrix
with rows = variants (CHROM:POS:REF:ALT) and columns = samples.

Output is written as CSV to path specified in config or CLI.

Requires: cyvcf2, pandas, pyyaml
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from cyvcf2 import VCF
import yaml
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("build_snp_matrix")


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def genotype_to_allele_count(gt_tuple):
    # cyvcf2 rec.genotypes entries are lists: [GT1, GT2, phased, ...] for diploid
    # For haploid (bacterial) VCFs, cyvcf2 may present single allele per sample
    # We'll attempt to compute allele count of ALT alleles robustly.
    try:
        if gt_tuple is None:
            return None
        # If genotype is like [0, 1, ...] or [1,0,...], sum non-ref alleles
        alleles = [a for a in gt_tuple if isinstance(a, (int, float))]
        if len(alleles) == 0:
            return None
        # treat missing as None (represented by None or -1 or '.')
        # convert floats or ints; ignore negative values
        alleles = [int(a) for a in alleles if int(a) >= 0]
        return sum(alleles)
    except Exception:
        return None


def build_matrix(vcf_dir, out_csv, missing_as=0):
    vcf_dir = Path(vcf_dir)
    assert vcf_dir.exists(), f"VCF directory not found: {vcf_dir}"
    sample_files = sorted([p for p in vcf_dir.glob("*.vcf.gz")])
    if not sample_files:
        sample_files = sorted([p for p in vcf_dir.glob("*.vcf")])
    if not sample_files:
        raise FileNotFoundError(f"No VCFs found in {vcf_dir}")

    samples = [
        p.stem.replace(".filtered", "").replace(".annotated", "") for p in sample_files
    ]
    logger.info(f"Found {len(samples)} VCF files. Samples: {samples}")

    # We'll first collect variant keys and per-sample genotype map incrementally.
    variant_index = []  # keep order list of variant keys
    data = {}  # variant_key -> {sample: allele_count}

    for p, sample in zip(sample_files, samples):
        logger.info(f"Parsing {p.name} (sample {sample})")
        vcf = VCF(str(p))
        for rec in vcf:
            # variant key: CHROM:POS:REF:ALT (first ALT only)
            alt = rec.ALT[0] if rec.ALT else "."
            key = f"{rec.CHROM}:{rec.POS}:{rec.REF}:{alt}"
            # read genotype for this sample
            try:
                # cyvcf2 rec.genotypes is a list of lists for multi-sample VCFs.
                # For single-sample VCF produced per-sample, rec.genotypes returns list of genotype tuples per sample.
                # Use rec.format or rec.genotypes; safer to use rec.genotypes and pick 1st sample
                gts = rec.genotypes  # list of genotype lists
                # For per-sample VCF, gts[0] corresponds to the sample
                gt0 = gts[0] if isinstance(gts, list) and len(gts) > 0 else None
                allele_count = genotype_to_allele_count(gt0)
            except Exception:
                allele_count = None

            if key not in data:
                data[key] = {}
                variant_index.append(key)
            data[key][sample] = missing_as if allele_count is None else allele_count

    # Build DataFrame: rows variants, columns samples
    df = pd.DataFrame(index=variant_index, columns=samples)
    for variant in variant_index:
        for sample in samples:
            df.at[variant, sample] = data.get(variant, {}).get(sample, missing_as)

    # Convert types
    df = df.fillna(missing_as).astype(int)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    logger.info(f"Wrote SNP matrix to {out_csv} (shape {df.shape})")


def main():
    parser = argparse.ArgumentParser(
        description="Build SNP matrix from per-sample VCFs."
    )
    parser.add_argument(
        "--config", required=False, help="Path to bioinfo_config.yaml (optional)."
    )
    parser.add_argument(
        "--vcf_dir",
        required=True,
        help="Directory containing per-sample VCFs (.vcf.gz or .vcf).",
    )
    parser.add_argument(
        "--out_csv",
        required=False,
        help="Output CSV path (if omitted, read from config).",
    )
    parser.add_argument(
        "--missing_as",
        type=int,
        default=None,
        help="Value to use for missing genotypes (overrides config).",
    )
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = load_config(args.config)

    out_csv = (
        args.out_csv
        or (cfg and cfg.get("snp_matrix", {}).get("output_csv"))
        or "data/processed/features/snp_matrix.csv"
    )
    missing_as = (
        args.missing_as
        if args.missing_as is not None
        else (cfg.get("snp_matrix", {}).get("missing_as") if cfg else 0)
    )

    build_matrix(args.vcf_dir, out_csv, missing_as=missing_as)


if __name__ == "__main__":
    main()
