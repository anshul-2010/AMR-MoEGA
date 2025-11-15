#!/usr/bin/env python3
"""
Reads a SNP matrix CSV (samples x variants or variants x samples),
applies QC filters (missingness, minor allele frequency, variance),
removes monomorphic sites, and writes a cleaned SNP matrix.

Assumptions:
- Input CSV: rows are variants (or positions) and columns are sample IDs, OR
  input CSV: rows are samples and columns are variants. The script will detect orientation.
- Genotypes are encoded as 0/1/2 or 0/1 (allele counts).
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger("preprocess_variants")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


def infer_orientation(df: pd.DataFrame) -> str:
    # Heuristic: if number of rows >> number of columns, likely variants x samples
    if df.shape[0] > df.shape[1]:
        return "variants_rows"
    else:
        return "samples_rows"


def compute_maf(series: pd.Series):
    # series contains genotypes (0,1,2) with NaN allowed
    vals = series.dropna().astype(float)
    if vals.empty:
        return 0.0
    allele_sum = vals.sum()  # allele counts under 0/1/2 encoding
    n_alleles = 2 * len(
        vals
    )  # diploid assumption; for bacteria may be 1 (haploid) — adjust as needed
    # Note: if your genotypes are 0/1 only and haploid, you might want to compute freq differently
    maf = min(allele_sum / n_alleles, 1 - allele_sum / n_alleles)
    return float(maf)


def main(args):
    in_csv = Path(args.input)
    out_csv = Path(args.output)
    assert in_csv.exists(), f"Input file not found: {in_csv}"

    logger.info(f"Loading SNP matrix from {in_csv}")
    df = pd.read_csv(in_csv, index_col=0) if args.index_col else pd.read_csv(in_csv)

    orientation = infer_orientation(df)
    logger.info(
        f"Inferred orientation: {orientation} (rows={df.shape[0]}, cols={df.shape[1]})"
    )

    if orientation == "samples_rows":
        # convert to variants x samples for consistent processing
        df = df.transpose()

    # At this point rows are variants, columns are samples
    n_variants, n_samples = df.shape
    logger.info(f"Variants: {n_variants}, Samples: {n_samples}")

    # Convert common missing encodings to NaN
    df = df.replace([".", "NA", "nan", ""], np.nan)

    # Ensure dtype numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # 1) Filter by missingness
    missing_rate = df.isna().mean(axis=1)  # per variant
    keep_missing = missing_rate <= args.max_missing
    logger.info(
        f"Filtering variants by missingness: <= {args.max_missing} (kept {keep_missing.sum()} / {len(keep_missing)})"
    )
    df = df.loc[keep_missing]

    # 2) Remove monomorphic sites (variance == 0 or single value)
    var_series = df.var(axis=1, ddof=0)  # population variance
    keep_variance = var_series > 0
    logger.info(
        f"Removing monomorphic sites (variance == 0). Kept {keep_variance.sum()} / {len(keep_variance)}"
    )
    df = df.loc[keep_variance]

    # 3) Minor allele frequency filter
    # Warning: For bacteria (haploid) the MAF computation differs (use allele freq rather than allele count).
    maf_vals = df.apply(lambda row: compute_maf(row), axis=1)
    keep_maf = maf_vals >= args.min_maf
    logger.info(
        f"Filtering by MAF >= {args.min_maf}. Kept {keep_maf.sum()} / {len(keep_maf)}"
    )
    df = df.loc[keep_maf]

    # 4) Variance threshold (optional)
    if args.min_variance > 0:
        var_series = df.var(axis=1, ddof=0)
        keep_var = var_series >= args.min_variance
        logger.info(
            f"Filtering by variance >= {args.min_variance}. Kept {keep_var.sum()} / {len(keep_var)}"
        )
        df = df.loc[keep_var]

    # Optionally transpose back so output orientation matches input
    if orientation == "samples_rows" and not args.force_variants_rows:
        df_out = df.transpose()
    else:
        df_out = df

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv)
    logger.info(f"Wrote cleaned SNP matrix to {out_csv} (shape {df_out.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess SNP matrix: missingness, MAF, variance, monomorphic removal."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input SNP matrix CSV (with index col for variant IDs).",
    )
    parser.add_argument(
        "--output", required=True, help="Output cleaned SNP matrix CSV."
    )
    parser.add_argument(
        "--index-col",
        action="store_true",
        help="If set, the first column is an index/variant ID and should be used as index.",
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=0.2,
        help="Max allowed missing rate per variant (0-1). Default 0.2",
    )
    parser.add_argument(
        "--min-maf",
        type=float,
        default=0.01,
        help="Minimum minor allele frequency (0-0.5). Default 0.01",
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=0.0,
        help="Minimum variance threshold (default 0.0).",
    )
    parser.add_argument(
        "--force-variants-rows",
        action="store_true",
        help="Force output to be variants x samples (do not transpose back).",
    )
    args = parser.parse_args()
    main(args)
