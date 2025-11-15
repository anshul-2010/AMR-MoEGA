"""
Build aggregated/functional features (e.g., counts of AMR genes, efflux pumps).
"""
import argparse
import pandas as pd
from utils.logger import get_logger
from pathlib import Path

logger = get_logger("functional_features")


def build_features(gene_pa_csv: str, snp_matrix_csv: str, out_csv: str):
    # TODO: combine gene-level and snp-level features; add domain-specific aggregations
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame()
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved functional features to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gene_pa", required=True)
    p.add_argument("--snp_matrix", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()
    build_features(args.gene_pa, args.snp_matrix, args.out_csv)
