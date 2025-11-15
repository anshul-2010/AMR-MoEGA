"""
Construct gene presence/absence features from assemblies/annotations.
"""
import argparse
import pandas as pd
from utils.logger import get_logger
from pathlib import Path

logger = get_logger("gene_presence")


def build_gene_pa(annotations_dir: str, out_csv: str):
    # TODO: parse GFF/Prokka outputs and build presence/absence table
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame()
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved gene presence/absence to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()
    build_gene_pa(args.annotations_dir, args.out_csv)
