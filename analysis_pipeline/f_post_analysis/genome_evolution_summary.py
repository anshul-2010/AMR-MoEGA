"""
Summarize genome-level changes across generations / time points.
This is intentionally high-level; fill with domain-specific analyses.
"""
import argparse
import pandas as pd
from utils.logger import get_logger

logger = get_logger("genome_summary")


def summarize_variants(variant_tables, out_csv):
    # variant_tables: list of CSVs per timepoint/gen
    # TODO: compute allele freq changes, new mutations, dN/dS, etc.
    res = []
    for p in variant_tables:
        df = pd.read_csv(p)
        res.append({"file": p, "n_variants": len(df)})
    pd.DataFrame(res).to_csv(out_csv, index=False)
    logger.info(f"Wrote genome summary to {out_csv}")
