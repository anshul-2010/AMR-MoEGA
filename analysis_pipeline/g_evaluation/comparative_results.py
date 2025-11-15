"""
Aggregate experimental outputs and create comparatives (CSV/plots).
"""
import argparse
import json
import pandas as pd
from utils.logger import get_logger

logger = get_logger("comparative")


def aggregate_results(json_paths, out_csv):
    rows = []
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        rows.append(data)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote comparative results to {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    aggregate_results(args.inputs, args.out)
