"""
Compute coverage stats. Placeholder: integrate mosdepth/bedtools or samtools depth.
"""
import argparse
import subprocess
from utils.logger import get_logger

logger = get_logger("coverage_stats")


def compute_coverage(bam, out_depth):
    cmd = f"samtools depth -a {bam} > {out_depth}"
    logger.info("Computing depth per base")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bam", required=True)
    p.add_argument("--out_depth", required=True)
    args = p.parse_args()
    compute_coverage(args.bam, args.out_depth)
