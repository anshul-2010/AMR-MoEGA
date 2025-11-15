"""
Variant filtering skeleton (e.g., quality, depth).
"""
import argparse
import subprocess
from utils.logger import get_logger

logger = get_logger("filter_variants")


def filter_vcf(in_vcf_gz: str, out_vcf: str, min_dp: int = 10, min_qual: int = 20):
    # Example using bcftools filter
    cmd = f"bcftools view {in_vcf_gz} | bcftools filter -e 'DP<{min_dp} || QUAL<{min_qual}' -o {out_vcf}"
    logger.info("Filtering variants")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in_vcf_gz", required=True)
    p.add_argument("--out_vcf", required=True)
    p.add_argument("--min_dp", type=int, default=10)
    p.add_argument("--min_qual", type=int, default=20)
    args = p.parse_args()
    filter_vcf(args.in_vcf_gz, args.out_vcf, args.min_dp, args.min_qual)
