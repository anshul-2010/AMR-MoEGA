"""
Skeleton for variant calling (e.g., bcftools/mpileup + bcftools call) and refinement.
This is pseudo-code-level but shows expected inputs/outputs.
"""
import argparse
import subprocess
from utils.logger import get_logger
from pathlib import Path

logger = get_logger("variant_calling")


def call_variants(reference: str, bam: str, out_vcf: str, threads: int = 4):
    Path(out_vcf).parent.mkdir(parents=True, exist_ok=True)
    # Example pipeline using bcftools:
    cmd = (
        f"bcftools mpileup -f {reference} {bam} | bcftools call -mv -Oz -o {out_vcf}.gz"
    )
    logger.info("Calling variants: " + cmd)
    subprocess.run(cmd, shell=True, check=True)
    subprocess.run(f"bcftools index {out_vcf}.gz", shell=True, check=True)
    # optional: normalize & refine
    logger.info(f"Variants written to {out_vcf}.gz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--bam", required=True)
    p.add_argument("--out_vcf", required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    call_variants(args.reference, args.bam, args.out_vcf, args.threads)
