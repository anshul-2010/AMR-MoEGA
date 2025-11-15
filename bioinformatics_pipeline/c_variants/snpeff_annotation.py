"""
SnpEff annotation wrapper.
"""
import argparse
import subprocess
from utils.logger import get_logger

logger = get_logger("snpeff")


def annotate(vcf_in: str, vcf_out: str, snpeff_db: str):
    # snpeff usage: java -Xmx4g -jar snpEff.jar build ... ; java -jar snpEff.jar ann ref input.vcf > output.vcf
    cmd = f"snpEff -v {snpeff_db} {vcf_in} > {vcf_out}"
    logger.info("Running SnpEff annotation")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vcf_in", required=True)
    p.add_argument("--vcf_out", required=True)
    p.add_argument("--db", required=True)
    args = p.parse_args()
    annotate(args.vcf_in, args.vcf_out, args.db)
