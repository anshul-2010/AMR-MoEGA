"""
Collect alignment QC metrics (samtools flagstat, coverage, etc.).
"""
import argparse
import subprocess
from utils.logger import get_logger

logger = get_logger("post_alignment_qc")


def flagstat(bam, out_txt):
    cmd = ["samtools", "flagstat", bam]
    logger.info("Running samtools flagstat")
    with open(out_txt, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bam", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    flagstat(args.bam, args.out)
