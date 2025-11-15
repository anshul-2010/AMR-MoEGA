"""
Wrapper for alignment (e.g., bwa mem -> samtools).
"""
import argparse
import subprocess
from utils.logger import get_logger

logger = get_logger("align_reads")


def align_bwa(reference, forward, reverse, out_bam, threads=4):
    # Example: bwa mem ref.fa R1.fastq R2.fastq | samtools view -bS - | samtools sort -o out.bam
    cmd = f"bwa mem -t {threads} {reference} {forward} {reverse} | samtools view -bS - | samtools sort -o {out_bam}"
    logger.info("Running alignment pipeline")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--forward", required=True)
    p.add_argument("--reverse", required=True)
    p.add_argument("--out_bam", required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    align_bwa(args.reference, args.forward, args.reverse, args.out_bam, args.threads)
