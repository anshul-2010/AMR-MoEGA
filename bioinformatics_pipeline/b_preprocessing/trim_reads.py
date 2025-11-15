"""
Wrapper for trimming (fastp) with CLI calls.
"""
import argparse
import subprocess
from utils.logger import get_logger

logger = get_logger("trim_reads")


def trim_pair(forward, reverse, out_forward, out_reverse, threads=4):
    cmd = [
        "fastp",
        "-i",
        forward,
        "-I",
        reverse,
        "-o",
        out_forward,
        "-O",
        out_reverse,
        "-w",
        str(threads),
        "--detect_adapter_for_pe",
    ]
    logger.info("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forward", required=True)
    p.add_argument("--reverse", required=True)
    p.add_argument("--out_forward", required=True)
    p.add_argument("--out_reverse", required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    trim_pair(
        args.forward, args.reverse, args.out_forward, args.out_reverse, args.threads
    )
