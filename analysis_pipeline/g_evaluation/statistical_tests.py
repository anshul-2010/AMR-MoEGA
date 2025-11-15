"""
Run statistical tests (e.g., paired t-test, bootstrap) between methods' metrics.
"""
import argparse
from scipy import stats
import json
from utils.logger import get_logger

logger = get_logger("stat_tests")


def paired_t_test(list_a, list_b):
    return stats.ttest_rel(list_a, list_b)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="json list file")
    p.add_argument("--b", required=True)
    args = p.parse_args()
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    t = paired_t_test(a, b)
    logger.info(f"Paired t-test: statistic={t.statistic}, pvalue={t.pvalue}")
