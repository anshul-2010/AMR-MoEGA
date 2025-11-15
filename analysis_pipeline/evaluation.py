"""
Driver for evaluation modules under 06_evaluation.
"""

from pipeline.g_evaluation.cross_validation import run_cross_validation
from pipeline.g_evaluation.comparative_results import generate_comparative_results
from pipeline.g_evaluation.statistical_tests import run_stats_tests


def run_evaluation(config):
    ev = config.get("evaluation", {})

    print("\n========== MODEL EVALUATION PIPELINE ==========\n")

    if ev.get("cross_validation", True):
        print("[1/3] Running cross-validation...")
        run_cross_validation(config)

    if ev.get("comparative_results", True):
        print("[2/3] Generating comparative metrics...")
        generate_comparative_results(config)

    if ev.get("stat_tests", True):
        print("[3/3] Running statistical significance tests...")
        run_stats_tests(config)

    print("\n Evaluation completed!\n")
