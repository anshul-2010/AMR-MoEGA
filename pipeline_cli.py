#!/usr/bin/env python3
"""
Unified CLI for the AMR Evolution Prediction Pipeline
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Import pipeline modules
from bioinformatics_pipeline.bioinfo import run_bioinfo_pipeline
from ml_pipeline.feature_engineering import run_feature_engineering
from ml_pipeline.modeling import run_model_training
from analysis_pipeline.evaluation import run_evaluation


# Helper utilities
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# CLI Commands
def run_bioinfo(args):
    config = load_config(args.config)
    print("[INFO] Loaded configuration.")
    run_bioinfo_pipeline(config)
    print("[DONE] Bioinformatics pipeline completed.")


def run_features(args):
    config = load_config(args.config)
    run_feature_engineering(config)
    print("[DONE] Feature engineering completed.")


def run_model(args):
    config = load_config(args.config)
    run_model_training(config)
    print("[DONE] Model training completed.")


def run_eval(args):
    config = load_config(args.config)
    run_evaluation(config)
    print("[DONE] Evaluation completed.")


# Main parser
def build_parser():
    parser = argparse.ArgumentParser(
        description="AMR Evolution Prediction Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Bioinformatics
    bioinfo = subparsers.add_parser("bioinfo", help="Run the bioinformatics pipeline.")
    bioinfo.add_argument(
        "-c", "--config", required=True, help="Path to bioinfo_config.yaml"
    )
    bioinfo.set_defaults(func=run_bioinfo)

    # Feature Engineering
    feat = subparsers.add_parser("features", help="Run feature engineering.")
    feat.add_argument("-c", "--config", required=True, help="Path to main config.yaml")
    feat.set_defaults(func=run_features)

    # Modeling
    model = subparsers.add_parser("model", help="Run ML/MoE-GA training.")
    model.add_argument("-c", "--config", required=True, help="Path to main config.yaml")
    model.set_defaults(func=run_model)

    # Evaluation
    evalp = subparsers.add_parser("eval", help="Run evaluation.")
    evalp.add_argument("-c", "--config", required=True, help="Path to main config.yaml")
    evalp.set_defaults(func=run_eval)

    return parser


# Entry point
def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
