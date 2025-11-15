"""
Driver for all modeling algorithms under 04_modeling.
"""

from ml_pipeline.e_modeling.static_ml_training import train_static_models
from ml_pipeline.e_modeling.ga_static_fitness import run_ga_baseline
from ml_pipeline.e_modeling.moe_ga_engine import train_moe_ga


def run_model_training(config):
    m = config.get("modeling", {})

    print("\n========== MODEL TRAINING PIPELINE ==========\n")

    if m.get("train_static_ml", True):
        print("[1/3] Training baseline static models...")
        train_static_models(config)

    if m.get("train_ga_baseline", True):
        print("[2/3] Running GA with static fitness...")
        run_ga_baseline(config)

    if m.get("train_moega", True):
        print("[3/3] Training MoE-GA...")
        train_moe_ga(config)

    print("\n Modeling completed!\n")
