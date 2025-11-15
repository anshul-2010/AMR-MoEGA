from typing import Dict, Any
import numpy as np
import random

# Define defaults and sampling functions for each expert's hyperparameters.
# These are simple spaces suitable for RandomizedSearch.
XGB_SPACE = {
    "n_estimators": (50, 500),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 1.0),
}

LGBM_SPACE = {
    "n_estimators": (50, 500),
    "num_leaves": (15, 255),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 1.0),
}

RF_SPACE = {
    "n_estimators": (50, 1000),
    "max_depth": (5, 50),
    "max_features": (0.1, 1.0),  # fraction
}


def sample_uniform_int(low: int, high: int) -> int:
    return int(np.random.randint(low, high + 1))


def sample_uniform_float(low: float, high: float) -> float:
    return float(np.random.uniform(low, high))


def sample_xgb_params() -> Dict[str, Any]:
    return {
        "n_estimators": sample_uniform_int(*XGB_SPACE["n_estimators"]),
        "max_depth": sample_uniform_int(*XGB_SPACE["max_depth"]),
        "learning_rate": round(sample_uniform_float(*XGB_SPACE["learning_rate"]), 4),
        "subsample": round(sample_uniform_float(*XGB_SPACE["subsample"]), 3),
        "colsample_bytree": round(
            sample_uniform_float(*XGB_SPACE["colsample_bytree"]), 3
        ),
    }


def sample_lgbm_params() -> Dict[str, Any]:
    return {
        "n_estimators": sample_uniform_int(*LGBM_SPACE["n_estimators"]),
        "num_leaves": sample_uniform_int(*LGBM_SPACE["num_leaves"]),
        "learning_rate": round(sample_uniform_float(*LGBM_SPACE["learning_rate"]), 4),
        "subsample": round(sample_uniform_float(*LGBM_SPACE["subsample"]), 3),
        "colsample_bytree": round(
            sample_uniform_float(*LGBM_SPACE["colsample_bytree"]), 3
        ),
    }


def sample_rf_params() -> Dict[str, Any]:
    return {
        "n_estimators": sample_uniform_int(*RF_SPACE["n_estimators"]),
        "max_depth": sample_uniform_int(*RF_SPACE["max_depth"]),
        "max_features": round(sample_uniform_float(*RF_SPACE["max_features"]), 3),
    }


def sample_chromosome(n_features: int, include_mask_prob: float = 0.2) -> Dict:
    """
    Create a random chromosome:
    - feature_mask: list of 0/1 length n_features (sparse by default)
    - expert_params: dict for xgb, lgbm, rf
    """
    mask = [1 if random.random() < include_mask_prob else 0 for _ in range(n_features)]
    return {
        "feature_mask": mask,
        "xgb_params": sample_xgb_params(),
        "lgbm_params": sample_lgbm_params(),
        "rf_params": sample_rf_params(),
    }
