import numpy as np
from sklearn.metrics import roc_auc_score
from .experts import train_experts_for_chromosome, predict_proba_safe
from .gating import (
    build_gating_network,
    train_gating_network,
    gating_weighted_prediction,
)
from .chromosome import chromosome_hash
import joblib
import os
from pathlib import Path

# Simple cache directory
CACHE_DIR = Path("experiments/moega_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_chromosome(
    chrom: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cache=True,
):
    """
    Train experts according to chrom, train gating, compute validation AUC.
    Returns fitness (AUC) and auxiliary info.
    """
    key = chromosome_hash(chrom)
    cache_file = CACHE_DIR / f"{key}.joblib"
    if cache and cache_file.exists():
        return joblib.load(cache_file)

    # Train experts on selected features
    models, val_scores = train_experts_for_chromosome(
        chrom, X_train, y_train, X_val, y_val
    )

    # Get expert predictions on validation set
    mask = np.array(chrom["feature_mask"], dtype=bool)
    if mask.sum() == 0:
        mask[0] = True
    Xval_sel = X_val[:, mask]

    p_xgb = predict_proba_safe(models["xgb"], Xval_sel)
    p_lgb = predict_proba_safe(models["lgbm"], Xval_sel)
    p_rf = predict_proba_safe(models["rf"], Xval_sel)

    # Stack expert probs -> (n_samples, n_experts)
    expert_probs = np.vstack([p_xgb, p_lgb, p_rf]).T

    # Build and train gating network on training set (use expert outputs on train)
    # For gating training we need expert outputs on train set
    Xtrain_sel = X_train[:, mask]
    p_xgb_tr = predict_proba_safe(models["xgb"], Xtrain_sel)
    p_lgb_tr = predict_proba_safe(models["lgbm"], Xtrain_sel)
    p_rf_tr = predict_proba_safe(models["rf"], Xtrain_sel)
    expert_probs_train = np.vstack([p_xgb_tr, p_lgb_tr, p_rf_tr]).T

    gating = build_gating_network(hidden_dim=64)
    gating_weights_train = train_gating_network(gating, Xtrain_sel, expert_probs_train)
    # On validation, compute gating weights
    try:
        gating_weights_val = gating.predict_proba(Xval_sel)
    except Exception:
        # fallback: use training gating weights averaged
        gating_weights_val = np.tile(
            np.mean(gating_weights_train, axis=0, keepdims=True), (Xval_sel.shape[0], 1)
        )

    # Weighted ensemble
    ens_probs = gating_weighted_prediction(expert_probs, gating_weights_val)
    try:
        auc = float(roc_auc_score(y_val, ens_probs))
    except Exception:
        auc = 0.0

    result = {
        "auc": auc,
        "models": models,
        "gating": gating,
        "expert_val_scores": val_scores,
    }

    if cache:
        joblib.dump(result, cache_file)

    return result
