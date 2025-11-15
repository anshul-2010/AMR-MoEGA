from typing import Dict, Any, Tuple
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import warnings

# Try to import XGBoost and LightGBM; if missing, fallback to sklearn wrappers
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

warnings.filterwarnings("ignore")


def build_xgb(params: Dict[str, Any]) -> BaseEstimator:
    if XGBClassifier is None:
        # fallback to RandomForest if XGBoost not available
        print(
            "Warning: xgboost not installed; using RandomForest as fallback for XGB expert"
        )
        return RandomForestClassifier(
            n_estimators=max(50, params.get("n_estimators", 100))
        )
    return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)


def build_lgbm(params: Dict[str, Any]) -> BaseEstimator:
    if LGBMClassifier is None:
        print(
            "Warning: lightgbm not installed; using RandomForest as fallback for LGBM expert"
        )
        return RandomForestClassifier(
            n_estimators=max(50, params.get("n_estimators", 100))
        )
    return LGBMClassifier(**params)


def build_rf(params: Dict[str, Any]) -> BaseEstimator:
    # convert max_features if fractional
    mf = params.get("max_features", None)
    if isinstance(mf, float) and 0 < mf <= 1:
        params["max_features"] = mf
    return RandomForestClassifier(**params)


def tune_and_train(
    model_builder, param_distribution: Dict, X, y, cv=3, n_iter=10, random_state=0
):
    """
    Randomized search with StratifiedKFold and then fit final model on full train.
    param_distribution: dict where values are tuples (low, high) or lists
    """
    # Simple wrapper: construct estimator with default params, then RandomizedSearch
    estimator = model_builder({})
    # Convert param_distribution into sklearn style distributions
    # Here, implement simple conversion: for (low, high) tuples treat as uniform ints/floats
    from scipy.stats import randint, uniform

    sk_params = {}
    for k, v in param_distribution.items():
        if isinstance(v, tuple) and len(v) == 2:
            low, high = v
            if isinstance(low, int) and isinstance(high, int):
                sk_params[k] = randint(low, high + 1)
            else:
                sk_params[k] = uniform(low, high - low)
        elif isinstance(v, list):
            sk_params[k] = v
        else:
            sk_params[k] = [v]

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rnd = RandomizedSearchCV(
        estimator,
        sk_params,
        n_iter=n_iter,
        cv=cv_split,
        scoring="roc_auc",
        n_jobs=1,
        random_state=random_state,
    )
    rnd.fit(X, y)
    best = rnd.best_estimator_
    return best, rnd.best_score_


def train_experts_for_chromosome(
    chrom: Dict, X_train, y_train, X_val=None, y_val=None, tuning_iters=8
):
    """
    Train XGB, LGBM, RF experts on selected features from the chromosome.
    Returns trained models and validation AUCs (if X_val provided).
    """
    # Extract features by mask
    mask = np.array(chrom["feature_mask"], dtype=bool)
    if mask.sum() == 0:
        # fallback: use first feature to avoid crash
        mask[0] = True

    Xtr = X_train[:, mask]
    Xval = X_val[:, mask] if X_val is not None else None

    # Train XGBoost
    xgb_params = chrom.get("xgb_params", {})
    xgb = build_xgb(xgb_params)
    # Optionally tune: here simply set params and fit for speed
    xgb.set_params(**{k: v for k, v in xgb_params.items() if k in xgb.get_params()})
    xgb.fit(Xtr, y_train)

    # Train LightGBM
    lgbm_params = chrom.get("lgbm_params", {})
    lgbm = build_lgbm(lgbm_params)
    lgbm.set_params(**{k: v for k, v in lgbm_params.items() if k in lgbm.get_params()})
    lgbm.fit(Xtr, y_train)

    # Train RandomForest
    rf_params = chrom.get("rf_params", {})
    rf = build_rf(rf_params)
    rf.set_params(**{k: v for k, v in rf_params.items() if k in rf.get_params()})
    rf.fit(Xtr, y_train)

    # Compute val predictions if available
    val_scores = {}
    if Xval is not None and y_val is not None:
        preds_x = predict_proba_safe(xgb, Xval)
        preds_l = predict_proba_safe(lgbm, Xval)
        preds_r = predict_proba_safe(rf, Xval)
        # compute aucs
        try:
            from sklearn.metrics import roc_auc_score

            val_scores["xgb_auc"] = roc_auc_score(y_val, preds_x)
            val_scores["lgbm_auc"] = roc_auc_score(y_val, preds_l)
            val_scores["rf_auc"] = roc_auc_score(y_val, preds_r)
        except Exception:
            val_scores = {}
    return {"xgb": xgb, "lgbm": lgbm, "rf": rf}, val_scores


def predict_proba_safe(model, X):
    """
    Return probability for positive class; fallback to predict if necessary.
    """
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # handle binary vs multiclass
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
            else:
                # if only one column returned, treat as pos probability
                return p.ravel()
        else:
            preds = model.predict(X)
            return preds
    except Exception:
        return np.zeros(X.shape[0])
