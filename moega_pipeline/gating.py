import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_gating_network(hidden_dim: int = 64, random_state: int = 0):
    # gating network: takes raw features (optionally reduced) or expert outputs
    # For simplicity, gating takes original (selected) features and outputs expert responsibilities
    scaler = StandardScaler()
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,), max_iter=300, random_state=random_state
    )
    model = Pipeline([("scaler", scaler), ("mlp", mlp)])
    return model


def train_gating_network(gating_model, X, expert_probs, responsibilities=None):
    """
    Option A: learn gating from inputs -> chosen expert (argmax of expert_probs) (hard labels)
    Option B: learn gating to predict soft weights (we convert to multi-output regression) — here we use classification to predict argmax.
    """
    # derive soft targets: argmax of expert predictions per sample
    # expert_probs: array shape (n_samples, n_experts) with expert class probs
    targets = np.argmax(expert_probs, axis=1)
    gating_model.fit(X, targets)
    # return gating probs per expert
    if hasattr(gating_model, "predict_proba"):
        return gating_model.predict_proba(X)
    else:
        # fallback: create one-hot predictions
        preds = gating_model.predict(X)
        proba = np.zeros((X.shape[0], np.max(preds) + 1))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba


def gating_weighted_prediction(expert_probs, gating_weights):
    """
    expert_probs: list or array of shape (n_samples, n_experts) predicted positive-class probs from each expert.
    gating_weights: array (n_samples, n_experts) weights (rows sum to 1)
    returns weighted ensemble probability per sample.
    """
    import numpy as np

    expert_probs = np.array(
        expert_probs
    )  # (n_experts, n_samples) or (n_samples, n_experts)
    if expert_probs.ndim == 2 and expert_probs.shape[1] == gating_weights.shape[1]:
        # assume shape (n_samples, n_experts)
        weighted = (expert_probs * gating_weights).sum(axis=1)
    else:
        # try transpose
        weighted = (expert_probs.T * gating_weights).sum(axis=1)
    return weighted
