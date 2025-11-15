"""
Feature importance calculation and plotting, for tree models or permutation importance.
"""
import argparse
import joblib
import pandas as pd
from utils.logger import get_logger

logger = get_logger("feat_imp")


def save_feature_importance(model_path, feature_names_csv, out_csv):
    model = joblib.load(model_path)
    feature_names = pd.read_csv(feature_names_csv).columns.tolist()
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        logger.warning(
            "Model has no feature_importances_ attribute; consider permutation importance"
        )
        importances = [0] * len(feature_names)
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df.sort_values("importance", ascending=False, inplace=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote feature importance to {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    save_feature_importance(args.model, args.features, args.out)
