"""
PCA + dimensionality reduction pipeline for features.
"""
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from utils.logger import get_logger
from pathlib import Path

logger = get_logger("pca_reduction")


def pca_reduce(features_csv: str, out_csv: str, n_components: int = 50):
    df = pd.read_csv(features_csv)
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(df.values)
    emb_df = pd.DataFrame(
        embeddings,
        index=df.index,
        columns=[f"pc{i+1}" for i in range(embeddings.shape[1])],
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(out_csv)
    logger.info(f"PCA embeddings saved to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--n_components", type=int, default=50)
    args = p.parse_args()
    pca_reduce(args.features_csv, args.out_csv, args.n_components)
