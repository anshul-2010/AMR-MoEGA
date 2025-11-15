"""
Visualize PCA embeddings with matplotlib (and optionally seaborn in notebooks).
"""
import argparse
import pandas as pd
from utils.plot_utils import line_plot
from utils.logger import get_logger
import matplotlib.pyplot as plt

logger = get_logger("pca_vis")


def scatter_pca(embeddings_csv, labels_csv, out_png):
    emb = pd.read_csv(embeddings_csv, index_col=0)
    labels = pd.read_csv(labels_csv, index_col=0)
    x = emb.iloc[:, 0]
    y = emb.iloc[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=labels.values.ravel(), alpha=0.7)
    ax.set_title("PCA scatter")
    fig.savefig(out_png, bbox_inches="tight")
    logger.info(f"PCA plot saved to {out_png}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    scatter_pca(args.emb, args.labels, args.out)
