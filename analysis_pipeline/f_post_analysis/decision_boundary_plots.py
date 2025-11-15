"""
Decision boundary plotting (2D projections).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger("decision_boundary")


def decision_boundary_2d(model, X, y, out_path):
    # expects X to have 2 dims (or project before calling)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    plt.savefig(out_path, bbox_inches="tight")
    logger.info(f"Saved decision boundary to {out_path}")
