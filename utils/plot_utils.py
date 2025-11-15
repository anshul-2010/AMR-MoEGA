"""
Plot helpers. Keep styling minimal; notebooks can override.
"""
import matplotlib.pyplot as plt
import os
from typing import Optional


def savefig(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)


def line_plot(
    x, ys, labels, title=None, xlabel=None, ylabel=None, out_path: Optional[str] = None
):
    fig, ax = plt.subplots()
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.legend()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if out_path:
        savefig(fig, out_path)
    return fig
