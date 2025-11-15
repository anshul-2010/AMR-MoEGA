"""
Plot GA/MoE fitness trajectories saved as lists/CSVs.
"""
import argparse
import pandas as pd
from utils.plot_utils import line_plot
from utils.logger import get_logger

logger = get_logger("fitness_plots")


def plot_fitness(csv_paths, labels, out_png):
    # assume csv has column 'gen' and 'fitness' or simply fitness per row
    ys = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "fitness" in df.columns:
            ys.append(df["fitness"].values)
        else:
            ys.append(df.iloc[:, 0].values)
    x = range(len(ys[0]))
    fig = line_plot(
        x,
        ys,
        labels,
        title="Fitness trajectories",
        xlabel="Generation",
        ylabel="Fitness",
        out_path=out_png,
    )
    logger.info(f"Saved fitness plot to {out_png}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    plot_fitness(args.inputs, args.labels, args.out)
