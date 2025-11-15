#!/usr/bin/env bash
set -euo pipefail
# Example: draw PCA and fitness plots
python pipeline/05_post_analysis/pca_visualization.py --emb data/processed/PCA_embeddings.csv --labels data/processed/labels.csv --out results/pca.png
python pipeline/05_post_analysis/fitness_trajectory_plots.py --inputs experiments/exp1/fitness.csv experiments/exp3/fitness.csv --labels baseline MoEGA --out results/fitness.png
echo "Plots generated"