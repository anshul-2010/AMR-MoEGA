# pipeline/04_modeling/moega/ga_engine.py
#!/usr/bin/env python3
"""
GA driver for AMR-MoEGA.

Usage:
    python ga_engine.py --features data/processed/features/snp_matrix.csv --labels data/processed/labels/labels.csv \
       --pop 20 --gens 10 --out experiments/moega_run
"""

import argparse
import os
import json
import numpy as np
import joblib
from pathlib import Path
from .search_space import sample_chromosome
from .genetic_operators import tournament_selection, one_point_crossover, mutate
from .fitness import evaluate_chromosome
from .trainer import load_features_and_labels, train_val_split
from .chromosome import chromosome_hash
from .logging_utils import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features",
        required=True,
        help="CSV of features (samples x features). First column may be index.",
    )
    p.add_argument("--labels", required=True, help="CSV of labels (samples x 1).")
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--mut_mask_rate", type=float, default=0.01)
    p.add_argument("--mut_param_rate", type=float, default=0.1)
    p.add_argument("--tourney_k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="experiments/moega_run")
    p.add_argument("--test_size", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("moega-ga", logfile=str(out_dir / "moega.log"))
    logger.info("Loading data...")
    X, y = load_features_and_labels(args.features, args.labels)
    X_train, X_val, y_train, y_val = train_val_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=True
    )

    n_features = X.shape[1]
    logger.info(f"Dataset: n_samples={X.shape[0]} n_features={n_features}")

    # Initialize population
    logger.info("Initializing population...")
    population = [sample_chromosome(n_features) for _ in range(args.pop)]

    fitnesses = []
    cache = {}
    # Evaluate initial population
    logger.info("Evaluating initial population...")
    for i, chrom in enumerate(population):
        res = evaluate_chromosome(chrom, X_train, y_train, X_val, y_val, cache=True)
        fitness = float(res["auc"])
        fitnesses.append(fitness)
        logger.info(f"[init] idx={i} auc={fitness:.4f}")

    # GA loop
    for gen in range(args.gens):
        logger.info(f"=== Generation {gen+1}/{args.gens} ===")
        new_pop = []

        # Elitism: carry best 'elitism' individuals
        elite_idxs = sorted(
            range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True
        )[: args.elitism]
        for idx in elite_idxs:
            new_pop.append(population[idx])

        # create children until population size filled
        while len(new_pop) < args.pop:
            parent_a = tournament_selection(population, fitnesses, k=args.tourney_k)
            parent_b = tournament_selection(population, fitnesses, k=args.tourney_k)
            child_a, child_b = one_point_crossover(parent_a, parent_b)
            child_a = mutate(
                child_a,
                mut_rate_mask=args.mut_mask_rate,
                mut_rate_param=args.mut_param_rate,
            )
            child_b = mutate(
                child_b,
                mut_rate_mask=args.mut_mask_rate,
                mut_rate_param=args.mut_param_rate,
            )
            new_pop.extend([child_a, child_b])

        population = new_pop[: args.pop]

        # Evaluate population
        fitnesses = []
        for i, chrom in enumerate(population):
            logger.info(f"Evaluating individual {i}...")
            res = evaluate_chromosome(chrom, X_train, y_train, X_val, y_val, cache=True)
            fitness = float(res["auc"])
            fitnesses.append(fitness)
            logger.info(f"[gen {gen+1}] idx={i} auc={fitness:.4f}")

        # Log best
        best_idx = int(np.argmax(fitnesses))
        best_chrom = population[best_idx]
        best_fit = fitnesses[best_idx]
        logger.info(f"Generation {gen+1} best AUC={best_fit:.4f} (idx={best_idx})")
        # Save best so far
        joblib.dump(
            {"chrom": best_chrom, "fitness": best_fit},
            out_dir / f"best_gen{gen+1}.joblib",
        )

    # final best
    final_best_idx = int(np.argmax(fitnesses))
    final_best = population[final_best_idx]
    final_best_fit = fitnesses[final_best_idx]
    logger.info(f"Final best AUC={final_best_fit:.4f}")
    joblib.dump(
        {"chrom": final_best, "fitness": final_best_fit},
        out_dir / "best_overall.joblib",
    )

    # save population summary
    summary = [
        {"hash": chromosome_hash(ch), "fitness": f}
        for ch, f in zip(population, fitnesses)
    ]
    with open(out_dir / "population_summary.json", "w") as fh:
        fh.write(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
