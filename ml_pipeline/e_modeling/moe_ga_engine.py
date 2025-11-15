"""
Main skeleton for MoE + GA integration.
This has:
- population of 'architectures' (e.g., which experts are active)
- evaluation of architectures by training experts and gating network on a task
- GA operators to evolve architectures
"""
import argparse
import random
import numpy as np
from utils.logger import get_logger
from utils.genetic_operators import (
    tournament_selection,
    one_point_crossover,
    uniform_mutation,
)

logger = get_logger("moe_ga")


def build_expert(expert_config):
    # TODO: build and return an expert model (e.g., sklearn or torch model)
    return None


def build_gating_network(n_experts):
    # TODO: build gating network that takes input features and outputs weights per expert
    return None


def evaluate_architecture(arch, X_train, y_train, X_val, y_val):
    # arch: representation of experts/config
    # TODO: train experts and gating, return validation metric (e.g., f1)
    return random.random()


def run_moe_ga(X_train, y_train, X_val, y_val, pop_size=10, gens=30):
    # Example chromosome: list of booleans indicating active experts + hyperparams
    population = [[random.randint(0, 1) for _ in range(5)] for _ in range(pop_size)]
    fitnesses = [
        evaluate_architecture(p, X_train, y_train, X_val, y_val) for p in population
    ]
    for g in range(gens):
        new_pop = []
        while len(new_pop) < pop_size:
            a = tournament_selection(population, fitnesses)
            b = tournament_selection(population, fitnesses)
            c1, c2 = one_point_crossover(a, b)
            c1 = uniform_mutation(c1, 0.05)
            c2 = uniform_mutation(c2, 0.05)
            new_pop.extend([c1, c2])
        population = new_pop[:pop_size]
        fitnesses = [
            evaluate_architecture(p, X_train, y_train, X_val, y_val) for p in population
        ]
        logger.info(f"MoE-GA Gen {g}: best {max(fitnesses)}")
    best_idx = int(np.argmax(fitnesses))
    return population[best_idx], fitnesses[best_idx]


if __name__ == "__main__":
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    args = p.parse_args()
    X = pd.read_csv(args.features).values
    y = pd.read_csv(args.labels).iloc[:, 0].values
    # quick split
    from sklearn.model_selection import train_test_split

    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    arch, fit = run_moe_ga(Xtr, ytr, Xval, yval)
    logger.info(f"Best arch {arch} fit {fit}")
