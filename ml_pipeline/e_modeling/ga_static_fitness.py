"""
A GA baseline that optimizes a static fitness function (e.g., classifier accuracy over features subset).
This provides the fitness wrapper and GA loop skeleton.
"""
import argparse
import random
from utils.logger import get_logger
from utils.genetic_operators import (
    tournament_selection,
    one_point_crossover,
    uniform_mutation,
)
import numpy as np

logger = get_logger("ga_static")


def evaluate_solution(solution, X, y):
    # solution: binary mask for feature selection
    # TODO: use sklearn classifier to evaluate selected features
    return random.random()  # placeholder


def run_ga(X, y, pop_size=20, gens=50, mut_rate=0.01):
    n_features = X.shape[1]
    population = [
        [random.randint(0, 1) for _ in range(n_features)] for _ in range(pop_size)
    ]
    fitnesses = [evaluate_solution(p, X, y) for p in population]
    for g in range(gens):
        new_pop = []
        while len(new_pop) < pop_size:
            parent_a = tournament_selection(population, fitnesses)
            parent_b = tournament_selection(population, fitnesses)
            c1, c2 = one_point_crossover(parent_a, parent_b)
            c1 = uniform_mutation(c1, mut_rate)
            c2 = uniform_mutation(c2, mut_rate)
            new_pop.extend([c1, c2])
        population = new_pop[:pop_size]
        fitnesses = [evaluate_solution(p, X, y) for p in population]
        logger.info(f"Gen {g}: best fitness {max(fitnesses)}")
    best_idx = int(np.argmax(fitnesses))
    return population[best_idx], fitnesses[best_idx]


if __name__ == "__main__":
    import pandas as pd, numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    args = p.parse_args()
    X = pd.read_csv(args.features).values
    y = pd.read_csv(args.labels).iloc[:, 0].values
    sol, fit = run_ga(X, y)
    logger.info(f"Best fitness {fit}")
