"""
Basic GA operators used by MoE-GA and GA baselines.
These are general-purpose and intentionally simple.
"""
import random
from typing import List, Tuple
import numpy as np


def tournament_selection(population: List, fitnesses: List[float], k=3):
    idxs = random.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best]


def one_point_crossover(parent_a: List, parent_b: List) -> Tuple[List, List]:
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must be same length")
    n = len(parent_a)
    if n < 2:
        return parent_a.copy(), parent_b.copy()
    pt = random.randint(1, n - 1)
    child1 = parent_a[:pt] + parent_b[pt:]
    child2 = parent_b[:pt] + parent_a[pt:]
    return child1, child2


def uniform_mutation(chromosome: List, mut_rate: float, gene_space=None):
    child = chromosome.copy()
    for i in range(len(child)):
        if random.random() < mut_rate:
            if gene_space:
                child[i] = (
                    random.choice(gene_space[i])
                    if isinstance(gene_space, list)
                    else random.choice(gene_space)
                )
            else:
                # flip binary or add small noise for numeric
                if isinstance(child[i], (int, bool)):
                    child[i] = 1 - int(child[i])
                elif isinstance(child[i], float):
                    child[i] += random.gauss(0, 0.1 * abs(child[i] if child[i] else 1))
    return child
