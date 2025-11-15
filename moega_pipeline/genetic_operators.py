import random
from typing import List, Dict, Any, Tuple
import copy
import numpy as np


def tournament_selection(population: List[Dict], fitnesses: List[float], k: int = 3):
    idxs = random.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best])


def one_point_crossover(parent_a: Dict, parent_b: Dict) -> Tuple[Dict, Dict]:
    # Crossover the feature_mask at one point, and swap params with some probability.
    child_a = copy.deepcopy(parent_a)
    child_b = copy.deepcopy(parent_b)

    mask_len = len(parent_a["feature_mask"])
    if mask_len > 1:
        pt = random.randint(1, mask_len - 1)
        a_mask = parent_a["feature_mask"]
        b_mask = parent_b["feature_mask"]
        child_a["feature_mask"] = a_mask[:pt] + b_mask[pt:]
        child_b["feature_mask"] = b_mask[:pt] + a_mask[pt:]

    # For params, do uniform crossover per param key
    for key in ["xgb_params", "lgbm_params", "rf_params"]:
        for p in parent_a[key].keys():
            if random.random() < 0.5:
                child_a[key][p] = parent_a[key][p]
                child_b[key][p] = parent_b[key][p]
            else:
                child_a[key][p] = parent_b[key][p]
                child_b[key][p] = parent_a[key][p]
    return child_a, child_b


def mutate(
    chrom: Dict[str, Any], mut_rate_mask: float = 0.01, mut_rate_param: float = 0.1
):
    # Bitflip mutation on mask
    mask = chrom["feature_mask"]
    for i in range(len(mask)):
        if random.random() < mut_rate_mask:
            mask[i] = 1 - mask[i]
    chrom["feature_mask"] = mask

    # Gaussian perturbation for numeric params
    for key in ["xgb_params", "lgbm_params", "rf_params"]:
        for p, v in chrom[key].items():
            if random.random() < mut_rate_param:
                if isinstance(v, int):
                    # mutate in ±20% range
                    delta = max(1, int(0.2 * v))
                    chrom[key][p] = max(1, int(v + random.randint(-delta, delta)))
                elif isinstance(v, float):
                    sigma = 0.15 * (abs(v) if v != 0 else 1.0)
                    chrom[key][p] = round(
                        max(1e-6, float(np.random.normal(v, sigma))), 6
                    )
    return chrom
