"""
Small helper utilities used across the project.
"""
from typing import Iterable, List, Tuple
import numpy as np


def chunked(iterable: Iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def set_seed(seed: int):
    import random, numpy as _np

    random.seed(seed)
    _np.random.seed(seed)


def safe_div(a, b):
    return a / b if b else 0.0


def flatten(list_of_lists):
    return [x for l in list_of_lists for x in l]
