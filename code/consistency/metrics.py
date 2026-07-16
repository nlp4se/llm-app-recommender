"""Shared consistency metrics (aggregated across features)."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def jaccard_sets(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


def rbo(list1: list, list2: list, p: float = 0.9) -> float:
    """Rank-Biased Overlap in [0, 1]."""
    if not list1 or not list2:
        return 0.0
    if list1 == list2:
        return 1.0

    max_depth = max(len(list1), len(list2))
    agreement = 0.0
    normalization = 0.0
    for d in range(1, max_depth + 1):
        s1 = set(list1[:d])
        s2 = set(list2[:d])
        agreement += (len(s1 & s2) / d) * (p ** (d - 1))
        normalization += p ** (d - 1)
    return agreement / normalization if normalization > 0 else 0.0


def mean_pairwise_metric(pairs: Iterable[tuple], metric_fn) -> float:
    scores = [metric_fn(a, b) for a, b in pairs]
    return float(np.mean(scores)) if scores else 0.0
