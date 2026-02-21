"""Utility helpers for the ETHOS Demo application."""

import math


def wilson_margin(k: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    return z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
