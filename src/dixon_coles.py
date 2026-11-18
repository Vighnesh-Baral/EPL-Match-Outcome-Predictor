"""Dixon-Coles bivariate goal distribution model for association football.

Reference:
    Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market. Journal of the Royal
    Statistical Society: Series C (Applied Statistics), 46(2), 265-280.
"""
from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import pandas as pd


def tau_adjustment(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Compute the Dixon-Coles low-scoring dependence adjustment factor tau_{x, y}."""
    if x == 0 and y == 0:
        val = 1.0 - lam * mu * rho
    elif x == 0 and y == 1:
        val = 1.0 + lam * rho
    elif x == 1 and y == 0:
        val = 1.0 + mu * rho
    elif x == 1 and y == 1:
        val = 1.0 - rho
    else:
        val = 1.0
    return max(0.0, float(val))


def dixon_coles_joint_matrix(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.11,
    max_goals: int = 10,
) -> np.ndarray:
    """Compute the joint probability matrix P(X=x, Y=y) for scores up to max_goals."""
    lam = float(np.clip(lambda_h, 0.05, 6.0))
    mu = float(np.clip(lambda_a, 0.05, 6.0))
    r = float(np.clip(rho, -0.25, 0.10))

    # Precompute 1D Poisson marginals
    p_h = np.array([math.exp(-lam) * (lam**x) / math.factorial(x) for x in range(max_goals + 1)], dtype=float)
    p_a = np.array([math.exp(-mu) * (mu**y) / math.factorial(y) for y in range(max_goals + 1)], dtype=float)

    # Outer product for independent Poisson
    matrix = np.outer(p_h, p_a)

    # Apply Dixon-Coles tau adjustments to low scores
    matrix[0, 0] *= max(0.0, 1.0 - lam * mu * r)
    matrix[0, 1] *= max(0.0, 1.0 + lam * r)
    matrix[1, 0] *= max(0.0, 1.0 + mu * r)
    matrix[1, 1] *= max(0.0, 1.0 - r)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def dixon_coles_outcome_probs(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.11,
    max_goals: int = 10,
) -> Tuple[float, float, float]:
    """Return (P(Home Win), P(Draw), P(Away Win)) summing to 1.0."""
    matrix = dixon_coles_joint_matrix(lambda_h, lambda_a, rho=rho, max_goals=max_goals)
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def batch_dixon_coles_probs(
    lambdas_h: np.ndarray,
    lambdas_a: np.ndarray,
    rho: float = -0.11,
    max_goals: int = 10,
) -> np.ndarray:
    """Vectorized / batched computation of Dixon-Coles outcome probabilities."""
    n = len(lambdas_h)
    out = np.zeros((n, 3), dtype=float)
    for i in range(n):
        h, d, a = dixon_coles_outcome_probs(lambdas_h[i], lambdas_a[i], rho=rho, max_goals=max_goals)
        out[i, 0] = h
        out[i, 1] = d
        out[i, 2] = a
    return out

