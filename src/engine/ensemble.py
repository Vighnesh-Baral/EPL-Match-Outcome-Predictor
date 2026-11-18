"""Ensembling and (optional) calibration utilities fitted on out-of-sample predictions only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.engine.metrics import log_loss, normalise_rows


def blend(probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, float)
    return normalise_rows(sum(w * p for w, p in zip(weights, probs)))


def fit_weights(probs: list[np.ndarray], y: np.ndarray, shrink: float = 0.0) -> np.ndarray:
    """Convex (non-negative, sum-to-one) weights minimising log loss.

    ``shrink`` adds ``shrink * ||w - uniform||^2`` to pull toward equal weighting, which
    stabilises the solution on small selection samples.
    """
    k = len(probs)
    uniform = np.full(k, 1.0 / k)

    def objective(w: np.ndarray) -> float:
        return log_loss(y, blend(probs, w)) + shrink * float(np.sum((w - uniform) ** 2))

    result = minimize(objective, uniform, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}], options={"maxiter": 200, "ftol": 1e-9})
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def temperature(p: np.ndarray, T: float) -> np.ndarray:
    """Multiclass temperature scaling on log-probabilities (T > 1 softens, T < 1 sharpens)."""
    logits = np.log(np.clip(p, 1e-9, 1.0)) / T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    result = minimize(lambda t: log_loss(y, temperature(p, float(t[0]))), x0=[1.0], bounds=[(0.6, 1.8)], method="L-BFGS-B")
    return float(result.x[0])
