"""Probabilistic and classification metrics for 3-way (H/D/A) football forecasts.

Class order everywhere in PitchIQ v2.5: 0 = Home win, 1 = Draw, 2 = Away win.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

EPS = 1e-12
CLASSES = ("H", "D", "A")


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(y):
        raise ValueError(f"Expected probabilities of shape (n, 3) matching y; got {p.shape} vs {y.shape}.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1 for every row.")
    return y, np.clip(p, EPS, 1.0)


def per_match_log_loss(y, p) -> np.ndarray:
    y, p = _check(y, p)
    return -np.log(p[np.arange(len(y)), y])


def log_loss(y, p) -> float:
    return float(per_match_log_loss(y, p).mean())


def per_match_brier(y, p) -> np.ndarray:
    y, p = _check(y, p)
    return ((p - np.eye(3)[y]) ** 2).sum(axis=1)


def brier(y, p) -> float:
    """Multiclass Brier score (sum over classes, mean over matches; range 0..2)."""
    return float(per_match_brier(y, p).mean())


def per_match_rps(y, p) -> np.ndarray:
    """Ranked probability score for ordered outcomes H < D < A (lower is better)."""
    y, p = _check(y, p)
    cum_p = np.cumsum(p, axis=1)[:, :2]
    cum_y = np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return ((cum_p - cum_y) ** 2).sum(axis=1) / 2.0


def rps(y, p) -> float:
    return float(per_match_rps(y, p).mean())


def classification_metrics(y, p) -> dict[str, float]:
    """Top-class (argmax) metrics. Draw recall under argmax is reported but see docs:
    a calibrated model rarely makes Draw the single most likely class."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y, p = _check(y, p)
    pred = p.argmax(axis=1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    prec = precision_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "home_recall": float(rec[0]), "draw_recall": float(rec[1]), "away_recall": float(rec[2]),
        "draw_precision": float(prec[1]),
        "predicted_draw_share": float((pred == 1).mean()),
    }


def expected_calibration_error(y, p, class_index: int, bins: int = 10) -> float:
    """One-vs-rest ECE for a single class using equal-width probability bins."""
    y, p = _check(y, p)
    prob = p[:, class_index]
    hit = (y == class_index).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = ids == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - hit[mask].mean())
    return float(ece)


def reliability_table(y, p, class_index: int, bins: int = 8) -> list[dict[str, float]]:
    """Quantile-binned reliability table (equal counts per bin => stable at n~760)."""
    y, p = _check(y, p)
    prob = p[:, class_index]
    hit = (y == class_index).astype(float)
    order = np.argsort(prob, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        rows.append({"n": int(len(chunk)), "mean_predicted": float(prob[chunk].mean()),
                     "observed_frequency": float(hit[chunk].mean())})
    return rows


def all_metrics(y, p) -> dict[str, float]:
    y, p = _check(y, p)
    out = {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p), "rps": rps(y, p)}
    out.update(classification_metrics(y, p))
    out["ece_home"] = expected_calibration_error(y, p, 0)
    out["ece_draw"] = expected_calibration_error(y, p, 1)
    out["ece_away"] = expected_calibration_error(y, p, 2)
    out["mean_p_draw"] = float(p[:, 1].mean())
    out["observed_draw_rate"] = float((y == 1).mean())
    return out


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 5000, seed: int = 20260101) -> dict[str, float]:
    """Paired bootstrap of mean(loss_a - loss_b). Negative delta => A is better.

    Matches are resampled with replacement (i.i.d.), which ignores week-level
    dependence; treat the interval as indicative, not exact.
    """
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.shape != b.shape:
        raise ValueError("Paired losses must have the same shape.")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "prob_a_better": float((means < 0).mean())}


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 20260101) -> tuple[float, float]:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def normalise_rows(p: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, float), floor, None)
    return p / p.sum(axis=1, keepdims=True)


def labels_to_int(labels) -> np.ndarray:
    mapping: Mapping[str, int] = {"H": 0, "D": 1, "A": 2}
    return np.asarray([mapping[str(x)] for x in labels], dtype=int)
