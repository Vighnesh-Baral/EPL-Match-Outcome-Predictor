"""Candidate model factories and thin wrappers (all return ordered H/D/A probabilities)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.engine.metrics import normalise_rows


def derive_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add symmetric 'closeness' features so linear models can express draw propensity."""
    out = frame.copy()
    if "elo_diff" in out:
        out["abs_elo_diff"] = out["elo_diff"].abs()
    if "lam_diff" in out:
        out["abs_lam_diff"] = out["lam_diff"].abs()
    if {"lam_home", "lam_away"} <= set(out.columns):
        out["lam_total"] = out["lam_home"] + out["lam_away"]
    return out


def ordered_probs(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    full = np.zeros((len(X), 3))
    for column, cls in enumerate(model.classes_):
        full[:, int(cls)] = raw[:, column]
    return normalise_rows(full)


class ClassifierModel:
    """Fit/predict wrapper around sklearn-style classifiers with optional sample weights."""

    def __init__(self, name: str, factory, columns: Sequence[str], derive: bool = True):
        self.name, self.factory, self.columns, self.derive = name, factory, list(columns), derive
        self.model = None
        self.used_columns: list[str] = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state["factory"] = None        # lambdas are not picklable; the fitted model is all inference needs
        return state

    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        data = derive_columns(frame) if self.derive else frame
        extras = []
        if self.derive:
            if "elo_diff" in self.columns:
                extras.append("abs_elo_diff")
            if "lam_diff" in self.columns:
                extras.append("abs_lam_diff")
            if "lam_home" in self.columns and "lam_away" in self.columns:
                extras.append("lam_total")
        self.used_columns = self.columns + [c for c in extras if c not in self.columns]
        return data[self.used_columns].to_numpy(dtype=float)

    def fit(self, frame: pd.DataFrame, y: np.ndarray, weights: np.ndarray | None = None) -> "ClassifierModel":
        self.model = self.factory()
        X = self._x(frame)
        if isinstance(self.model, Pipeline):
            self.model.fit(X, y, **({"model__sample_weight": weights} if weights is not None else {}))
        else:
            self.model.fit(X, y, **({"sample_weight": weights} if weights is not None else {}))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return ordered_probs(self.model, self._x(frame))


def logistic_factory(C: float = 0.3):
    return lambda: Pipeline([("scale", StandardScaler()),
                             ("model", LogisticRegression(C=C, max_iter=3000, solver="lbfgs"))])


def rf_factory(n_estimators: int = 300, max_depth: int = 6, min_samples_leaf: int = 25, max_features: str | float = "sqrt"):
    return lambda: RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, random_state=42, n_jobs=1)


def xgb_factory(n_estimators: int = 200, max_depth: int = 2, learning_rate: float = 0.03, min_child_weight: float = 15,
                subsample: float = 0.8, colsample_bytree: float = 0.7, reg_lambda: float = 10.0):
    def make():
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
                             reg_lambda=reg_lambda, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                             random_state=42, n_jobs=1, tree_method="hist", verbosity=0)
    return make


# ---------------------------------------------------------------------------------------
# Score-line machinery shared by the goal models and the Dixon-Coles model
# ---------------------------------------------------------------------------------------
MAX_GOALS = 10


def poisson_pmf_matrix(lams: np.ndarray, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(n, max_goals+1) Poisson pmf rows (tail mass folded into the last bin)."""
    lams = np.clip(np.asarray(lams, float), 0.05, 6.0)[:, None]
    k = np.arange(max_goals + 1)[None, :]
    log_pmf = -lams + k * np.log(lams) - np.array([math.lgamma(i + 1) for i in range(max_goals + 1)])[None, :]
    pmf = np.exp(log_pmf)
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
    return pmf


def dc_tau(x: int, y: int, lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return np.full_like(lam_h, 1.0 - rho)
    return np.ones_like(lam_h)


def score_matrix(lam_h: np.ndarray, lam_a: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """(n, G+1, G+1) joint score probabilities with the Dixon-Coles low-score correction."""
    lam_h = np.clip(np.asarray(lam_h, float), 0.05, 6.0)
    lam_a = np.clip(np.asarray(lam_a, float), 0.05, 6.0)
    ph, pa = poisson_pmf_matrix(lam_h), poisson_pmf_matrix(lam_a)
    joint = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        for x in (0, 1):
            for y in (0, 1):
                joint[:, x, y] *= np.clip(dc_tau(x, y, lam_h, lam_a, rho), 1e-6, None)
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    return joint


def outcome_probs_from_scores(joint: np.ndarray) -> np.ndarray:
    g = joint.shape[1]
    idx = np.arange(g)
    home_mask = idx[:, None] > idx[None, :]
    draw_mask = idx[:, None] == idx[None, :]
    away_mask = idx[:, None] < idx[None, :]
    return np.stack([(joint * home_mask).sum(axis=(1, 2)), (joint * draw_mask).sum(axis=(1, 2)),
                     (joint * away_mask).sum(axis=(1, 2))], axis=1)


def estimate_rho(goals_h: np.ndarray, goals_a: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray,
                 weights: np.ndarray | None = None) -> float:
    """Profile-likelihood estimate of the Dixon-Coles dependence parameter (bounded search)."""
    from scipy.optimize import minimize_scalar
    w = np.ones(len(goals_h)) if weights is None else np.asarray(weights, float)
    low = (goals_h <= 1) & (goals_a <= 1)
    if low.sum() < 30:
        return 0.0
    gh, ga, lh, la, ww = goals_h[low].astype(int), goals_a[low].astype(int), lam_h[low], lam_a[low], w[low]

    def neg_ll(rho: float) -> float:
        total = 0.0
        for x in (0, 1):
            for y in (0, 1):
                m = (gh == x) & (ga == y)
                if m.any():
                    total += float((ww[m] * np.log(np.clip(dc_tau(x, y, lh[m], la[m], rho), 1e-9, None))).sum())
        return -total
    bound = 0.95 * min(1.0 / max(float(lam_h.max()), 1e-6), 1.0 / max(float(lam_a.max()), 1e-6), 1.0)
    result = minimize_scalar(neg_ll, bounds=(-bound, bound), method="bounded")
    return float(result.x)


class SeparateGoalModels:
    """Separate regularised Poisson regressions for home goals and away goals,
    combined into H/D/A probabilities via a Dixon-Coles-corrected score grid."""

    def __init__(self, columns: Sequence[str], alpha: float = 20.0, use_rho: bool = True):
        self.columns, self.alpha, self.use_rho = list(columns), alpha, use_rho
        self.rho = 0.0
        self.scaler = StandardScaler()
        self.home_model = PoissonRegressor(alpha=alpha, max_iter=1000)
        self.away_model = PoissonRegressor(alpha=alpha, max_iter=1000)

    def fit(self, frame: pd.DataFrame, goals_h: np.ndarray, goals_a: np.ndarray, weights: np.ndarray | None = None) -> "SeparateGoalModels":
        X = self.scaler.fit_transform(frame[self.columns].to_numpy(dtype=float))
        self.home_model = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, goals_h, sample_weight=weights)
        self.away_model = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, goals_a, sample_weight=weights)
        if self.use_rho:
            self.rho = estimate_rho(goals_h, goals_a, self.home_model.predict(X), self.away_model.predict(X), weights)
        return self

    def lambdas(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.scaler.transform(frame[self.columns].to_numpy(dtype=float))
        return self.home_model.predict(X), self.away_model.predict(X)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.lambdas(frame)
        return normalise_rows(outcome_probs_from_scores(score_matrix(lam_h, lam_a, self.rho)))
