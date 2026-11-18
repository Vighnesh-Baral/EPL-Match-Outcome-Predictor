"""Time-decayed Dixon-Coles model (independent-Poisson MLE + low-score dependence).

Parameters: per-club attack and defence (ridge-shrunk toward league average),
a home-advantage term and a goals intercept.  Matches are weighted by
``0.5 ** (age_days / half_life_days)`` where age is measured from ``as_of``; only matches
dated strictly before ``as_of`` are used.  The dependence parameter rho is estimated by
a one-dimensional profile likelihood given the fitted rates (standard practical variant).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.engine.metrics import normalise_rows
from src.engine.models import estimate_rho, outcome_probs_from_scores, score_matrix


@dataclass
class DixonColesParams:
    half_life_days: float = 550.0
    ridge: float = 2.0
    refit_days: int = 14
    min_matches: int = 200


class DixonColes:
    def __init__(self, params: DixonColesParams | None = None):
        self.params = params or DixonColesParams()
        self.teams: list[str] = []
        self.att: dict[str, float] = {}
        self.dfn: dict[str, float] = {}
        self.hfa = 0.25
        self.mu = 0.0
        self.rho = 0.0
        self.as_of: pd.Timestamp | None = None
        self.n_fit = 0

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp) -> "DixonColes":
        data = matches[matches["Date"] < as_of]
        if len(data) < self.params.min_matches:
            raise ValueError(f"Dixon-Coles needs >= {self.params.min_matches} prior matches; got {len(data)}.")
        teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = data["HomeTeam"].map(index).to_numpy()
        a = data["AwayTeam"].map(index).to_numpy()
        gh = data["FTHG"].to_numpy(dtype=float)
        ga = data["FTAG"].to_numpy(dtype=float)
        age = (as_of - data["Date"]).dt.days.to_numpy(dtype=float)
        w = 0.5 ** (age / self.params.half_life_days)
        ridge = self.params.ridge

        def objective(theta: np.ndarray):
            att, dfn, hfa, mu = theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1]
            eta_h = mu + hfa + att[h] - dfn[a]
            eta_a = mu + att[a] - dfn[h]
            lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)
            nll = -np.sum(w * (gh * eta_h - lam_h + ga * eta_a - lam_a))
            nll += ridge * (np.sum(att ** 2) + np.sum(dfn ** 2))
            err_h = w * (gh - lam_h)
            err_a = w * (ga - lam_a)
            g_att = -(np.bincount(h, err_h, n) + np.bincount(a, err_a, n)) + 2 * ridge * att
            g_dfn = (np.bincount(a, err_h, n) + np.bincount(h, err_a, n)) + 2 * ridge * dfn
            g_hfa = -err_h.sum()
            g_mu = -(err_h.sum() + err_a.sum())
            return nll, np.concatenate([g_att, g_dfn, [g_hfa, g_mu]])

        x0 = np.zeros(2 * n + 2)
        x0[2 * n + 1] = np.log(max(gh.mean(), 0.5))
        result = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        if not np.isfinite(result.fun):
            raise RuntimeError("Dixon-Coles optimisation failed (non-finite objective).")
        theta = result.x
        self.teams = teams
        self.att = {t: float(theta[i]) for t, i in index.items()}
        self.dfn = {t: float(theta[n + i]) for t, i in index.items()}
        self.hfa, self.mu = float(theta[2 * n]), float(theta[2 * n + 1])
        lam_h = np.exp(self.mu + self.hfa + theta[:n][h] - theta[n:2 * n][a])
        lam_a = np.exp(self.mu + theta[:n][a] - theta[n:2 * n][h])
        self.rho = estimate_rho(gh, ga, lam_h, lam_a, w)
        self.as_of, self.n_fit = as_of, int(len(data))
        return self

    def lambdas(self, home: str, away: str, fallback: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
        """Expected goals.  Clubs absent from the fit window use ``fallback[club] = (att, dfn)``."""
        def rating(team: str) -> tuple[float, float]:
            if team in self.att:
                return self.att[team], self.dfn[team]
            if fallback and team in fallback:
                return fallback[team]
            return 0.0, 0.0
        ah, dh = rating(home)
        aa, da = rating(away)
        lam_h = float(np.exp(self.mu + self.hfa + ah - da))
        lam_a = float(np.exp(self.mu + aa - dh))
        return float(np.clip(lam_h, 0.15, 5.0)), float(np.clip(lam_a, 0.15, 5.0))

    def predict_probs(self, home: str, away: str, fallback=None) -> np.ndarray:
        lam_h, lam_a = self.lambdas(home, away, fallback)
        joint = score_matrix(np.array([lam_h]), np.array([lam_a]), self.rho)
        return normalise_rows(outcome_probs_from_scores(joint))[0]


def rolling_dixon_coles(all_matches: pd.DataFrame, eval_rows: pd.DataFrame, params: DixonColesParams,
                        fallback_columns: tuple[str, str, str, str] = ("att_home", "def_home", "att_away", "def_away")) -> np.ndarray:
    """Out-of-sample DC probabilities for ``eval_rows`` (must carry the feature-table columns).

    The model is refit at most every ``refit_days`` days using *only* matches dated strictly
    before the refit date; rows between refits reuse the latest fit (never a future fit).
    """
    matches = all_matches.sort_values("Date")
    rows = eval_rows.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable")
    model, last_fit = None, None
    out = {}
    for r in rows.itertuples():
        date = r.Date
        if model is None or (date - last_fit).days >= params.refit_days:
            model = DixonColes(params).fit(matches, as_of=date)
            last_fit = date
        fallback = {r.HomeTeam: (getattr(r, fallback_columns[0]), getattr(r, fallback_columns[1])),
                    r.AwayTeam: (getattr(r, fallback_columns[2]), getattr(r, fallback_columns[3]))}
        out[r.Index] = model.predict_probs(r.HomeTeam, r.AwayTeam, fallback)
    ordered = np.vstack([out[i] for i in eval_rows.index])
    return ordered
