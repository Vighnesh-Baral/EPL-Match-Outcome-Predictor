"""Model registry and rolling-origin backtest runners (PitchIQ v2.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.engine.dixon_coles import DixonColesParams, rolling_dixon_coles
from src.engine.features import FEATURE_COLUMNS, FEATURE_GROUPS
from src.engine.metrics import normalise_rows
from src.engine.models import ClassifierModel, SeparateGoalModels, logistic_factory, rf_factory, xgb_factory
from src.engine.protocol import probs_of, rolling_origin

CORE_COLUMNS = FEATURE_GROUPS["goal_ratings"] + FEATURE_GROUPS["context"]

# Frozen hyper-parameters, chosen on the TUNING window (see reports/v25/tuning_results.json)
FROZEN = {
    "lr_full": {"C": 0.003, "half_life": None},
    "lr_core": {"C": 0.01, "half_life": 8.0},
    "elo_lr": {"C": 1.0, "half_life": None},
    "random_forest": {"n_estimators": 250, "max_depth": 4, "min_samples_leaf": 20, "half_life": 8.0},
    "xgboost": {"n_estimators": 250, "max_depth": 1, "half_life": 8.0},
    "goal_poisson": {"alpha": 1.0, "half_life": None},
    "dixon_coles": {"half_life_days": 550.0, "ridge": 2.0, "refit_days": 14},
}
MODEL_ORDER = ["elo_lr", "lr_core", "lr_full", "random_forest", "xgboost", "goal_poisson", "dixon_coles"]


def _weights(train: pd.DataFrame, season: str, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(train))
    age = int(season[:4]) - train["Season"].str[:4].astype(int).to_numpy()
    return 0.5 ** ((age - 1) / half_life)


def make_runner(name: str, matches: pd.DataFrame, fixed_origin_before: str | None = None) -> Callable:
    """Return ``fit_predict(train, test, season)`` for model ``name``.

    ``fixed_origin_before``: if set, training rows/DC fits are frozen at the start of that
    season (the v2.4 evaluation protocol: fit once, predict two seasons).
    """
    cfg = FROZEN[name]

    def restrict(train: pd.DataFrame, test: pd.DataFrame, season: str) -> tuple[pd.DataFrame, str]:
        if fixed_origin_before is None:
            return train, season
        base = pd.concat([train, test.iloc[0:0]])
        return base[base["Season"] < fixed_origin_before], fixed_origin_before

    if name == "dixon_coles":
        def run(train, test, season):
            params = DixonColesParams(cfg["half_life_days"], cfg["ridge"],
                                      refit_days=10 ** 6 if fixed_origin_before else cfg["refit_days"])
            if fixed_origin_before:
                # fit once at the origin, then score every test row with that single fit
                from src.engine.dixon_coles import DixonColes
                origin = pd.Timestamp(test["Date"].min()) if season == fixed_origin_before else None
                model = DixonColes(params).fit(matches, as_of=pd.Timestamp(f"{fixed_origin_before[:4]}-07-31"))
                out = []
                for r in test.itertuples():
                    fb = {r.HomeTeam: (r.att_home, r.def_home), r.AwayTeam: (r.att_away, r.def_away)}
                    out.append(model.predict_probs(r.HomeTeam, r.AwayTeam, fb))
                return np.vstack(out)
            return rolling_dixon_coles(matches, test, params)
        return run

    def classifier_run(factory, columns):
        def run(train, test, season):
            tr, origin = restrict(train, test, season)
            model = ClassifierModel(name, factory, columns).fit(tr, tr["y"].to_numpy(), _weights(tr, origin, cfg.get("half_life")))
            return model.predict_proba(test)
        return run

    if name == "elo_lr":
        return classifier_run(logistic_factory(cfg["C"]), ["elo_diff"])
    if name == "lr_core":
        return classifier_run(logistic_factory(cfg["C"]), CORE_COLUMNS)
    if name == "lr_full":
        return classifier_run(logistic_factory(cfg["C"]), FEATURE_COLUMNS)
    if name == "random_forest":
        return classifier_run(rf_factory(cfg["n_estimators"], cfg["max_depth"], cfg["min_samples_leaf"]), FEATURE_COLUMNS)
    if name == "xgboost":
        return classifier_run(xgb_factory(cfg["n_estimators"], cfg["max_depth"]), FEATURE_COLUMNS)
    if name == "goal_poisson":
        def run(train, test, season):
            tr, origin = restrict(train, test, season)
            model = SeparateGoalModels(FEATURE_COLUMNS, alpha=cfg["alpha"]).fit(
                tr, tr["FTHG"].to_numpy(), tr["FTAG"].to_numpy(), _weights(tr, origin, cfg.get("half_life")))
            return model.predict_proba(test)
        return run
    raise KeyError(name)


def climatology_runner(train: pd.DataFrame, test: pd.DataFrame, season: str) -> np.ndarray:
    freq = np.bincount(train["y"], minlength=3) / len(train)
    return np.tile(freq, (len(test), 1))


def always_home_runner(train, test, season) -> np.ndarray:
    return np.tile(np.array([0.9998, 0.0001, 0.0001]), (len(test), 1))   # accuracy baseline only


def run_models(table: pd.DataFrame, matches: pd.DataFrame, seasons: tuple[str, ...], names=MODEL_ORDER,
               fixed_origin_before: str | None = None) -> dict[str, pd.DataFrame]:
    return {n: rolling_origin(table, seasons, make_runner(n, matches, fixed_origin_before)) for n in names}
