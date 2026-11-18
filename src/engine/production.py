"""PitchIQ v2.5 production engine: trains the frozen ensemble recipe and serves predictions.

Recipe (frozen before the final test, see reports/v25/):
  equal-weight average of 7 models -- Elo logistic, core logistic, full logistic,
  Random Forest, XGBoost, separate home/away Poisson goal models, Dixon-Coles --
  all fed by the same leakage-safe FeatureEngine used in the backtest.
No manual home-advantage or draw adjustment is applied anywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.config import MODELS_DIR, REPORTS_DIR, TEAMS_2026_27
from src.engine.backtest import CORE_COLUMNS, FROZEN, MODEL_ORDER, _weights
from src.engine.dixon_coles import DixonColes, DixonColesParams
from src.engine.features import FEATURE_COLUMNS, FeatureEngine, RatingParams, build_feature_table
from src.engine.metrics import normalise_rows
from src.engine.models import ClassifierModel, SeparateGoalModels, logistic_factory, rf_factory, xgb_factory

VERSION = "2.5"
MODEL_FILE = MODELS_DIR / "pitchiq_v25.joblib"
ENSEMBLE_NAME = "PitchIQ v2.5 Ensemble"
ML_MODELS = ("elo_lr", "lr_core", "lr_full", "random_forest", "xgboost")
GOAL_MODELS = ("goal_poisson", "dixon_coles")
DISCLAIMER = ("Probabilities come from a statistical ensemble trained on 2016-2026 Premier League results. "
              "Football outcomes are highly random; measured holdout accuracy is about 50% and draws are not "
              "separable from the available data. This is not betting or financial advice.")


@dataclass
class ProductionEngine:
    models: dict[str, Any]
    dc: DixonColes
    feature_engine: FeatureEngine
    as_of: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)

    def feature_row(self, home: str, away: str, date: pd.Timestamp | None = None) -> pd.DataFrame:
        date = pd.Timestamp(date) if date is not None else self.as_of
        return pd.DataFrame([self.feature_engine.features(home, away, date)])

    def predict(self, home: str, away: str, date: pd.Timestamp | None = None) -> dict[str, Any]:
        row = self.feature_row(home, away, date)
        fallback = {home: (self.feature_engine.att[home], self.feature_engine.dfn[home]),
                    away: (self.feature_engine.att[away], self.feature_engine.dfn[away])}
        components = {n: self.models[n].predict_proba(row)[0] for n in MODEL_ORDER if n != "dixon_coles"}
        components["dixon_coles"] = self.dc.predict_probs(home, away, fallback)
        stack = np.vstack([components[n] for n in MODEL_ORDER])
        probs = normalise_rows(stack.mean(axis=0, keepdims=True))[0]
        lam_gp = self.models["goal_poisson"].lambdas(row)
        lam_dc = self.dc.lambdas(home, away, fallback)
        return {"probs": probs, "components": components,
                "lambda_home": float((lam_gp[0][0] + lam_dc[0]) / 2), "lambda_away": float((lam_gp[1][0] + lam_dc[1]) / 2),
                "features": row.iloc[0].to_dict()}


def known_current_results(openfootball: pd.DataFrame, repo_log: pd.DataFrame) -> pd.DataFrame:
    """Union of match-level 2026/27 results from OpenFootball and the repo log (dedupe by pairing)."""
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    played = openfootball[openfootball["Played"]][cols]
    log = repo_log[cols] if len(repo_log) else pd.DataFrame(columns=cols)
    merged = pd.concat([played, log], ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.drop_duplicates(["HomeTeam", "AwayTeam"], keep="first").sort_values("Date")
    merged["Season"] = "2026-27"
    return merged.reset_index(drop=True)


def _fit_models(table: pd.DataFrame, train_end_season: str) -> dict[str, Any]:
    train = table[(table["Season"] >= "2016-17") & (table["Season"] <= train_end_season)]
    y = train["y"].to_numpy()
    target = f"{int(train_end_season[:4]) + 1}-{str((int(train_end_season[:4]) + 2) % 100).zfill(2)}"
    cfg = FROZEN
    models: dict[str, Any] = {}
    specs = {
        "elo_lr": (logistic_factory(cfg["elo_lr"]["C"]), ["elo_diff"]),
        "lr_core": (logistic_factory(cfg["lr_core"]["C"]), CORE_COLUMNS),
        "lr_full": (logistic_factory(cfg["lr_full"]["C"]), FEATURE_COLUMNS),
        "random_forest": (rf_factory(cfg["random_forest"]["n_estimators"], cfg["random_forest"]["max_depth"],
                                     cfg["random_forest"]["min_samples_leaf"]), FEATURE_COLUMNS),
        "xgboost": (xgb_factory(cfg["xgboost"]["n_estimators"], cfg["xgboost"]["max_depth"]), FEATURE_COLUMNS),
    }
    for name, (factory, columns) in specs.items():
        models[name] = ClassifierModel(name, factory, columns).fit(train, y, _weights(train, target, cfg[name].get("half_life")))
    models["goal_poisson"] = SeparateGoalModels(FEATURE_COLUMNS, alpha=cfg["goal_poisson"]["alpha"]).fit(
        train, train["FTHG"].to_numpy(), train["FTAG"].to_numpy(), _weights(train, target, cfg["goal_poisson"].get("half_life")))
    return models


def _test_metrics_for_ui() -> list[dict[str, Any]]:
    path = REPORTS_DIR / "v25" / "test_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, m in data["metrics"].items():
        if name.startswith("Always-home"):
            continue
        label = ENSEMBLE_NAME if "production recipe" in name else name.replace("  <- production recipe", "")
        rows.append({"model": label, "accuracy": m["accuracy"], "log_loss": m["log_loss"], "brier_score": m["brier"],
                     "rps": m["rps"], "macro_f1": m["macro_f1"], "home_recall": m["home_recall"],
                     "draw_recall": m["draw_recall"], "away_recall": m["away_recall"]})
    return rows


def train_production_engine(historical: pd.DataFrame, current_results: pd.DataFrame | None = None,
                            season_teams_current: list[str] | None = None, save: bool = True) -> dict[str, Any]:
    """Fit all models on every complete historical season and prime ratings with 2026/27 results."""
    hist = historical.copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    last_complete = str(hist["Season"].max())
    frames = [hist]
    season_teams = {s: set(g["HomeTeam"]) | set(g["AwayTeam"]) for s, g in hist.groupby("Season")}
    if current_results is not None and len(current_results):
        frames.append(current_results)
        season_teams["2026-27"] = set(season_teams_current or TEAMS_2026_27)
    all_matches = pd.concat(frames, ignore_index=True)
    table, engine = build_feature_table(all_matches, RatingParams(), season_teams=season_teams)
    models = _fit_models(table, last_complete)         # supervised fit uses complete seasons only
    as_of = pd.Timestamp(all_matches["Date"].max()) + pd.Timedelta(days=1)
    dc = DixonColes(DixonColesParams(FROZEN["dixon_coles"]["half_life_days"], FROZEN["dixon_coles"]["ridge"])).fit(all_matches, as_of=as_of)
    train_rows = table[(table["Season"] >= "2016-17") & (table["Season"] <= last_complete)]
    metadata = {
        "version": VERSION, "sklearn_version": sklearn.__version__, "xgboost_version": xgboost.__version__,
        "model_name": "Elo/Logistic/RF/XGBoost classifiers", "ensemble_name": ENSEMBLE_NAME,
        "ml_probability_weight": len(ML_MODELS) / len(MODEL_ORDER), "poisson_probability_weight": len(GOAL_MODELS) / len(MODEL_ORDER),
        "home_advantage_factor": None,
        "ensemble_members": list(MODEL_ORDER), "ensemble_weights": {n: 1 / len(MODEL_ORDER) for n in MODEL_ORDER},
        "feature_columns": FEATURE_COLUMNS, "rating_params": RatingParams().to_dict(), "frozen_hyperparameters": FROZEN,
        "training_matches": int(len(hist)), "supervised_training_rows": int(len(train_rows)),
        "training_start": str(hist["Date"].min().date()), "training_end": str(hist["Date"].max().date()),
        "partial_current_season_results_excluded_from_training": True,
        "state_includes_current_season_matches": int(len(current_results)) if current_results is not None else 0,
        "validation_period": ["2022-23", "2023-24"], "test_period": ["2024-25", "2025-26"], "test_matches": 760,
        "test_metrics": _test_metrics_for_ui(), "disclaimer": DISCLAIMER,
    }
    bundle = {"metadata": metadata, "engine": ProductionEngine(models, dc, engine, as_of, metadata)}
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_FILE.with_suffix(".joblib.tmp")
        joblib.dump(bundle, tmp)
        tmp.replace(MODEL_FILE)          # atomic: a crash can never leave a half-written artifact
    return bundle


def load_or_train_model(historical: pd.DataFrame | None = None, current_results: pd.DataFrame | None = None) -> dict[str, Any]:
    """Load the saved v2.5 bundle; retrain deterministically if data or libraries changed."""
    from src.data_loader import load_historical_matches
    from src.data_sources.openfootball import load_openfootball_season
    from src.data_loader import load_current_season_results
    hist = historical if historical is not None else load_historical_matches()
    if current_results is None:
        try:
            current_results = known_current_results(load_openfootball_season("2026-27"), load_current_season_results())
        except FileNotFoundError:
            current_results = load_current_season_results().assign(Season="2026-27")
    bundle = None
    if MODEL_FILE.exists():
        try:
            bundle = joblib.load(MODEL_FILE)
        except Exception:            # corrupt/incompatible artifact -> retrain rather than crash
            bundle = None
    if bundle is not None:
        meta = bundle.get("metadata", {})
        if (meta.get("version") == VERSION and meta.get("training_matches") == len(hist)
                and meta.get("state_includes_current_season_matches") == len(current_results)
                and meta.get("sklearn_version") == sklearn.__version__ and meta.get("xgboost_version") == xgboost.__version__):
            return bundle
    return train_production_engine(hist, current_results)
