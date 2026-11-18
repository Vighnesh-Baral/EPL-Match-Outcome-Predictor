"""Version 2.5 match prediction: leakage-safe rating/feature engine + 7-model ensemble.

The public contract of ``predict_match`` is unchanged from v2.4 (same keys), with additive
fields.  The legacy hand-weighted "current-season Poisson" branch has been removed from the
probability path: it was never backtested and differed from the model that was evaluated.
Squad ratings (team_ratings) are still returned for display, but are not injected into the
probabilities because they cannot be validated historically.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import standardize_team_name
from src.engine.production import ML_MODELS, GOAL_MODELS


@lru_cache(maxsize=1)
def _schedule_lookup() -> dict[tuple[str, str], tuple[pd.Timestamp, str | None]]:
    """(home, away) -> (date, kick-off) from the cached OpenFootball 2026/27 schedule."""
    try:
        from src.data_sources.openfootball import load_openfootball_season
        sched = load_openfootball_season("2026-27")
    except FileNotFoundError:
        return {}
    return {(r.HomeTeam, r.AwayTeam): (r.Date, r.KickOff) for r in sched.itertuples()}


def fixture_schedule(home: str, away: str) -> dict[str, str | None]:
    entry = _schedule_lookup().get((home, away))
    if entry is None:
        return {"date": None, "kickoff": None}
    return {"date": str(entry[0].date()), "kickoff": entry[1]}


def _pct(p: np.ndarray) -> dict[str, float]:
    return {"home_win": round(float(p[0]) * 100, 2), "draw": round(float(p[1]) * 100, 2), "away_win": round(float(p[2]) * 100, 2)}


def predict_match(home_team: str, away_team: str, team_strength: pd.DataFrame,
                  form: pd.DataFrame, historical_matches: pd.DataFrame, model_bundle: dict) -> dict:
    home = standardize_team_name(home_team)
    away = standardize_team_name(away_team)
    if home == away:
        raise ValueError("Home and away teams must be different.")
    team_idx = team_strength.set_index("Squad")
    form_idx = form.set_index("Squad")
    if home not in team_idx.index:
        raise ValueError(f"Unknown home club: {home}")
    if away not in team_idx.index:
        raise ValueError(f"Unknown away club: {away}")
    h, a = team_idx.loc[home], team_idx.loc[away]
    hf, af = form_idx.loc[home], form_idx.loc[away]
    metadata = model_bundle["metadata"]
    engine = model_bundle["engine"]

    schedule = fixture_schedule(home, away)
    date = pd.Timestamp(schedule["date"]) if schedule["date"] else None
    result = engine.predict(home, away, date)
    probs = result["probs"]
    comps = result["components"]
    ml_mean = np.mean([comps[n] for n in ML_MODELS], axis=0)
    goal_mean = np.mean([comps[n] for n in GOAL_MODELS], axis=0)

    # Class order is Home Win, Draw, Away Win.  The label is the plain argmax: probabilities are
    # never boosted or reweighted toward any outcome.
    pct = _pct(probs)
    pct["away_win"] = round(100.0 - pct["home_win"] - pct["draw"], 2)
    outcome_key = max(pct, key=pct.get)
    outcome = {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}[outcome_key]
    home_probs = [float(c[0]) for c in comps.values()]
    feats = result["features"]

    return {
        "home_team": home, "away_team": away, "probabilities": pct, "predicted_outcome": outcome,
        "expected_goals": {"home": round(result["lambda_home"], 2), "away": round(result["lambda_away"], 2)},
        "team_ratings": {
            "home": {"overall": float(h["Overall_Team_Strength"]), "attack": float(h["Attacking_Strength"]), "midfield": float(h["Midfield_Strength"]), "defence": float(h["Defensive_Strength"]), "goalkeeping": float(h["Goalkeeping_Strength"]), "depth": float(h["Squad_Depth_Score"])},
            "away": {"overall": float(a["Overall_Team_Strength"]), "attack": float(a["Attacking_Strength"]), "midfield": float(a["Midfield_Strength"]), "defence": float(a["Defensive_Strength"]), "goalkeeping": float(a["Goalkeeping_Strength"]), "depth": float(a["Squad_Depth_Score"])},
        },
        "current_form": {"home": str(hf["Last_5"]), "away": str(af["Last_5"])},
        "model": metadata["ensemble_name"],
        "model_components": {
            "historical_classifier": metadata["model_name"],
            "historical_classifier_weight_pct": round(float(metadata["ml_probability_weight"]) * 100, 1),
            "current_season_poisson_weight_pct": round(float(metadata["poisson_probability_weight"]) * 100, 1),
            "home_advantage_factor": None,
            "home_advantage_note": "Learned from data (no manual factor)",
            "ensemble_members": metadata["ensemble_members"],
        },
        "component_probabilities": {
            "historical_classifier": _pct(ml_mean),
            "current_season_poisson": _pct(goal_mean),
        },
        "model_probabilities": {name: _pct(p) for name, p in comps.items()},
        "model_agreement": {"home_win_spread_pct": round((max(home_probs) - min(home_probs)) * 100, 1)},
        "rating_context": {"elo_home": round(float(feats["elo_home"])), "elo_away": round(float(feats["elo_away"])),
                           "rest_days_home": feats["h_rest"], "rest_days_away": feats["a_rest"]},
        "fixture": schedule,
        "training": {
            "historical_matches": metadata["training_matches"],
            "training_period": f'{metadata["training_start"]} to {metadata["training_end"]}',
            "test_period": "2024/25–2025/26", "test_metrics": metadata["test_metrics"],
        },
        "model_disclaimer": metadata["disclaimer"],
    }
