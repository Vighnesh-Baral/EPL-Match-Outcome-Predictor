"""Version 2: recency-weighted historical classifier + current-season Poisson branch."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.config import HOME_ADVANTAGE_FACTOR, POISSON_MAX_GOALS, standardize_team_name
from src.historical_model import build_current_fixture_features, predict_classifier_probabilities


def _poisson_probs(lam: float, max_goals: int = POISSON_MAX_GOALS) -> np.ndarray:
    lam = max(0.05, min(float(lam), 6.0))
    values = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(max_goals + 1)], dtype=float)
    total = values.sum()
    return values / total if total > 0 else np.eye(1, max_goals + 1, 0)[0]


def _poisson_outcome_probs(lambda_home: float, lambda_away: float) -> np.ndarray:
    home_goals = _poisson_probs(lambda_home)
    away_goals = _poisson_probs(lambda_away)
    matrix = np.outer(home_goals, away_goals)
    values = np.array([np.tril(matrix, k=-1).sum(), np.trace(matrix), np.triu(matrix, k=1).sum()], dtype=float)
    return values / values.sum()


def _normalise_probabilities(values: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    return result / result.sum()


def predict_match(home_team: str, away_team: str, team_strength: pd.DataFrame,
                  form: pd.DataFrame, historical_matches: pd.DataFrame,
                  model_bundle: dict) -> dict:
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
    league_goal_rate = float(team_strength["GF"].sum() / team_strength["MP"].sum())

    # Current-snapshot team-strength adjustments. The 2026/27 FBref + FPL team
    # ratings are used here, while historic team-result patterns are learned by
    # the classifier branch. Component multipliers remain bounded and visible.
    h_attack_factor = 0.75 + 0.50 * (float(h["Attacking_Strength"]) / 100.0)
    a_attack_factor = 0.75 + 0.50 * (float(a["Attacking_Strength"]) / 100.0)
    h_defence_factor = 1.25 - 0.50 * (float(h["Defensive_Strength"]) / 100.0)
    a_defence_factor = 1.25 - 0.50 * (float(a["Defensive_Strength"]) / 100.0)
    h_midfield_factor = 0.97 + 0.06 * (float(h["Midfield_Strength"]) / 100.0)
    a_midfield_factor = 0.97 + 0.06 * (float(a["Midfield_Strength"]) / 100.0)
    h_goalkeeper_factor = 1.08 - 0.16 * (float(h["Goalkeeping_Strength"]) / 100.0)
    a_goalkeeper_factor = 1.08 - 0.16 * (float(a["Goalkeeping_Strength"]) / 100.0)
    h_form_factor = 0.95 + 0.10 * (float(hf["Form_Rating_Last_5"]) / 100.0)
    a_form_factor = 0.95 + 0.10 * (float(af["Form_Rating_Last_5"]) / 100.0)
    h_squad_factor = 0.97 + 0.06 * (float(h["Squad_Depth_Score"]) / 100.0)
    a_squad_factor = 0.97 + 0.06 * (float(a["Squad_Depth_Score"]) / 100.0)

    # Explicit home advantage remains independent of the club-strength ratings.
    lambda_home = league_goal_rate * h_attack_factor * a_defence_factor * h_midfield_factor * a_goalkeeper_factor * h_form_factor * h_squad_factor * HOME_ADVANTAGE_FACTOR
    lambda_away = league_goal_rate * a_attack_factor * h_defence_factor * a_midfield_factor * h_goalkeeper_factor * a_form_factor * a_squad_factor / HOME_ADVANTAGE_FACTOR
    lambda_home = float(np.clip(lambda_home, 0.15, 4.5))
    lambda_away = float(np.clip(lambda_away, 0.15, 4.5))
    poisson_probs = _poisson_outcome_probs(lambda_home, lambda_away)

    h2h_features = build_current_fixture_features(
        h, a, hf, af, historical_matches, league_avg_team_goals=league_goal_rate
    )
    ml_probs = predict_classifier_probabilities(model_bundle, h2h_features)[0]
    metadata = model_bundle["metadata"]
    ml_weight = float(metadata["ml_probability_weight"])
    poisson_weight = float(metadata["poisson_probability_weight"])
    ensemble_probs = _normalise_probabilities(ml_weight * ml_probs + poisson_weight * poisson_probs)

    # Match class order is Home Win, Draw, Away Win.
    probs = {
        "home_win": round(float(ensemble_probs[0]) * 100, 2),
        "draw": round(float(ensemble_probs[1]) * 100, 2),
        "away_win": round(float(ensemble_probs[2]) * 100, 2),
    }
    probs["away_win"] = round(100.0 - probs["home_win"] - probs["draw"], 2)
    outcome_key = max(probs, key=probs.get)
    outcome = {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}[outcome_key]

    return {
        "home_team": home, "away_team": away, "probabilities": probs,
        "predicted_outcome": outcome,
        "expected_goals": {"home": round(lambda_home, 2), "away": round(lambda_away, 2)},
        "team_ratings": {
            "home": {"overall": float(h["Overall_Team_Strength"]), "attack": float(h["Attacking_Strength"]), "midfield": float(h["Midfield_Strength"]), "defence": float(h["Defensive_Strength"]), "goalkeeping": float(h["Goalkeeping_Strength"]), "depth": float(h["Squad_Depth_Score"])},
            "away": {"overall": float(a["Overall_Team_Strength"]), "attack": float(a["Attacking_Strength"]), "midfield": float(a["Midfield_Strength"]), "defence": float(a["Defensive_Strength"]), "goalkeeping": float(a["Goalkeeping_Strength"]), "depth": float(a["Squad_Depth_Score"])},
        },
        "current_form": {"home": str(hf["Last_5"]), "away": str(af["Last_5"])},
        "model": metadata["ensemble_name"],
        "model_components": {
            "historical_classifier": metadata["model_name"],
            "historical_classifier_weight_pct": round(ml_weight * 100, 1),
            "current_season_poisson_weight_pct": round(poisson_weight * 100, 1),
            "home_advantage_factor": HOME_ADVANTAGE_FACTOR,
        },
        "component_probabilities": {
            "historical_classifier": {"home_win": round(float(ml_probs[0]) * 100, 2), "draw": round(float(ml_probs[1]) * 100, 2), "away_win": round(float(ml_probs[2]) * 100, 2)},
            "current_season_poisson": {"home_win": round(float(poisson_probs[0]) * 100, 2), "draw": round(float(poisson_probs[1]) * 100, 2), "away_win": round(float(poisson_probs[2]) * 100, 2)},
        },
        "training": {
            "historical_matches": metadata["training_matches"],
            "training_period": f'{metadata["training_start"]} to {metadata["training_end"]}',
            "test_period": "2024/25–2025/26", "test_metrics": metadata["test_metrics"],
        },
        "model_disclaimer": metadata["disclaimer"],
    }
