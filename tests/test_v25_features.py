"""Tests for PitchIQ Version 2.5 features, Dixon-Coles model, and Ensemble."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, get_data
from src.config import TEAMS_2026_27
from src.dixon_coles import dixon_coles_outcome_probs, tau_adjustment
from src.historical_model import FEATURE_COLUMNS, build_historical_features, load_or_train_model, load_training_matches
from src.predict import predict_match
from src.team_strength import compute_team_strength


def test_dixon_coles_math_properties():
    # Sum of probabilities must be exactly 1.0
    h, d, a = dixon_coles_outcome_probs(1.5, 1.2, rho=-0.12)
    assert abs((h + d + a) - 1.0) < 1e-6
    assert 0 < h < 1 and 0 < d < 1 and 0 < a < 1

    # Low-scoring tau adjustments
    tau_00 = tau_adjustment(0, 0, 1.5, 1.2, -0.12)
    assert tau_00 > 1.0  # Boosts 0-0 draws when rho < 0
    tau_11 = tau_adjustment(1, 1, 1.5, 1.2, -0.12)
    assert tau_11 > 1.0  # Boosts 1-1 draws when rho < 0


def test_v25_feature_set_columns_and_invariants():
    matches = pd.DataFrame([
        {"Date": "2024-08-17", "Season": "2024-25", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 0, "FTR": "H", "HST": 5, "AST": 2},
        {"Date": "2024-08-17", "Season": "2024-25", "HomeTeam": "Liverpool", "AwayTeam": "Everton", "FTHG": 1, "FTAG": 1, "FTR": "D", "HST": 4, "AST": 3},
        {"Date": "2024-08-24", "Season": "2024-25", "HomeTeam": "Arsenal", "AwayTeam": "Liverpool", "FTHG": 3, "FTAG": 1, "FTR": "H", "HST": 6, "AST": 4},
    ])
    features = build_historical_features(matches)
    assert set(FEATURE_COLUMNS).issubset(set(features.columns))
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()

    # Pre-match Elo is assigned before post-match update
    assert features.loc[0, "home_elo"] == 1500.0
    assert features.loc[1, "home_elo"] == 1500.0
    # Arsenal won on 2024-08-17, so their pre-match Elo on 2024-08-24 must be higher than 1500
    assert features.loc[2, "home_elo"] > 1500.0


def test_v25_model_bundle_metadata():
    bundle = load_or_train_model()
    metadata = bundle["metadata"]
    assert metadata["version"] == "2.5.0"
    assert "Ensemble" in metadata["ensemble_name"]
    assert metadata["training_matches"] == 4180
    assert len(metadata["feature_columns"]) == len(FEATURE_COLUMNS)
    assert "dixon_coles_rho" in metadata


def test_v25_prediction_output_integrity():
    strength, form, _ = compute_team_strength()
    matches = get_data()["matches"]
    bundle = load_or_train_model()
    result = predict_match("Manchester City", "Arsenal", strength, form, matches, bundle)
    probs = result["probabilities"]
    assert round(sum(probs.values()), 2) == 100.00
    assert all(0 <= p <= 100 for p in probs.values())
    assert result["expected_goals"]["home"] > 0
    assert result["expected_goals"]["away"] > 0
    assert "dixon_coles_rho" in result["model_components"]

