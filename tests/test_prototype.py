import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, get_data
from src.config import SEASON_SAMPLE_WEIGHTS, TEAMS_2026_27
from src.h2h import get_head_to_head
from src.historical_model import (
    FEATURE_COLUMNS,
    build_historical_features,
    load_or_train_model,
    season_sample_weights,
)
from src.predict import predict_match
from src.team_strength import compute_team_strength


def test_current_data_integrity():
    strength, form, players = compute_team_strength()
    assert len(strength) == 20
    assert set(strength["Squad"]) == set(TEAMS_2026_27)
    assert len(players) == 670
    assert players["Player_ID"].is_unique
    assert set(players["Club"]) == set(TEAMS_2026_27)
    assert strength["Overall_Team_Strength"].between(0, 100).all()
    assert strength["Squad_Selection_Complete"].all()
    assert len(form) == 20
    assert {"W", "D", "L"}.issubset(strength.columns)


def test_recency_weights_follow_user_priority():
    seasons = pd.Series(["2015-16", "2019-20", "2021-22", "2023-24", "2024-25", "2025-26"])
    weights = season_sample_weights(seasons)
    assert np.all(np.diff(weights) > 0)
    assert weights[-1] == SEASON_SAMPLE_WEIGHTS["2025-26"]


def test_same_date_matches_do_not_leak_into_each_other():
    matches = pd.DataFrame([
        {"Date": "2024-08-17", "Season": "2024-25", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 0, "FTR": "H"},
        {"Date": "2024-08-17", "Season": "2024-25", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": "2024-08-24", "Season": "2024-25", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 0, "FTR": "H"},
    ])
    features = build_historical_features(matches)
    assert features.loc[0, "home_form5_ppm"] == 1.35
    assert features.loc[1, "home_form5_ppm"] == 1.35
    # Only the previous date's result is visible before the third fixture.
    assert features.loc[2, "home_form5_ppm"] == 3.0
    assert list(features[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_historical_feature_set_is_complete_and_finite():
    data = get_data()
    from src.historical_model import load_training_matches
    features = build_historical_features(load_training_matches())
    assert len(features) == 4181
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
    assert data["model_bundle"]["metadata"]["training_matches"] == 4181


def test_probabilities_sum_to_100_and_have_both_model_branches():
    strength, form, _ = compute_team_strength()
    matches = get_data()["matches"]
    bundle = load_or_train_model()
    result = predict_match("Arsenal", "Liverpool", strength, form, matches, bundle)
    p = result["probabilities"]
    assert round(sum(p.values()), 2) == 100.00
    assert all(0 <= value <= 100 for value in p.values())
    assert result["model_components"]["historical_classifier_weight_pct"] > 0
    assert result["model_components"]["current_season_poisson_weight_pct"] >= 0
    assert result["expected_goals"]["home"] > 0
    assert result["expected_goals"]["away"] > 0


def test_h2h_returns_only_pair_meetings_and_is_model_feature():
    data = get_data()
    result = get_head_to_head("Arsenal", "Liverpool", data["matches"], limit=10)
    assert result["matches_found"] <= 10
    assert result["matches_found"] == len(result["matches"])
    assert result["home_team_wins"] + result["draws"] + result["away_team_wins"] == result["matches_found"]
    assert "historical classifier" in result["note"]


def test_api_health_and_prediction():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["clubs"] == 20
    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["model_type"]
    prediction = client.post("/api/predict", json={"home_team": "Arsenal", "away_team": "Liverpool"})
    assert prediction.status_code == 200
    assert round(sum(prediction.json()["probabilities"].values()), 2) == 100.00


def test_same_team_prediction_rejected():
    strength, form, _ = compute_team_strength()
    try:
        predict_match("Arsenal", "Arsenal", strength, form, get_data()["matches"], load_or_train_model())
    except ValueError:
        return
    raise AssertionError("Same-team fixture should be rejected")


def test_2026_27_brighton_arsenal_result_is_in_h2h():
    from src.historical_model import load_training_matches
    matches = load_training_matches()
    result = get_head_to_head("Brighton", "Arsenal", matches, limit=10)
    assert any(row["date"] == "2026-09-19" and row["score"] == "3–0" for row in result["matches"])


def test_previous_season_prior_is_visible_and_bounded():
    strength, _, _ = compute_team_strength()
    assert strength["Prior_Season_Weight"].between(0, 0.15).all()
    assert strength["Prior_Season_Strength"].between(0, 100).all()
    assert (strength["Prior_Season_Weight"] > 0).all()
