"""Matchday forecasts and Monte Carlo end-of-season table simulation.

Each outstanding fixture is forecast by the same production match predictor.
Season simulations draw scorelines from Poisson score distributions reweighted
so their home/draw/away marginals match the model's final probabilities.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from src.config import RAW_CURRENT_DIR, TEAMS_2026_27, standardize_team_name
from src.data_loader import _read_csv
from src.predict import predict_match

FIXTURES_PATH = RAW_CURRENT_DIR / "fixtures_matchday_11_38.csv"
STANDINGS_PATH = RAW_CURRENT_DIR / "current_standings_2026_27.csv"
SNAPSHOT_PATH = RAW_CURRENT_DIR / "snapshot_metadata_v24.json"
MAX_GOALS = 10


def load_remaining_fixtures() -> pd.DataFrame:
    if not FIXTURES_PATH.exists():
        raise FileNotFoundError(f"Remaining fixture schedule not found: {FIXTURES_PATH}")
    fixtures = _read_csv(FIXTURES_PATH)
    required = {"Matchday", "Fixture", "HomeTeam", "AwayTeam", "Status"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture schedule is missing columns: {sorted(missing)}")
    fixtures = fixtures.copy()
    fixtures["Matchday"] = pd.to_numeric(fixtures["Matchday"], errors="coerce")
    fixtures["HomeTeam"] = fixtures["HomeTeam"].map(standardize_team_name)
    fixtures["AwayTeam"] = fixtures["AwayTeam"].map(standardize_team_name)
    fixtures["Status"] = fixtures["Status"].fillna("Scheduled").astype(str).str.strip().str.title()
    if fixtures[["Matchday", "HomeTeam", "AwayTeam"]].isna().any().any():
        raise ValueError("Fixture schedule contains missing matchday or club values.")
    fixtures = fixtures[fixtures["Status"].str.lower() != "completed"].copy()
    fixtures["Matchday"] = fixtures["Matchday"].astype(int)
    if fixtures.duplicated(["Matchday", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Fixture schedule contains duplicate matchday fixtures.")
    expected = set(TEAMS_2026_27)
    found = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    if found != expected:
        raise ValueError(f"Fixture clubs do not match the 20-team league. Missing={sorted(expected-found)}; unexpected={sorted(found-expected)}")
    # A double round-robin league must have 14 remaining home and away fixtures
    # per club after ten matchdays have been played.
    home_counts = fixtures["HomeTeam"].value_counts().reindex(TEAMS_2026_27, fill_value=0)
    away_counts = fixtures["AwayTeam"].value_counts().reindex(TEAMS_2026_27, fill_value=0)
    if len(fixtures) != 280 or not (home_counts == 14).all() or not (away_counts == 14).all():
        raise ValueError("Remaining fixture schedule failed the 280-fixture / 14-home / 14-away integrity check.")
    return fixtures.sort_values(["Matchday", "Fixture"], kind="stable").reset_index(drop=True)


def load_current_standings() -> pd.DataFrame:
    if not STANDINGS_PATH.exists():
        raise FileNotFoundError(f"Current standings snapshot not found: {STANDINGS_PATH}")
    table = _read_csv(STANDINGS_PATH)
    required = {"Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Current standings snapshot is missing columns: {sorted(missing)}")
    table = table.copy()
    table["Squad"] = table["Squad"].map(standardize_team_name)
    for col in ["MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if table["Squad"].duplicated().any() or table[list(required - {"Squad"})].isna().any().any():
        raise ValueError("Current standings contains duplicate clubs or invalid numeric totals.")
    if set(table["Squad"]) != set(TEAMS_2026_27) or len(table) != 20:
        raise ValueError("Current standings must contain exactly the current 20 Premier League clubs.")
    if not (table["MP"] == 10).all():
        raise ValueError("The current standings snapshot is not consistently through Matchday 10.")
    if not np.allclose(table["W"] + table["D"] + table["L"], table["MP"]):
        raise ValueError("Current standings W/D/L totals do not equal matches played.")
    if not np.allclose(table["GF"] - table["GA"], table["GD"]):
        raise ValueError("Current standings goal difference does not match GF minus GA.")
    return table.sort_values("Squad").reset_index(drop=True)



def build_fixture_prediction_map(strength: pd.DataFrame, form: pd.DataFrame,
                                 historical_matches: pd.DataFrame, model_bundle: dict) -> dict[tuple[str, str], dict]:
    """Precompute each outstanding fixture once for repeatable season simulations."""
    fixtures = load_remaining_fixtures()
    forecasts = {}
    for row in fixtures.itertuples(index=False):
        key = (str(row.HomeTeam), str(row.AwayTeam))
        forecasts[key] = predict_match(key[0], key[1], strength, form, historical_matches, model_bundle)
    return forecasts

def _poisson_vector(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    lam = float(np.clip(lam, 0.10, 5.0))
    p = np.asarray([math.exp(-lam) * lam**goal / math.factorial(goal) for goal in range(max_goals + 1)], dtype=float)
    return p / p.sum()


def _scoreline_distribution(prediction: dict) -> np.ndarray:
    """Joint score probabilities with exact model W/D/L marginals."""
    xg = prediction["expected_goals"]
    probs = prediction["probabilities"]
    p_h = _poisson_vector(xg["home"])
    p_a = _poisson_vector(xg["away"])
    joint = np.outer(p_h, p_a)
    home_mask = np.fromfunction(lambda h, a: h > a, joint.shape, dtype=int).astype(bool)
    draw_mask = np.eye(joint.shape[0], dtype=bool)
    away_mask = np.fromfunction(lambda h, a: h < a, joint.shape, dtype=int).astype(bool)
    masks = [home_mask, draw_mask, away_mask]
    target = np.asarray([probs["home_win"], probs["draw"], probs["away_win"]], dtype=float) / 100.0
    target = np.clip(target, 1e-8, None)
    target /= target.sum()
    result = np.zeros_like(joint)
    for mask, weight in zip(masks, target):
        mass = float(joint[mask].sum())
        if mass <= 0:
            raise ValueError("Poisson scoreline distribution has an empty outcome class.")
        result[mask] = weight * joint[mask] / mass
    result /= result.sum()
    return result


def get_matchday_predictions(matchday: int, strength: pd.DataFrame, form: pd.DataFrame,
                             historical_matches: pd.DataFrame, model_bundle: dict) -> dict:
    if matchday < 11 or matchday > 38:
        raise ValueError("Matchday must be between 11 and 38.")
    fixtures = load_remaining_fixtures()
    selected = fixtures[fixtures["Matchday"] == matchday]
    if selected.empty:
        raise ValueError(f"No outstanding fixtures found for Matchday {matchday}.")
    items = []
    for row in selected.itertuples(index=False):
        prediction = predict_match(row.HomeTeam, row.AwayTeam, strength, form, historical_matches, model_bundle)
        values = sorted(prediction["probabilities"].values(), reverse=True)
        prediction_margin = round(values[0] - values[1], 2)
        items.append({
            "fixture_number": int(row.Fixture),
            "home_team": row.HomeTeam,
            "away_team": row.AwayTeam,
            "status": row.Status,
            **{k: v for k, v in prediction.get("fixture", {}).items()},
            "prediction": prediction,
            "confidence_pct": round(max(prediction["probabilities"].values()), 2),
            "probability_margin_pct": prediction_margin,
        })
    summary = {
        "home_wins": sum(x["prediction"]["predicted_outcome"] == "Home Win" for x in items),
        "draws": sum(x["prediction"]["predicted_outcome"] == "Draw" for x in items),
        "away_wins": sum(x["prediction"]["predicted_outcome"] == "Away Win" for x in items),
    }
    most_confident = max(items, key=lambda x: x["confidence_pct"])
    closest = min(items, key=lambda x: x["probability_margin_pct"])
    return {
        "season": "2026/27", "matchday": matchday, "fixtures_count": len(items),
        "summary": summary, "most_confident_fixture": {
            "home_team": most_confident["home_team"], "away_team": most_confident["away_team"],
            "outcome": most_confident["prediction"]["predicted_outcome"], "confidence_pct": most_confident["confidence_pct"]},
        "closest_fixture": {
            "home_team": closest["home_team"], "away_team": closest["away_team"],
            "outcome": closest["prediction"]["predicted_outcome"], "probability_margin_pct": closest["probability_margin_pct"]},
        "model": model_bundle["metadata"]["ensemble_name"], "fixtures": items,
    }


def simulate_final_table(strength: pd.DataFrame, form: pd.DataFrame, historical_matches: pd.DataFrame,
                         model_bundle: dict, simulations: int = 10_000, seed: int = 42,
                         fixture_predictions: dict[tuple[str, str], dict] | None = None) -> dict:
    if simulations < 1_000 or simulations > 50_000:
        raise ValueError("Simulation count must be between 1,000 and 50,000.")
    fixtures = load_remaining_fixtures()
    standings = load_current_standings()
    teams = standings["Squad"].tolist()
    team_idx = {name: idx for idx, name in enumerate(teams)}
    n_teams = len(teams)
    rng = np.random.default_rng(seed)
    # Building a live prediction scans the current club/H2H context. Reuse the
    # cached forecast map across simulation-count changes instead of rebuilding
    # 280 expensive match predictions on every request.
    if fixture_predictions is None:
        fixture_predictions = build_fixture_prediction_map(strength, form, historical_matches, model_bundle)

    points = np.tile(standings["Pts"].to_numpy(dtype=np.int32), (simulations, 1))
    wins = np.tile(standings["W"].to_numpy(dtype=np.int16), (simulations, 1))
    draws = np.tile(standings["D"].to_numpy(dtype=np.int16), (simulations, 1))
    losses = np.tile(standings["L"].to_numpy(dtype=np.int16), (simulations, 1))
    gf = np.tile(standings["GF"].to_numpy(dtype=np.int16), (simulations, 1))
    ga = np.tile(standings["GA"].to_numpy(dtype=np.int16), (simulations, 1))

    for row in fixtures.itertuples(index=False):
        home_idx, away_idx = team_idx[row.HomeTeam], team_idx[row.AwayTeam]
        forecast = fixture_predictions[(str(row.HomeTeam), str(row.AwayTeam))]
        matrix = _scoreline_distribution(forecast)
        # searchsorted over one CDF is significantly faster than weighted
        # Generator.choice for millions of small categorical draws.
        cdf = np.cumsum(matrix.ravel())
        flat_idx = np.searchsorted(cdf, rng.random(simulations), side="right")
        flat_idx = np.minimum(flat_idx, matrix.size - 1)
        home_goals = (flat_idx // matrix.shape[1]).astype(np.int16)
        away_goals = (flat_idx % matrix.shape[1]).astype(np.int16)
        gf[:, home_idx] += home_goals
        ga[:, home_idx] += away_goals
        gf[:, away_idx] += away_goals
        ga[:, away_idx] += home_goals
        home_win = home_goals > away_goals
        draw = home_goals == away_goals
        away_win = home_goals < away_goals
        points[:, home_idx] += (3 * home_win + draw).astype(np.int16)
        points[:, away_idx] += (3 * away_win + draw).astype(np.int16)
        wins[:, home_idx] += home_win.astype(np.int16)
        draws[:, home_idx] += draw.astype(np.int16)
        losses[:, home_idx] += away_win.astype(np.int16)
        wins[:, away_idx] += away_win.astype(np.int16)
        draws[:, away_idx] += draw.astype(np.int16)
        losses[:, away_idx] += home_win.astype(np.int16)

    gd = gf - ga
    # Table ranking by points, goal difference, then goals scored. A rare exact
    # three-way tie remains deterministic by club order in the source table.
    order = np.lexsort((np.broadcast_to(np.arange(n_teams), (simulations, n_teams)), gf, gd, points), axis=1)[:, ::-1]
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(simulations)[:, None], order] = np.arange(1, n_teams + 1, dtype=np.int16)[None, :]

    expected = {
        "points": points.mean(axis=0), "wins": wins.mean(axis=0), "draws": draws.mean(axis=0),
        "losses": losses.mean(axis=0), "gf": gf.mean(axis=0), "ga": ga.mean(axis=0),
        "gd": gd.mean(axis=0), "mean_position": ranks.mean(axis=0),
        "p10_position": np.percentile(ranks, 10, axis=0), "p90_position": np.percentile(ranks, 90, axis=0),
    }
    title = (ranks == 1).mean(axis=0) * 100
    top4 = (ranks <= 4).mean(axis=0) * 100
    top6 = (ranks <= 6).mean(axis=0) * 100
    relegation = (ranks >= 18).mean(axis=0) * 100
    rows = []
    for idx, team in enumerate(teams):
        rows.append({
            "team": team, "expected_points": round(float(expected["points"][idx]), 2),
            "expected_wins": round(float(expected["wins"][idx]), 2),
            "expected_draws": round(float(expected["draws"][idx]), 2),
            "expected_losses": round(float(expected["losses"][idx]), 2),
            "expected_gf": round(float(expected["gf"][idx]), 2),
            "expected_ga": round(float(expected["ga"][idx]), 2),
            "expected_gd": round(float(expected["gd"][idx]), 2),
            "average_finishing_position": round(float(expected["mean_position"][idx]), 2),
            "position_p10": round(float(expected["p10_position"][idx]), 1),
            "position_p90": round(float(expected["p90_position"][idx]), 1),
            "title_probability": round(float(title[idx]), 2),
            "top4_probability": round(float(top4[idx]), 2),
            "top6_probability": round(float(top6[idx]), 2),
            "relegation_probability": round(float(relegation[idx]), 2),
            "position_probabilities": [round(float((ranks[:, idx] == pos).mean() * 100), 2) for pos in range(1, 21)],
        })
    rows.sort(key=lambda x: (x["expected_points"], x["expected_gd"], x["expected_gf"]), reverse=True)
    for pos, row in enumerate(rows, start=1):
        row["projected_position"] = pos
        row["matches_played"] = 38

    try:
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        source_date = snapshot.get("team_statistics_snapshot")
        results_through = snapshot.get("standings_through_matchday", 10)
    except (OSError, json.JSONDecodeError):
        source_date, results_through = None, 10
    return {
        "season": "2026/27", "simulations": simulations, "seed": seed,
        "standings_through_matchday": results_through,
        "team_statistics_snapshot": source_date,
        "fixtures_simulated": int(len(fixtures)),
        "method": "Monte Carlo scoreline simulation. Each fixture uses PitchIQ's final H/D/A probabilities; conditional scorelines are sampled from the Poisson goal distributions.",
        "ranking_rule": "Points, goal difference, goals scored; exact residual ties use deterministic club ordering.",
        "limitations": [
            "Current league standings are the supplied Matchday 10 snapshot.",
            "FBref advanced team metrics and FPL player inputs remain at the 4 November 2026 snapshot, so they do not include all Matchday 10 performance.",
            "The 2026/27 completed-results file is partial and is used for head-to-head context only, not historical model fitting.",
            "Fixture dates and kick-off times are not embedded in the matchday schedule; matchday and home/away assignments are used.",
        ],
        "model": model_bundle["metadata"]["ensemble_name"],
        "table": rows,
    }
