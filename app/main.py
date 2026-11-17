from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data_loader import load_current_players
from src.team_strength import compute_team_strength
from src.h2h import get_head_to_head
from src.predict import predict_match
from src.historical_model import load_or_train_model, load_training_matches
from src.season_simulator import get_matchday_predictions, simulate_final_table, load_remaining_fixtures, build_fixture_prediction_map

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="PitchIQ EPL Predictor", version="2.4.0", description="PitchIQ: validated historical ML, calibrated match probabilities, matchday predictions and Monte Carlo season table forecasts.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=STATIC_DIR / 'assets'), name="assets")


@lru_cache(maxsize=1)
def get_data() -> dict[str, Any]:
    strength, form, players = compute_team_strength()
    matches = load_training_matches()
    model_bundle = load_or_train_model()
    current_players = load_current_players()
    snapshot = None
    if "extracted_at" in current_players.columns and current_players["extracted_at"].notna().any():
        snapshot = str(current_players["extracted_at"].dropna().iloc[0])
    return {"strength": strength, "form": form, "players": players, "matches": matches,
            "snapshot": snapshot, "historical_match_count": int(len(matches)), "model_bundle": model_bundle}


class PredictionRequest(BaseModel):
    home_team: str = Field(min_length=2, max_length=80)
    away_team: str = Field(min_length=2, max_length=80)


def clean_record(record: dict) -> dict:
    clean = {}
    for k, v in record.items():
        if pd.isna(v): clean[k] = None
        elif hasattr(v, "item"): clean[k] = v.item()
        elif hasattr(v, "isoformat"): clean[k] = v.isoformat()
        else: clean[k] = v
    return clean


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/matchday", include_in_schema=False)
def matchday_page():
    return FileResponse(STATIC_DIR / "matchday.html")


@app.get("/table", include_in_schema=False)
def table_page():
    return FileResponse(STATIC_DIR / "table.html")


@app.get("/api/health")
def health():
    try:
        d = get_data()
        return {"status": "ok", "clubs": int(len(d["strength"])), "players": int(len(d["players"])),
                "h2h_matches_available": d["historical_match_count"], "snapshot": d["snapshot"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/teams")
def teams():
    try:
        d = get_data()
        records = d["strength"].to_dict(orient="records")
        return {"teams": [clean_record(r) for r in records], "season": "2026/27"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/teams/{team_name}")
def team_detail(team_name: str):
    try:
        d = get_data()
        row = d["strength"][d["strength"]["Squad"] == team_name]
        if row.empty:
            from src.config import standardize_team_name
            name = standardize_team_name(team_name)
            row = d["strength"][d["strength"]["Squad"] == name]
        if row.empty:
            raise HTTPException(status_code=404, detail="Club not found in 2026/27 data.")
        roster = d["players"][d["players"]["Club"] == row.iloc[0]["Squad"]]
        roster_cols = [c for c in ["Player_ID", "player_name", "Position", "total_points", "FPL_Event_Points", "FPL_Contribution_Rating", "now_cost"] if c in roster.columns]
        return {"team": clean_record(row.iloc[0].to_dict()), "players": [clean_record(r) for r in roster[roster_cols].to_dict(orient="records")]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/h2h")
def h2h(home_team: str = Query(...), away_team: str = Query(...), limit: int = Query(10, ge=1, le=30)):
    try:
        return get_head_to_head(home_team, away_team, get_data()["matches"], limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        d = get_data()
        return predict_match(request.home_team, request.away_team, d["strength"], d["form"], d["matches"], d["model_bundle"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/overview")
def overview():
    try:
        d = get_data()
        strength = d["strength"]
        players = d["players"]
        snapshot = d["snapshot"]
        return {
            "season": "2026/27", "clubs": int(strength["Squad"].nunique()),
            "players": int(players["Player_ID"].nunique()),
            "historical_h2h_matches": d["historical_match_count"],
            "snapshot": snapshot,
            "highest_rated_team": str(strength.iloc[0]["Squad"]),
            "top_rating": float(strength.iloc[0]["Overall_Team_Strength"]),
            "model_type": d["model_bundle"]["metadata"]["ensemble_name"],
            "training_status": f"Trained on {d['model_bundle']['metadata']['training_matches']} complete historical matches. The partial 2026/27 results log is excluded from supervised fitting and retained for H2H context.",
            "training_matches": int(d["model_bundle"]["metadata"]["training_matches"]),
            "training_period": f"{d['model_bundle']['metadata']['training_start']} to {d['model_bundle']['metadata']['training_end']}",
            "test_period": "2024/25–2025/26",
            "test_metrics": d["model_bundle"]["metadata"]["test_metrics"],
            "model_metadata": {k: d["model_bundle"]["metadata"].get(k) for k in ["version", "model_name", "ensemble_name", "ml_probability_weight", "poisson_probability_weight", "home_advantage_factor", "validation_period", "test_period", "test_matches"]},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def get_remaining_fixture_predictions() -> dict[tuple[str, str], dict]:
    """Cache all 280 season forecasts for repeated Monte Carlo run counts."""
    d = get_data()
    return build_fixture_prediction_map(d["strength"], d["form"], d["matches"], d["model_bundle"])


@app.get("/api/matchdays")
def matchdays():
    try:
        fixtures = load_remaining_fixtures()
        values = sorted(int(value) for value in fixtures["Matchday"].unique())
        return {"season": "2026/27", "matchdays": values, "fixture_count": int(len(fixtures))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/matchday/{matchday}")
def matchday_prediction(matchday: int):
    try:
        d = get_data()
        return get_matchday_predictions(matchday, d["strength"], d["form"], d["matches"], d["model_bundle"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/table-prediction")
def table_prediction(simulations: int = Query(10_000, ge=1_000, le=50_000)):
    try:
        d = get_data()
        forecasts = get_remaining_fixture_predictions()
        return simulate_final_table(d["strength"], d["form"], d["matches"], d["model_bundle"],
                                    simulations=simulations, fixture_predictions=forecasts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
