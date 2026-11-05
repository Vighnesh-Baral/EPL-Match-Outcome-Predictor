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

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="EPL 2026/27 Predictor", version="2.3.0", description="PitchIQ: recency-weighted historical ML, current-season team ratings, Poisson probabilities and H2H context.")
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
            "training_status": f"Trained on {d['model_bundle']['metadata']['training_matches']} historical match records with recency weighting; 2026/27 results are added when a match log is supplied.",
            "training_period": f"{d['model_bundle']['metadata']['training_start']} to {d['model_bundle']['metadata']['training_end']}",
            "test_metrics": d["model_bundle"]["metadata"]["test_metrics"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
