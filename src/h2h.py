"""Historical head-to-head summaries used as context and model features."""
from __future__ import annotations
import pandas as pd
from src.config import H2H_DEFAULT_LIMIT, standardize_team_name
from src.data_loader import load_historical_matches


def get_head_to_head(home_team: str, away_team: str, matches: pd.DataFrame | None = None, limit: int = H2H_DEFAULT_LIMIT) -> dict:
    home = standardize_team_name(home_team)
    away = standardize_team_name(away_team)
    if home == away:
        raise ValueError("Select two different clubs for H2H.")
    if limit < 1 or limit > 30:
        raise ValueError("H2H limit must be between 1 and 30.")
    data = matches if matches is not None else load_historical_matches()
    mask = ((data["HomeTeam"] == home) & (data["AwayTeam"] == away)) | ((data["HomeTeam"] == away) & (data["AwayTeam"] == home))
    pair = data.loc[mask].sort_values("Date", ascending=False).head(limit).copy()
    records = []
    home_wins = draws = away_wins = 0
    for _, row in pair.iterrows():
        h_score, a_score = int(row["FTHG"]), int(row["FTAG"])
        if row["HomeTeam"] == home:
            selected_home_goals, selected_away_goals = h_score, a_score
            result = "W" if h_score > a_score else "D" if h_score == a_score else "L"
            venue = f"{home} (H)"
        else:
            selected_home_goals, selected_away_goals = a_score, h_score
            result = "W" if a_score > h_score else "D" if a_score == h_score else "L"
            venue = f"{home} (A)"
        if result == "W": home_wins += 1
        elif result == "D": draws += 1
        else: away_wins += 1
        records.append({
            "date": row["Date"].strftime("%Y-%m-%d"), "season": str(row["Season"]),
            "home_team": row["HomeTeam"], "away_team": row["AwayTeam"],
            "score": f"{h_score}–{a_score}", "home_team_perspective_result": result,
            "selected_home_score": selected_home_goals, "selected_away_score": selected_away_goals,
            "venue": venue,
        })
    n = len(records)
    return {
        "home_team": home, "away_team": away, "matches_found": n, "matches_requested": limit,
        "home_team_wins": home_wins, "draws": draws, "away_team_wins": away_wins,
        "note": "H2H is shown separately and also enters the historical classifier as a limited pre-match feature. It is not treated as a standalone prediction.",
        "matches": records,
    }
