"""Official Fantasy Premier League API (free, unauthenticated) and its community archive.

Live API : https://fantasy.premierleague.com/api/bootstrap-static/   (current squad data)
Archive  : https://github.com/vaastav/Fantasy-Premier-League  (MIT) -- historical per-gameweek
           player data, the only free route to *backtestable* player-strength features.

Status in v2.5: the live snapshot parser is implemented and tested with a mocked payload
(the host was not reachable from the build sandbox).  The historical archive is NOT yet
wired into the model: no player-strength feature is claimed or used in v2.5 predictions.
Current-season squad ratings continue to be displayed (Team/Player Intelligence) but are not
injected into probabilities, because they cannot be validated on past matches.
"""
from __future__ import annotations

import pandas as pd

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def bootstrap_to_players(payload: dict) -> pd.DataFrame:
    """Flatten a bootstrap-static payload into one row per player (pure function)."""
    for key in ("elements", "teams"):
        if key not in payload:
            raise ValueError(f"FPL payload missing '{key}'.")
    teams = {t["id"]: t["name"] for t in payload["teams"]}
    rows = []
    for p in payload["elements"]:
        rows.append({
            "fpl_id": p["id"], "name": p.get("web_name"), "club": teams.get(p["team"]),
            "position": POSITIONS.get(p.get("element_type"), "UNK"), "minutes": p.get("minutes", 0),
            "total_points": p.get("total_points", 0), "goals": p.get("goals_scored", 0), "assists": p.get("assists", 0),
            "now_cost_m": p.get("now_cost", 0) / 10.0, "status": p.get("status"),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
        })
    return pd.DataFrame(rows)


def fetch_bootstrap(session=None, timeout: int = 30) -> dict:
    import requests
    client = session or requests
    response = client.get(BOOTSTRAP_URL, timeout=timeout, headers={"User-Agent": "PitchIQ-ingest/2.5 (research)"})
    response.raise_for_status()
    return response.json()
