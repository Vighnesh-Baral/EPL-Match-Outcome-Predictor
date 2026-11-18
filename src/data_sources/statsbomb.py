"""StatsBomb Open Data -> per-match xG table (Premier League 2015/16 only).

Source : https://github.com/statsbomb/open-data   (competition 2, season 27)
Licence: StatsBomb Open Data User Agreement -- attribution to StatsBomb is REQUIRED
         wherever results derived from this data are published.

Coverage reality check: the free release contains Premier League events for 2015/16 and
2003/04 only, so real xG exists for 1 of the 11 modelled seasons.  It therefore cannot be a
model feature; it is used to validate the shots-on-target features used by the model.
Shootout shots (period 5) are excluded; own-goal events carry no xG and are not counted.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, standardize_team_name

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID, SEASON_ID = 2, 27
OUTPUT = PROCESSED_DATA_DIR / "statsbomb_pl_2015_16_xg.csv"
ATTRIBUTION = "Data provided by StatsBomb (https://github.com/statsbomb/open-data)."


def _get_json(session, url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def aggregate_events(events: list[dict], home_team: str, away_team: str) -> dict[str, float]:
    """Sum shot xG / count shots per side from one match's event list (pure function)."""
    out = {"home_xg": 0.0, "away_xg": 0.0, "home_shots": 0, "away_shots": 0}
    for event in events:
        if event.get("type", {}).get("name") != "Shot" or event.get("period") == 5:
            continue
        side = "home" if event["team"]["name"] == home_team else "away" if event["team"]["name"] == away_team else None
        if side is None:
            continue
        out[f"{side}_xg"] += float(event["shot"].get("statsbomb_xg") or 0.0)
        out[f"{side}_shots"] += 1
    return out


def build_pl_2015_16_xg(session=None, workers: int = 8, limit: int | None = None) -> pd.DataFrame:
    import requests
    session = session or requests.Session()
    matches = _get_json(session, f"{BASE}/matches/{COMPETITION_ID}/{SEASON_ID}.json")[: limit or None]

    def one(match: dict) -> dict:
        home, away = match["home_team"]["home_team_name"], match["away_team"]["away_team_name"]
        events = _get_json(session, f"{BASE}/events/{match['match_id']}.json")
        return {"match_id": match["match_id"], "Date": match["match_date"],
                "HomeTeam": standardize_team_name(home), "AwayTeam": standardize_team_name(away),
                "FTHG": match["home_score"], "FTAG": match["away_score"], **aggregate_events(events, home, away)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, matches))
    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def save(frame: pd.DataFrame, path: Path = OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
