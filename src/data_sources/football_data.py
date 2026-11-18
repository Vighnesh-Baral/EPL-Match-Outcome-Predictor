"""Football-Data.co.uk Premier League results (free; attribution requested).

Source : https://www.football-data.co.uk/englandm.php   (files ``mmz4281/<yyzz>/E0.csv``)
Notes  : results, shots, corners, cards and bookmaker odds.  Odds are stored in a *separate*
         file and are used only as an external benchmark -- never as a model feature, because
         they are not available for far-future fixtures and would make the model a
         re-statement of the market.
Network: this module needs outbound access to www.football-data.co.uk.  The test-suite
         exercises it with a mocked session; it was NOT run against the live site from the
         build sandbox (the host was not reachable there).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_MATCHES_DIR

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
RESULT_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
                  "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
ODDS_DIR = RAW_DATA_DIR / "odds"
ODDS_PREFIXES = ("B365", "BW", "IW", "PS", "WH", "VC", "Max", "Avg")


def season_code(season: str) -> str:
    """'2025-26' -> '2526'."""
    start = int(season[:4])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def parse_csv(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), encoding_errors="replace")
    frame = frame.dropna(how="all")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Football-Data file is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    if frame["FTR"].isin(["H", "D", "A"]).mean() < 1.0:
        raise ValueError("Unexpected FTR values in Football-Data file.")
    return frame


def split_results_and_odds(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = frame[[c for c in RESULT_COLUMNS if c in frame.columns]].copy()
    odds_cols = [c for c in frame.columns if c.startswith(ODDS_PREFIXES) and c not in results.columns]
    odds = frame[["Date", "HomeTeam", "AwayTeam"] + odds_cols].copy()
    return results, odds


def download_season(season: str, session=None, results_dir: Path = RAW_MATCHES_DIR, odds_dir: Path = ODDS_DIR,
                    timeout: int = 30) -> dict[str, Path]:
    import requests
    client = session or requests
    response = client.get(URL.format(code=season_code(season)), timeout=timeout,
                          headers={"User-Agent": "PitchIQ-ingest/2.5 (research)"})
    response.raise_for_status()
    results, odds = split_results_and_odds(parse_csv(response.text))
    results_dir.mkdir(parents=True, exist_ok=True)
    odds_dir.mkdir(parents=True, exist_ok=True)
    code = season_code(season)
    paths = {"results": results_dir / f"season-{code}.csv", "odds": odds_dir / f"season-{code}_odds.csv"}
    results.to_csv(paths["results"], index=False)
    odds.to_csv(paths["odds"], index=False)
    return paths
