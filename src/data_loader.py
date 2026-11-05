"""Validated ingestion for current-season model inputs and historical fixtures."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import RAW_CURRENT_DIR, RAW_MATCHES_DIR, TEAMS_2026_27, standardize_team_name


class DataValidationError(ValueError):
    """Raised when an input dataset cannot safely support the prototype."""


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kwargs)


def _validate_team_rows(df: pd.DataFrame, label: str) -> None:
    if "Squad" not in df.columns:
        raise DataValidationError(f"{label} dataset has no Squad column.")
    if df["Squad"].isna().any():
        raise DataValidationError(f"{label} dataset contains blank club names.")
    dupes = df.loc[df["Squad"].duplicated(), "Squad"].tolist()
    if dupes:
        raise DataValidationError(f"{label} contains duplicate clubs: {dupes}")
    found = set(df["Squad"])
    expected = set(TEAMS_2026_27)
    if found != expected:
        raise DataValidationError(
            f"{label} club coverage mismatch. Missing={sorted(expected-found)}; unexpected={sorted(found-expected)}"
        )


def load_current_team_data() -> pd.DataFrame:
    """Merge 2026/27 standings and FBref squad-level statistics for all 20 clubs."""
    table_path = RAW_CURRENT_DIR / "premier_league_2026-27_table.csv"
    table = _read_csv(table_path)
    if "Squad" not in table.columns:
        raise DataValidationError("League table must contain a Squad column.")
    table["Squad"] = table["Squad"].map(standardize_team_name)
    _validate_team_rows(table, "League table")

    sources = {
        "std": "squad_standard_statistics_2026_27.csv",
        "shoot": "squad_shooting_statistics_2026_27-v3.csv",
        "gk": "squad_goalkeeping_statistics_2026_27.csv",
        "time": "squad_playing_time_statistics_2026_27-v4.csv",
        "misc": "squad_miscellaneous_statistics_2026_27.csv",
    }
    merged = table.copy()
    for prefix, filename in sources.items():
        # FBref exports have a top-level category row; the second row is the real header.
        part = _read_csv(RAW_CURRENT_DIR / filename, header=1)
        if "Squad" not in part.columns:
            raise DataValidationError(f"{filename} was not parsed with the expected FBref header row.")
        part["Squad"] = part["Squad"].map(standardize_team_name)
        _validate_team_rows(part, filename)
        part = part.rename(columns={c: f"{prefix}_{c}" for c in part.columns if c != "Squad"})
        merged = merged.merge(part, on="Squad", how="left", validate="one_to_one")

    _validate_team_rows(merged, "Merged 2026/27 team data")
    if len(merged) != 20:
        raise DataValidationError(f"Expected 20 clubs after merges, found {len(merged)}.")

    # The fields used by ratings and predictions must be numeric and present.
    required_numeric = [
        "MP", "GF", "GA", "Pts", "std_Poss", "shoot_SoT/90", "shoot_Sh/90",
        "shoot_G/Sh", "gk_Save%", "gk_GA90", "gk_CS%", "gk_SoTA",
        "misc_TklW", "misc_Int", "std_Ast", "time_# Pl",
    ]
    missing = [c for c in required_numeric if c not in merged.columns]
    if missing:
        raise DataValidationError(f"Missing required current-season metrics: {missing}")
    for col in required_numeric:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if merged[["MP", "GF", "GA"]].isna().any().any() or (merged["MP"] <= 0).any():
        raise DataValidationError("League table contains invalid match or goal totals.")
    return merged.sort_values("Squad").reset_index(drop=True)


def load_current_players() -> pd.DataFrame:
    """Join current FPL roster and current FPL stats by stable player ID."""
    metadata = _read_csv(RAW_CURRENT_DIR / "fpl_players-selected-columns.csv")
    stats = _read_csv(RAW_CURRENT_DIR / "fpl_player_statistics-selected-columns.csv")
    if "id" not in metadata or "player_id" not in stats:
        raise DataValidationError("FPL files must contain id and player_id keys.")
    if metadata["id"].duplicated().any():
        raise DataValidationError("FPL metadata contains duplicate player IDs.")
    if stats["player_id"].duplicated().any():
        raise DataValidationError("FPL statistics contain duplicate player IDs.")
    if set(metadata["id"]) != set(stats["player_id"]):
        raise DataValidationError("FPL metadata and statistics do not have identical player ID coverage.")

    players = metadata.merge(
        stats[[c for c in ["player_id", "total_points", "event_points", "now_cost", "club_name", "position_name", "extracted_at"] if c in stats.columns]],
        left_on="id", right_on="player_id", how="left", validate="one_to_one",
        suffixes=("_meta", "_stats"),
    )
    players["team_name"] = players["team"].map(standardize_team_name)
    players["position_code"] = players["position"].astype(str).str.upper()
    players["position_group"] = players["position_code"].map({"GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"})
    players["player_name"] = players["name"].fillna("Unknown player")
    players["total_points"] = pd.to_numeric(players["total_points"], errors="coerce").fillna(0.0)
    players["event_points"] = pd.to_numeric(players["event_points"], errors="coerce").fillna(0.0)
    players["now_cost"] = pd.to_numeric(players["now_cost_stats"].combine_first(players["now_cost_meta"]), errors="coerce")

    if players["team_name"].isna().any() or not set(players["team_name"]).issubset(set(TEAMS_2026_27)):
        unknown = sorted(set(players["team_name"]) - set(TEAMS_2026_27))
        raise DataValidationError(f"FPL contains clubs outside the current 20: {unknown}")
    if players["position_group"].isna().any():
        bad = sorted(players.loc[players["position_group"].isna(), "position_code"].unique())
        raise DataValidationError(f"Unknown FPL position codes: {bad}")

    if "club_name" in players.columns:
        stats_teams = players["club_name"].map(standardize_team_name)
        mismatch = stats_teams.notna() & (stats_teams != players["team_name"])
        if mismatch.any():
            raise DataValidationError(f"FPL metadata/statistics club mismatch for {int(mismatch.sum())} players.")

    if players["id"].nunique() != len(players):
        raise DataValidationError("Player ID uniqueness failed after FPL merge.")
    if set(players["team_name"].unique()) != set(TEAMS_2026_27):
        raise DataValidationError("FPL roster does not cover exactly the configured 20 clubs.")
    return players.sort_values(["team_name", "position_group", "player_name"]).reset_index(drop=True)


def load_historical_matches() -> pd.DataFrame:
    """Load 2015/16–2025/26 results for historical modelling and H2H context."""
    files = sorted(RAW_MATCHES_DIR.glob("season-*.csv"))
    if not files:
        raise FileNotFoundError(f"No historical result CSVs found in {RAW_MATCHES_DIR}")
    frames = []
    for path in files:
        df = _read_csv(path)
        required = {"Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
        missing = required - set(df.columns)
        if missing:
            raise DataValidationError(f"{path.name} is missing match-result columns: {sorted(missing)}")
        stem = path.stem.replace("season-", "")
        if not re.fullmatch(r"\d{4}", stem):
            raise DataValidationError(f"Unexpected historical filename: {path.name}")
        start_year = 2000 + int(stem[:2])
        season = f"{start_year}-{int(stem[2:]):02d}"
        df["Season"] = season
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["HomeTeam"] = df["HomeTeam"].map(standardize_team_name)
        df["AwayTeam"] = df["AwayTeam"].map(standardize_team_name)
        df["FTR"] = df["FTR"].astype(str).str.upper()
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df = df[df["FTR"].isin(["H", "D", "A"])].copy()
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    duplicate_mask = matches.duplicated(["Date", "HomeTeam", "AwayTeam"], keep=False)
    if duplicate_mask.any():
        raise DataValidationError(f"Historical H2H data contains {int(duplicate_mask.sum())} duplicate fixtures.")
    return matches.sort_values("Date").reset_index(drop=True)



def load_current_season_results() -> pd.DataFrame:
    """Load verified completed 2026/27 fixtures when a results file is available."""
    candidates = [
        RAW_CURRENT_DIR / "match_results_2026_27.csv",
        RAW_CURRENT_DIR / "season-2627.csv",
        RAW_CURRENT_DIR / "matches_2026_27.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "Season"])
    df = _read_csv(path)
    required = {"Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{path.name} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].map(standardize_team_name)
    df["AwayTeam"] = df["AwayTeam"].map(standardize_team_name)
    df["FTR"] = df["FTR"].astype(str).str.upper().str.strip()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df["Season"] = "2026-27"
    if df.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise DataValidationError("2026/27 results file contains duplicate fixtures.")
    return df.sort_values("Date").reset_index(drop=True)


def load_all_current_data() -> Dict[str, pd.DataFrame]:
    return {"teams": load_current_team_data(), "players": load_current_players(), "h2h_matches": load_historical_matches()}
