"""OpenFootball (public domain, CC0) Premier League season-file parser.

Source: https://github.com/openfootball/england  (files: ``<season>/1-premierleague.txt``)

The parser is intentionally strict: a line that looks like a fixture but cannot be
parsed raises, rather than being silently skipped.  A fixture with no score yet is
returned with ``FTHG``/``FTAG`` = NaN (it is *not* imputed).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, standardize_team_name

OPENFOOTBALL_DIR = RAW_DATA_DIR / "openfootball"
RAW_URL = "https://raw.githubusercontent.com/openfootball/england/master/{season}/1-premierleague.txt"

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_MD_RE = re.compile(r"^[▪\s]*(?:Matchday|Regular Season\s*-)\s*(\d+)\s*$")
_DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_FIXTURE_RE = re.compile(
    r"^\s+(?:(?P<time>\d{1,2}:\d{2})\s+)?(?P<home>.+?)\s+v\s+(?P<away>.+?)"
    r"(?:\s+(?P<hg>\d+)-(?P<ag>\d+)(?:\s+\((?P<hth>\d+)-(?P<hta>\d+)\))?)?\s*$"
)
# Layout B/C: "[time]  Home   2-1 (1-0)   Away" (score sits between two-or-more-space gaps).
_FIXTURE_MID_RE = re.compile(
    r"^\s+(?:(?P<time>\d{1,2}:\d{2})\s+)?(?P<home>.+?)\s{2,}(?P<hg>\d+)-(?P<ag>\d+)"
    r"(?:\s+\((?P<hth>\d+)-(?P<hta>\d+)\))?\s{2,}(?P<away>.+?)\s*$"
)
_SEASON_HEADER = re.compile(r"Premier League\s+(\d{4})/(\d{2,4})")


def _clean_team(name: str) -> str:
    name = name.strip()
    for suffix in (" FC", " AFC"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("AFC "):
        name = name[4:]
    return standardize_team_name(name.strip())


def parse_openfootball_text(text: str, season: str | None = None) -> pd.DataFrame:
    """Parse one OpenFootball Premier League text file into a fixture table."""
    lines = text.splitlines()
    start_year = None
    if season:
        start_year = int(season[:4])
    else:
        for line in lines[:5]:
            m = _SEASON_HEADER.search(line)
            if m:
                start_year = int(m.group(1))
                break
    if start_year is None:
        raise ValueError("Cannot determine the season start year from the OpenFootball file.")
    season_label = f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"

    rows, matchday, current_date = [], None, None
    year, previous_month = start_year, None
    for number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#") or line.startswith("="):
            continue
        md = _MD_RE.match(line)
        if md:
            matchday = int(md.group(1))
            continue
        dm = _DATE_RE.match(line)
        if dm:
            month = _MONTHS[dm.group(1)]
            day = int(dm.group(2))
            if dm.group(3):
                year = int(dm.group(3))
            elif previous_month is not None and month < previous_month:
                year += 1  # calendar rolled over (Dec -> Jan); robust to the Jun/Jul 2020 restart
            previous_month = month
            current_date = date(year, month, day)
            continue
        if line.strip().startswith("(") or (line.strip() and rows and not re.search(r"\d-\d| v ", line)):
            continue  # goal-scorer continuation lines (layout C)
        fm = _FIXTURE_MID_RE.match(line) or (_FIXTURE_RE.match(line) if " v " in line else None)
        if fm is not None or " v " in line or _FIXTURE_MID_RE.search(line):
            if not fm or matchday is None or current_date is None:
                raise ValueError(f"Unparseable OpenFootball fixture line {number}: {line!r}")
            hg, ag = fm.group("hg"), fm.group("ag")
            rows.append({
                "Season": season_label, "Matchday": matchday, "Date": pd.Timestamp(current_date),
                "KickOff": fm.group("time"),   # carries forward on the source side only when printed
                "HomeTeam": _clean_team(fm.group("home")), "AwayTeam": _clean_team(fm.group("away")),
                "FTHG": float(hg) if hg is not None else np.nan,
                "FTAG": float(ag) if ag is not None else np.nan,
                "HTHG": float(fm.group("hth")) if fm.group("hth") is not None else np.nan,
                "HTAG": float(fm.group("hta")) if fm.group("hta") is not None else np.nan,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("OpenFootball file contained no fixtures.")
    # Strictness guard: the header states the fixture count; skipped lines would break it.
    declared = re.search(r"#\s*Matches\s+(\d+)", text)
    if declared and int(declared.group(1)) != len(frame):
        raise ValueError(f"Parsed {len(frame)} fixtures but the file header declares {declared.group(1)}.")
    # Kick-off time is printed once per time-block; carry it forward within a date.
    frame["KickOff"] = frame.groupby("Date")["KickOff"].ffill()
    played = frame["FTHG"].notna() & frame["FTAG"].notna()
    frame["FTR"] = np.where(~played, None, np.where(frame["FTHG"] > frame["FTAG"], "H",
                                            np.where(frame["FTHG"] < frame["FTAG"], "A", "D")))
    frame["Played"] = played
    return frame


def load_openfootball_season(season: str) -> pd.DataFrame:
    """Load a cached OpenFootball season file (``season`` like ``2026-27``)."""
    path = OPENFOOTBALL_DIR / f"{season}.txt"
    if not path.exists():
        raise FileNotFoundError(f"OpenFootball cache missing for {season}: {path}. Run `python -m src.ingest openfootball`.")
    return parse_openfootball_text(path.read_text(encoding="utf-8"), season=season)


def download_openfootball_season(season: str, session=None, timeout: int = 30) -> Path:
    """Download a season file into the local cache (network required)."""
    import requests
    client = session or requests
    response = client.get(RAW_URL.format(season=season), timeout=timeout)
    response.raise_for_status()
    OPENFOOTBALL_DIR.mkdir(parents=True, exist_ok=True)
    path = OPENFOOTBALL_DIR / f"{season}.txt"
    path.write_text(response.text, encoding="utf-8")
    return path
