from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "Raw"
RAW_CURRENT_DIR = RAW_DATA_DIR / "2026_27"
RAW_MATCHES_DIR = RAW_DATA_DIR / "match_results"
PROCESSED_DATA_DIR = DATA_DIR / "Processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "Models"

for _path in (PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR):
    _path.mkdir(parents=True, exist_ok=True)

TEAMS_2026_27 = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Coventry City", "Crystal Palace", "Everton", "Fulham",
    "Hull City", "Ipswich Town", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
]

TEAM_ALIASES = {
    "Arsenal FC": "Arsenal",
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton &amp; Hove Albion": "Brighton",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester City": "Leicester",
    "Man City": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Manchester United FC": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolves",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

POSITION_MAP = {"GKP": "GK", "G": "GK", "DEF": "DEF", "D": "DEF", "MID": "MID", "M": "MID", "FWD": "FWD", "F": "FWD"}

# Transparent prototype assumptions. These are not fitted model parameters.
H2H_DEFAULT_LIMIT = 10
HOME_ADVANTAGE_FACTOR = 1.06
POISSON_MAX_GOALS = 12

# Recency hierarchy agreed for the historical training data. These are initial
# sample-weight multipliers, not claims that these values are optimal. The
# chronological validation step in src.historical_model tests their usefulness.
SEASON_SAMPLE_WEIGHTS = {
    "2015-16": 0.25,
    "2016-17": 0.27,
    "2017-18": 0.29,
    "2018-19": 0.31,
    "2019-20": 0.35,
    "2020-21": 0.45,
    "2021-22": 0.50,
    "2022-23": 0.58,
    "2023-24": 0.72,
    "2024-25": 0.86,
    "2025-26": 1.00,
    "2026-27": 1.15,
}


def standardize_team_name(name: object) -> str:
    if not isinstance(name, str):
        return str(name)
    value = " ".join(name.replace("&amp;", "&").strip().split())
    return TEAM_ALIASES.get(value, value)
