"""Validate shots-on-target features against real StatsBomb xG (Premier League 2015/16 only).

Descriptive only: 380 matches in one season are too few to *model* with, so this answers
"is SoT a defensible stand-in for xG?" and nothing more.   python -m src.engine.xg_validation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.data_loader import load_historical_matches


def run() -> dict:
    sb = pd.read_csv(PROCESSED_DATA_DIR / "statsbomb_pl_2015_16_xg.csv", parse_dates=["Date"])
    fd = load_historical_matches()
    fd = fd[fd["Season"] == "2015-16"]
    m = sb.merge(fd, on=["HomeTeam", "AwayTeam"], suffixes=("_sb", "_fd"))
    out: dict = {"matched_matches": int(len(m)), "of": int(len(sb)),
                 "score_mismatches": int(((m.FTHG_sb != m.FTHG_fd) | (m.FTAG_sb != m.FTAG_fd)).sum()),
                 "date_mismatches": int((m.Date_sb != m.Date_fd).sum())}
    # team-match long format
    home = pd.DataFrame({"Date": m.Date_fd, "team": m.HomeTeam, "xg": m.home_xg, "xga": m.away_xg, "gf": m.FTHG_fd, "ga": m.FTAG_fd,
                         "sot": m.HST, "sota": m.AST, "shots": m.HS})
    away = pd.DataFrame({"Date": m.Date_fd, "team": m.AwayTeam, "xg": m.away_xg, "xga": m.home_xg, "gf": m.FTAG_fd, "ga": m.FTHG_fd,
                         "sot": m.AST, "sota": m.HST, "shots": m.AS})
    long = pd.concat([home, away]).sort_values(["team", "Date"]).reset_index(drop=True)
    out["corr_match_xg_vs_goals"] = float(np.corrcoef(long.xg, long.gf)[0, 1])
    out["corr_match_sot_vs_xg"] = float(np.corrcoef(long.sot, long.xg)[0, 1])
    out["corr_match_shots_vs_xg"] = float(np.corrcoef(long.shots, long.xg)[0, 1])
    X = np.column_stack([np.ones(len(long)), long.sot, long.shots - long.sot])
    beta, *_ = np.linalg.lstsq(X, long.xg.to_numpy(), rcond=None)
    pred = X @ beta
    ss_res, ss_tot = ((long.xg - pred) ** 2).sum(), ((long.xg - long.xg.mean()) ** 2).sum()
    out["xg_from_shots_regression"] = {"intercept": float(beta[0]), "per_shot_on_target": float(beta[1]),
                                       "per_other_shot": float(beta[2]), "r2": float(1 - ss_res / ss_tot)}
    # predictive stability: pre-match rolling-5 (shifted) of each signal vs the *next* match's goals scored
    res = {}
    for col in ("xg", "sot", "gf"):
        roll = long.groupby("team")[col].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
        ok = roll.notna()
        res[f"corr_rolling5_{col}_vs_next_goals"] = float(np.corrcoef(roll[ok], long.loc[ok, "gf"])[0, 1])
        res["n_team_matches"] = int(ok.sum())
    out["predictiveness_of_rolling5_for_next_match_goals"] = res
    path = REPORTS_DIR / "v25" / "xg_validation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
