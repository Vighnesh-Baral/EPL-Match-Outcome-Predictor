"""
Leakage-Safe Chronological Feature Engineering Module
EPL Match Outcome Predictor 2026/27
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data_loader import load_historical_matches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Elo parameters
INITIAL_ELO = 1500.0
ELO_K = 25.0
HOME_ADVANTAGE_ELO = 65.0
SEASON_REGRESSION = 0.20  # Regress 20% toward mean at start of new season


def build_chronological_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct strictly time-aware, leakage-safe features for every historical match.
    All features represent strictly pre-kickoff knowledge.
    """
    df = matches_df.sort_values(["Date", "MatchID"]).reset_index(drop=True)

    # State tracking dictionaries
    current_season = None
    elo_ratings: Dict[str, float] = defaultdict(lambda: INITIAL_ELO)

    # Match history per team: list of dicts with match performance
    team_history: Dict[str, List[dict]] = defaultdict(list)
    home_history: Dict[str, List[dict]] = defaultdict(list)
    away_history: Dict[str, List[dict]] = defaultdict(list)

    # Overall league outcomes prior to current date
    total_home_wins_prior = 0
    total_matches_prior = 0

    feature_rows = []

    for idx, row in df.iterrows():
        season = row["Season"]
        match_id = row["MatchID"]
        date = row["Date"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        ftr = row["FTR"]

        # Check season transition for Elo mean-reversion
        if current_season is not None and season != current_season:
            for team in elo_ratings:
                elo_ratings[team] = (1.0 - SEASON_REGRESSION) * elo_ratings[team] + SEASON_REGRESSION * INITIAL_ELO
        current_season = season

        # 1. PRE-MATCH ELO
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]
        elo_diff = home_elo - away_elo

        # 2. REST DAYS
        h_hist = team_history[home_team]
        a_hist = team_history[away_team]

        if h_hist:
            last_h_date = h_hist[-1]["date"]
            home_rest = (date - last_h_date).days if pd.notna(date) and pd.notna(last_h_date) else 7
        else:
            home_rest = 14
        home_rest = int(np.clip(home_rest, 1, 30))

        if a_hist:
            last_a_date = a_hist[-1]["date"]
            away_rest = (date - last_a_date).days if pd.notna(date) and pd.notna(last_a_date) else 7
        else:
            away_rest = 14
        away_rest = int(np.clip(away_rest, 1, 30))
        rest_diff = home_rest - away_rest

        # 3. RECENT FORM HELPERS
        def calc_ppg(history: List[dict], n: int) -> float:
            if not history:
                return 1.35  # Historical league average points per game
            sub = history[-n:]
            pts = [m["pts"] for m in sub]
            return float(np.mean(pts))

        def calc_stat_avg(history: List[dict], key: str, n: int, default: float) -> float:
            if not history:
                return default
            sub = history[-n:]
            vals = [m[key] for m in sub if key in m]
            return float(np.mean(vals)) if vals else default

        # Form Last 3, 5, 10
        h_ppg_l3 = calc_ppg(h_hist, 3)
        a_ppg_l3 = calc_ppg(a_hist, 3)
        h_ppg_l5 = calc_ppg(h_hist, 5)
        a_ppg_l5 = calc_ppg(a_hist, 5)
        h_ppg_l10 = calc_ppg(h_hist, 10)
        a_ppg_l10 = calc_ppg(a_hist, 10)

        h_gf_l5 = calc_stat_avg(h_hist, "gf", 5, 1.4)
        a_gf_l5 = calc_stat_avg(a_hist, "gf", 5, 1.2)
        h_ga_l5 = calc_stat_avg(h_hist, "ga", 5, 1.2)
        a_ga_l5 = calc_stat_avg(a_hist, "ga", 5, 1.4)
        h_gd_l5 = h_gf_l5 - h_ga_l5
        a_gd_l5 = a_gf_l5 - a_ga_l5

        # Shots & Shots on Target Last 5
        h_sot_l5 = calc_stat_avg(h_hist, "sot", 5, 4.5)
        a_sot_l5 = calc_stat_avg(a_hist, "sot", 5, 4.0)
        h_sota_l5 = calc_stat_avg(h_hist, "sota", 5, 4.0)
        a_sota_l5 = calc_stat_avg(a_hist, "sota", 5, 4.5)

        # 4. VENUE-SPECIFIC FORM
        h_venue_hist = home_history[home_team]
        a_venue_hist = away_history[away_team]
        h_home_ppg = calc_ppg(h_venue_hist, 5)
        a_away_ppg = calc_ppg(a_venue_hist, 5)
        h_home_gd = calc_stat_avg(h_venue_hist, "gd", 5, 0.3)
        a_away_gd = calc_stat_avg(a_venue_hist, "gd", 5, -0.3)

        # 5. ROLLING ATTACKING & DEFENSIVE STRENGTH (Last 10 matches)
        # Attacking: 50% goals + 50% shots on target
        # Defensive: 50% goals conceded + 50% shots on target allowed (inverted)
        h_att_10 = 0.50 * calc_stat_avg(h_hist, "gf", 10, 1.4) + 0.50 * (calc_stat_avg(h_hist, "sot", 10, 4.5) / 3.0)
        a_att_10 = 0.50 * calc_stat_avg(a_hist, "gf", 10, 1.2) + 0.50 * (calc_stat_avg(a_hist, "sot", 10, 4.0) / 3.0)

        h_def_10 = 0.50 * calc_stat_avg(h_hist, "ga", 10, 1.2) + 0.50 * (calc_stat_avg(h_hist, "sota", 10, 4.0) / 3.0)
        a_def_10 = 0.50 * calc_stat_avg(a_hist, "ga", 10, 1.4) + 0.50 * (calc_stat_avg(a_hist, "sota", 10, 4.5) / 3.0)

        # Net strength = Attacking - Defensive
        h_net_strength = h_att_10 - h_def_10
        a_net_strength = a_att_10 - a_def_10

        # Squad depth proxy: stability and rotation index over prior matches
        h_depth = 50.0 + min(15.0, len(h_hist) * 1.5)
        a_depth = 50.0 + min(15.0, len(a_hist) * 1.5)

        # 6. LEAGUE HOME ADVANTAGE PRIOR
        league_home_win_pct = (total_home_wins_prior / total_matches_prior) if total_matches_prior > 50 else 0.455

        # Compile match record row
        feature_rows.append({
            "MatchID": match_id,
            "Date": date,
            "Season": season,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTR": ftr,
            # Elo
            "Home_Elo": round(home_elo, 2),
            "Away_Elo": round(away_elo, 2),
            "Elo_Diff": round(elo_diff, 2),
            # Form PPG
            "Home_Form_Last3_PPG": round(h_ppg_l3, 2),
            "Away_Form_Last3_PPG": round(a_ppg_l3, 2),
            "Form_Last3_PPG_Diff": round(h_ppg_l3 - a_ppg_l3, 2),
            "Home_Form_Last5_PPG": round(h_ppg_l5, 2),
            "Away_Form_Last5_PPG": round(a_ppg_l5, 2),
            "Form_Last5_PPG_Diff": round(h_ppg_l5 - a_ppg_l5, 2),
            "Home_Form_Last10_PPG": round(h_ppg_l10, 2),
            "Away_Form_Last10_PPG": round(a_ppg_l10, 2),
            "Form_Last10_PPG_Diff": round(h_ppg_l10 - a_ppg_l10, 2),
            # Goals & Goal Difference
            "Home_GF_Last5": round(h_gf_l5, 2),
            "Away_GF_Last5": round(a_gf_l5, 2),
            "GF_Last5_Diff": round(h_gf_l5 - a_gf_l5, 2),
            "Home_GA_Last5": round(h_ga_l5, 2),
            "Away_GA_Last5": round(a_ga_l5, 2),
            "GA_Last5_Diff": round(h_ga_l5 - a_ga_l5, 2),
            "Home_GD_Last5": round(h_gd_l5, 2),
            "Away_GD_Last5": round(a_gd_l5, 2),
            "GD_Last5_Diff": round(h_gd_l5 - a_gd_l5, 2),
            # Shots on Target
            "Home_SoT_Last5": round(h_sot_l5, 2),
            "Away_SoT_Last5": round(a_sot_l5, 2),
            "SoT_Last5_Diff": round(h_sot_l5 - a_sot_l5, 2),
            "Home_SoTA_Last5": round(h_sota_l5, 2),
            "Away_SoTA_Last5": round(a_sota_l5, 2),
            "SoTA_Last5_Diff": round(h_sota_l5 - a_sota_l5, 2),
            # Attacking & Defensive Strength
            "Home_Attacking_Strength": round(h_att_10, 2),
            "Away_Attacking_Strength": round(a_att_10, 2),
            "Attacking_Strength_Diff": round(h_att_10 - a_att_10, 2),
            "Home_Defensive_Strength": round(h_def_10, 2),
            "Away_Defensive_Strength": round(a_def_10, 2),
            "Defensive_Strength_Diff": round(a_def_10 - h_def_10, 2),  # Positive if home defense is better (concedes fewer)
            "Home_Net_Strength": round(h_net_strength, 2),
            "Away_Net_Strength": round(a_net_strength, 2),
            "Net_Strength_Diff": round(h_net_strength - a_net_strength, 2),
            # Squad Depth
            "Home_Squad_Depth": round(h_depth, 2),
            "Away_Squad_Depth": round(a_depth, 2),
            "Squad_Depth_Diff": round(h_depth - a_depth, 2),
            # Venue specific
            "HomeTeam_Home_PPG_Last5": round(h_home_ppg, 2),
            "AwayTeam_Away_PPG_Last5": round(a_away_ppg, 2),
            "Venue_Form_Diff": round(h_home_ppg - a_away_ppg, 2),
            "HomeTeam_Home_GD_Last5": round(h_home_gd, 2),
            "AwayTeam_Away_GD_Last5": round(a_away_gd, 2),
            "Venue_GD_Diff": round(h_home_gd - a_away_gd, 2),
            # Rest days
            "Home_Rest_Days": home_rest,
            "Away_Rest_Days": away_rest,
            "Rest_Days_Diff": rest_diff,
            # Home advantage
            "Home_Advantage": 1.0,
            "League_Home_Win_Pct_Prior": round(league_home_win_pct, 3),
        })

        # --- POST-MATCH UPDATES (STRICTLY AFTER EXTRACTING PRE-MATCH FEATURES) ---
        fthg = row.get("FTHG", 0)
        ftag = row.get("FTAG", 0)
        hs = row.get("HS", 10)
        as_shots = row.get("AS", 10)
        hst = row.get("HST", 4)
        ast = row.get("AST", 4)

        # Points awarded
        if ftr == "H":
            h_pts, a_pts = 3, 0
            act_h, act_a = 1.0, 0.0
            total_home_wins_prior += 1
        elif ftr == "D":
            h_pts, a_pts = 1, 1
            act_h, act_a = 0.5, 0.5
        else:
            h_pts, a_pts = 0, 3
            act_h, act_a = 0.0, 1.0
        total_matches_prior += 1

        # Elo Update
        exp_h = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ADVANTAGE_ELO)) / 400.0))
        exp_a = 1.0 - exp_h

        # Margin of victory multiplier
        goal_diff = abs(fthg - ftag)
        mov_multiplier = 1.0 if goal_diff <= 1 else (1.5 if goal_diff == 2 else (1.75 + (goal_diff - 3) / 8.0))

        elo_ratings[home_team] = home_elo + ELO_K * mov_multiplier * (act_h - exp_h)
        elo_ratings[away_team] = away_elo + ELO_K * mov_multiplier * (act_a - exp_a)

        # Record match into team history
        h_rec = {
            "date": date, "pts": h_pts, "gf": fthg, "ga": ftag, "gd": fthg - ftag,
            "shots": hs, "sot": hst, "sota": ast
        }
        a_rec = {
            "date": date, "pts": a_pts, "gf": ftag, "ga": fthg, "gd": ftag - fthg,
            "shots": as_shots, "sot": ast, "sota": hst
        }
        team_history[home_team].append(h_rec)
        team_history[away_team].append(a_rec)
        home_history[home_team].append(h_rec)
        away_history[away_team].append(a_rec)

    features_df = pd.DataFrame(feature_rows)
    output_path = PROCESSED_DATA_DIR / "match_features.csv"
    features_df.to_csv(output_path, index=False)
    logger.info(f"Generated leakage-safe match features for {len(features_df)} matches, saved to {output_path}.")
    return features_df


if __name__ == "__main__":
    raw_matches = load_historical_matches()
    features = build_chronological_features(raw_matches)
    print("\n--- Match Features Dataset Sample ---")
    print("Features shape:", features.shape)
    print("Columns count:", len(features.columns))
    print(features[["Date", "Season", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo", "Elo_Diff", "Form_Last5_PPG_Diff", "FTR"]].head(10))
    print("\nTail matches (2025/26 season):")
    print(features[["Date", "Season", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo", "Elo_Diff", "Form_Last5_PPG_Diff", "FTR"]].tail(5))
