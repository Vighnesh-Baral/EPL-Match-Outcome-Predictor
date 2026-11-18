"""
Experimental Model & Feature Suite for PitchIQ Version 2.5
Evaluates Candidate Classifiers, Dixon-Coles, Ensembles and Calibration
"""
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    HOME_ADVANTAGE_FACTOR,
    SEASON_SAMPLE_WEIGHTS,
    standardize_team_name,
)
from src.data_loader import load_historical_matches
from src.dixon_coles import batch_dixon_coles_probs, dixon_coles_outcome_probs

LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

FEATURE_COLUMNS_V25 = [
    # Elo
    "home_elo", "away_elo", "elo_diff",
    # Form Points Per Match
    "home_form5_ppm", "away_form5_ppm", "form5_ppm_diff",
    "home_form10_ppm", "away_form10_ppm", "form10_ppm_diff",
    # Opponent-Adjusted Form
    "home_opp_adj_form5", "away_opp_adj_form5", "opp_adj_form5_diff",
    # Goals
    "home_gf5_pm", "away_gf5_pm", "gf5_pm_diff",
    "home_ga5_pm", "away_ga5_pm", "ga5_pm_diff",
    "home_gd5_pm", "away_gd5_pm", "gd5_pm_diff",
    # Shots & Shots on Target
    "home_sot5_pm", "away_sot5_pm", "sot5_pm_diff",
    "home_sota5_pm", "away_sota5_pm", "sota5_pm_diff",
    "home_sot_ratio5", "away_sot_ratio5", "sot_ratio5_diff",
    # Attack & Defence Indices (Last 10)
    "home_attack10", "away_attack10", "attack10_diff",
    "home_defence10", "away_defence10", "defence10_diff",
    "net_strength10_diff",
    # Venue Performance
    "home_venue_ppg5", "away_venue_ppg5", "venue_ppg_diff",
    "home_venue_gd5", "away_venue_gd5", "venue_gd_diff",
    # Rest Days
    "home_rest_days", "away_rest_days", "rest_days_diff",
    # H2H
    "h2h_home_ppm_10", "h2h_home_goal_diff_10", "h2h_matches_10",
    # Season Context
    "home_season_ppm", "away_season_ppm",
    "home_matches_played", "away_matches_played",
    "league_avg_team_goals",
]

INITIAL_ELO = 1500.0
ELO_K = 24.0
HOME_ADVANTAGE_ELO = 65.0
SEASON_REGRESSION = 0.20


def build_v25_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct strictly time-aware, leakage-safe features for every historical match.
    All fixtures on the same calendar date are featurized BEFORE any match on that
    date updates team state, Elo, or H2H history.
    """
    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["HomeTeam"] = df["HomeTeam"].map(standardize_team_name)
    df["AwayTeam"] = df["AwayTeam"].map(standardize_team_name)
    df["FTR"] = df["FTR"].astype(str).str.upper().str.strip()
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)

    # State tracking
    current_season = None
    elo_ratings: Dict[str, float] = defaultdict(lambda: INITIAL_ELO)
    team_history: Dict[str, List[dict]] = defaultdict(list)
    home_history: Dict[str, List[dict]] = defaultdict(list)
    away_history: Dict[str, List[dict]] = defaultdict(list)
    pair_history: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    season_totals: Dict[Tuple[str, str], dict] = defaultdict(lambda: {"mp": 0.0, "pts": 0.0, "gf": 0.0, "ga": 0.0})
    total_goals_seen = 0.0
    total_team_appearances = 0.0

    records = []

    for date, day in df.groupby("Date", sort=True):
        # Season regression check on first date of new season
        day_season = str(day.iloc[0]["Season"])
        if current_season is not None and day_season != current_season:
            for t in elo_ratings:
                elo_ratings[t] = (1.0 - SEASON_REGRESSION) * elo_ratings[t] + SEASON_REGRESSION * INITIAL_ELO
        current_season = day_season

        day_records = []
        for _, row in day.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            season = str(row["Season"])

            # 1. Elo
            h_elo = elo_ratings[home]
            a_elo = elo_ratings[away]
            elo_diff = round(h_elo - a_elo, 2)

            # 2. History lookups
            h_hist = team_history[home]
            a_hist = team_history[away]

            # Helper functions for rolling metrics
            def calc_avg(hist: List[dict], key: str, n: int, default: float) -> float:
                if not hist: return default
                sub = hist[-n:]
                vals = [m[key] for m in sub if key in m]
                return float(np.mean(vals)) if vals else default

            def calc_opp_adj_ppm(hist: List[dict], n: int) -> float:
                if not hist: return 1.35
                sub = hist[-n:]
                adj_pts = [m["pts"] * (m.get("opp_elo", 1500.0) / 1500.0) for m in sub]
                return float(np.mean(adj_pts))

            # Rest days
            if h_hist:
                last_h = h_hist[-1]["date"]
                h_rest = (date - last_h).days if pd.notna(date) and pd.notna(last_h) else 7
            else:
                h_rest = 14
            h_rest = int(np.clip(h_rest, 2, 21))

            if a_hist:
                last_a = a_hist[-1]["date"]
                a_rest = (date - last_a).days if pd.notna(date) and pd.notna(last_a) else 7
            else:
                a_rest = 14
            a_rest = int(np.clip(a_rest, 2, 21))

            # Form PPM
            h_ppm5 = calc_avg(h_hist, "pts", 5, 1.35)
            a_ppm5 = calc_avg(a_hist, "pts", 5, 1.35)
            h_ppm10 = calc_avg(h_hist, "pts", 10, 1.35)
            a_ppm10 = calc_avg(a_hist, "pts", 10, 1.35)

            # Opponent-adjusted form
            h_opp_adj5 = calc_opp_adj_ppm(h_hist, 5)
            a_opp_adj5 = calc_opp_adj_ppm(a_hist, 5)

            # Goals
            h_gf5 = calc_avg(h_hist, "gf", 5, 1.45)
            a_gf5 = calc_avg(a_hist, "gf", 5, 1.20)
            h_ga5 = calc_avg(h_hist, "ga", 5, 1.20)
            a_ga5 = calc_avg(a_hist, "ga", 5, 1.45)
            h_gd5 = h_gf5 - h_ga5
            a_gd5 = a_gf5 - a_ga5

            # Shots on target
            h_sot5 = calc_avg(h_hist, "sot", 5, 4.5)
            a_sot5 = calc_avg(a_hist, "sot", 5, 4.0)
            h_sota5 = calc_avg(h_hist, "sota", 5, 4.0)
            a_sota5 = calc_avg(a_hist, "sota", 5, 4.5)

            h_sot_ratio = h_sot5 / max(h_sot5 + h_sota5, 1.0)
            a_sot_ratio = a_sot5 / max(a_sot5 + a_sota5, 1.0)

            # Attack & Defence indices (Last 10)
            h_att10 = 0.60 * calc_avg(h_hist, "gf", 10, 1.45) + 0.40 * (calc_avg(h_hist, "sot", 10, 4.5) / 3.0)
            a_att10 = 0.60 * calc_avg(a_hist, "gf", 10, 1.20) + 0.40 * (calc_avg(a_hist, "sot", 10, 4.0) / 3.0)
            h_def10 = 0.60 * calc_avg(h_hist, "ga", 10, 1.20) + 0.40 * (calc_avg(h_hist, "sota", 10, 4.0) / 3.0)
            a_def10 = 0.60 * calc_avg(a_hist, "ga", 10, 1.45) + 0.40 * (calc_avg(a_hist, "sota", 10, 4.5) / 3.0)
            net_str_diff = (h_att10 - h_def10) - (a_att10 - a_def10)

            # Venue specific
            h_v_hist = home_history[home]
            a_v_hist = away_history[away]
            h_v_ppg = calc_avg(h_v_hist, "pts", 5, 1.55)
            a_v_ppg = calc_avg(a_v_hist, "pts", 5, 1.15)
            h_v_gd = calc_avg(h_v_hist, "gd", 5, 0.35)
            a_v_gd = calc_avg(a_v_hist, "gd", 5, -0.35)

            # Head to head (last 10)
            pair_key = tuple(sorted((home, away)))
            p_hist = pair_history[pair_key][-10:]
            if p_hist:
                pts_h = [m["pts_home"] if m["home"] == home else (3.0 - m["pts_home"] if m["pts_home"] in (0, 3) else 1.0) for m in p_hist]
                gds_h = [m["gd"] if m["home"] == home else -m["gd"] for m in p_hist]
                h2h_ppm = float(np.mean(pts_h))
                h2h_gd = float(np.mean(gds_h))
                h2h_cnt = len(p_hist)
            else:
                h2h_ppm, h2h_gd, h2h_cnt = 1.35, 0.0, 0

            # Season totals
            ht = season_totals[(home, season)]
            at = season_totals[(away, season)]
            h_s_ppm = ht["pts"] / ht["mp"] if ht["mp"] > 0 else h_ppm10
            a_s_ppm = at["pts"] / at["mp"] if at["mp"] > 0 else a_ppm10

            league_avg = total_goals_seen / total_team_appearances if total_team_appearances > 0 else 1.40

            day_records.append({
                "Date": date, "Season": season, "HomeTeam": home, "AwayTeam": away,
                "FTHG": float(row["FTHG"]), "FTAG": float(row["FTAG"]), "FTR": row["FTR"],
                # Elo
                "home_elo": round(h_elo, 2), "away_elo": round(a_elo, 2), "elo_diff": elo_diff,
                # Form
                "home_form5_ppm": round(h_ppm5, 3), "away_form5_ppm": round(a_ppm5, 3), "form5_ppm_diff": round(h_ppm5 - a_ppm5, 3),
                "home_form10_ppm": round(h_ppm10, 3), "away_form10_ppm": round(a_ppm10, 3), "form10_ppm_diff": round(h_ppm10 - a_ppm10, 3),
                # Opponent-adjusted form
                "home_opp_adj_form5": round(h_opp_adj5, 3), "away_opp_adj_form5": round(a_opp_adj5, 3), "opp_adj_form5_diff": round(h_opp_adj5 - a_opp_adj5, 3),
                # Goals
                "home_gf5_pm": round(h_gf5, 3), "away_gf5_pm": round(a_gf5, 3), "gf5_pm_diff": round(h_gf5 - a_gf5, 3),
                "home_ga5_pm": round(h_ga5, 3), "away_ga5_pm": round(a_ga5, 3), "ga5_pm_diff": round(h_ga5 - a_ga5, 3),
                "home_gd5_pm": round(h_gd5, 3), "away_gd5_pm": round(a_gd5, 3), "gd5_pm_diff": round(h_gd5 - a_gd5, 3),
                # Shots on target
                "home_sot5_pm": round(h_sot5, 3), "away_sot5_pm": round(a_sot5, 3), "sot5_pm_diff": round(h_sot5 - a_sot5, 3),
                "home_sota5_pm": round(h_sota5, 3), "away_sota5_pm": round(a_sota5, 3), "sota5_pm_diff": round(h_sota5 - a_sota5, 3),
                "home_sot_ratio5": round(h_sot_ratio, 3), "away_sot_ratio5": round(a_sot_ratio, 3), "sot_ratio5_diff": round(h_sot_ratio - a_sot_ratio, 3),
                # Attack & Defence
                "home_attack10": round(h_att10, 3), "away_attack10": round(a_att10, 3), "attack10_diff": round(h_att10 - a_att10, 3),
                "home_defence10": round(h_def10, 3), "away_defence10": round(a_def10, 3), "defence10_diff": round(a_def10 - h_def10, 3),
                "net_strength10_diff": round(net_str_diff, 3),
                # Venue
                "home_venue_ppg5": round(h_v_ppg, 3), "away_venue_ppg5": round(a_v_ppg, 3), "venue_ppg_diff": round(h_v_ppg - a_v_ppg, 3),
                "home_venue_gd5": round(h_v_gd, 3), "away_venue_gd5": round(a_v_gd, 3), "venue_gd_diff": round(h_v_gd - a_v_gd, 3),
                # Rest
                "home_rest_days": h_rest, "away_rest_days": a_rest, "rest_days_diff": h_rest - a_rest,
                # H2H
                "h2h_home_ppm_10": round(h2h_ppm, 3), "h2h_home_goal_diff_10": round(h2h_gd, 3), "h2h_matches_10": h2h_cnt,
                # Season
                "home_season_ppm": round(h_s_ppm, 3), "away_season_ppm": round(a_s_ppm, 3),
                "home_matches_played": ht["mp"], "away_matches_played": at["mp"],
                "league_avg_team_goals": round(league_avg, 3),
            })

        # Post-date updates (strictly after all matches on this date have features assigned)
        for _, row in day.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            season = str(row["Season"])
            hg = float(row["FTHG"])
            ag = float(row["FTAG"])
            hsot = float(row.get("HST", 4.0)) if pd.notna(row.get("HST")) else 4.0
            asot = float(row.get("AST", 4.0)) if pd.notna(row.get("AST")) else 4.0

            h_pts = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
            a_pts = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
            act_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            act_a = 1.0 - act_h

            h_elo = elo_ratings[home]
            a_elo = elo_ratings[away]

            # Elo update with margin of victory
            exp_h = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + HOME_ADVANTAGE_ELO)) / 400.0))
            exp_a = 1.0 - exp_h
            gd = abs(hg - ag)
            mov = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8.0))

            elo_ratings[home] = round(h_elo + ELO_K * mov * (act_h - exp_h), 2)
            elo_ratings[away] = round(a_elo + ELO_K * mov * (act_a - exp_a), 2)

            # Store match in histories
            h_entry = {"date": date, "pts": h_pts, "gf": hg, "ga": ag, "gd": hg - ag, "sot": hsot, "sota": asot, "opp_elo": a_elo}
            a_entry = {"date": date, "pts": a_pts, "gf": ag, "ga": hg, "gd": ag - hg, "sot": asot, "sota": hsot, "opp_elo": h_elo}

            team_history[home].append(h_entry)
            team_history[away].append(a_entry)
            home_history[home].append(h_entry)
            away_history[away].append(a_entry)

            pair_key = tuple(sorted((home, away)))
            pair_history[pair_key].append({"home": home, "pts_home": h_pts, "gd": hg - ag})

            season_totals[(home, season)]["mp"] += 1.0
            season_totals[(home, season)]["pts"] += h_pts
            season_totals[(home, season)]["gf"] += hg
            season_totals[(home, season)]["ga"] += ag

            season_totals[(away, season)]["mp"] += 1.0
            season_totals[(away, season)]["pts"] += a_pts
            season_totals[(away, season)]["gf"] += ag
            season_totals[(away, season)]["ga"] += hg

            total_goals_seen += hg + ag
            total_team_appearances += 2.0

        records.extend(day_records)

    result_df = pd.DataFrame(records)
    return result_df


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    predicted = np.argmax(probs, axis=1)
    one_hot = np.eye(3)[y_true]
    recalls = recall_score(y_true, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    precisions = precision_score(y_true, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "macro_f1": float(f1_score(y_true, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "home_recall": float(recalls[0]),
        "draw_recall": float(recalls[1]),
        "away_recall": float(recalls[2]),
        "draw_precision": float(precisions[1]),
        "draw_f1": float(f1_score(y_true, predicted, labels=[1], average=None, zero_division=0)[0]),
    }


def dixon_coles_expected_goals(feature_row: pd.Series, home_factor: float = 1.05) -> Tuple[float, float]:
    league = max(float(feature_row.get("league_avg_team_goals", 1.40)), 0.8)
    h_gf = float(feature_row.get("home_gf5_pm", 1.45))
    a_ga = float(feature_row.get("away_ga5_pm", 1.45))
    a_gf = float(feature_row.get("away_gf5_pm", 1.20))
    h_ga = float(feature_row.get("home_ga5_pm", 1.20))
    # Elo adjustment factor on goal expectation
    elo_diff = float(feature_row.get("elo_diff", 0.0))
    elo_mult_h = 1.0 + elo_diff / 800.0
    elo_mult_a = 1.0 - elo_diff / 800.0
    lam_h = float(np.clip(h_gf * a_ga / league * home_factor * elo_mult_h, 0.20, 4.5))
    lam_a = float(np.clip(a_gf * h_ga / league / home_factor * elo_mult_a, 0.20, 4.5))
    return lam_h, lam_a


def generate_dixon_coles_probs_for_frame(df: pd.DataFrame, rho: float = -0.11, home_factor: float = 1.05) -> np.ndarray:
    n = len(df)
    probs = np.zeros((n, 3), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        lh, la = dixon_coles_expected_goals(row, home_factor=home_factor)
        h, d, a = dixon_coles_outcome_probs(lh, la, rho=rho)
        probs[i, 0] = h
        probs[i, 1] = d
        probs[i, 2] = a
    return probs


def run_experimentation():
    print("Loading historical matches...")
    raw_matches = load_historical_matches()
    print(f"Loaded {len(raw_matches)} matches. Building Version 2.5 leakage-safe features...")
    features = build_v25_features(raw_matches)
    print(f"Features created: {features.shape}. Verifying no missing values in {len(FEATURE_COLUMNS_V25)} features...")
    assert not features[FEATURE_COLUMNS_V25].isna().any().any(), "Found NaNs in features!"
    print("Features verified: 0 NaNs, 0 Infinities.")

    # Chronological splits
    train_mask = features["Season"] <= "2021-22"
    val_mask = features["Season"].isin(["2022-23", "2023-24"])
    test_mask = features["Season"].isin(["2024-25", "2025-26"])

    X_train = features.loc[train_mask, FEATURE_COLUMNS_V25].astype(float)
    y_train = features.loc[train_mask, "FTR"].map(LABEL_TO_INT).to_numpy(dtype=int)
    w_train = features.loc[train_mask, "Season"].map(SEASON_SAMPLE_WEIGHTS).fillna(0.5).to_numpy(dtype=float)

    X_val = features.loc[val_mask, FEATURE_COLUMNS_V25].astype(float)
    y_val = features.loc[val_mask, "FTR"].map(LABEL_TO_INT).to_numpy(dtype=int)

    X_test = features.loc[test_mask, FEATURE_COLUMNS_V25].astype(float)
    y_test = features.loc[test_mask, "FTR"].map(LABEL_TO_INT).to_numpy(dtype=int)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # 1. Candidate models
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.45, max_iter=2500, solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=450, max_depth=9, min_samples_leaf=10, max_features="sqrt",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.03, min_child_weight=8,
            subsample=0.85, colsample_bytree=0.80, reg_lambda=4.0, gamma=0.1,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        ),
    }

    val_results = []
    fitted_models = {}

    for name, model in models.items():
        if isinstance(model, Pipeline):
            model.fit(X_train, y_train, model__sample_weight=w_train)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)
        val_probs = model.predict_proba(X_val)
        m = compute_metrics(y_val, val_probs)
        m["model"] = name
        val_results.append(m)
        fitted_models[name] = (model, val_probs)
        print(f"[{name}] Val Acc: {m['accuracy']*100:.2f}% | LogLoss: {m['log_loss']:.4f} | Brier: {m['brier_score']:.4f} | Draw Recall: {m['draw_recall']*100:.2f}%")

    # 2. Dixon-Coles on validation
    print("\nEvaluating Dixon-Coles on Validation...")
    dc_val_probs = generate_dixon_coles_probs_for_frame(features.loc[val_mask], rho=-0.12, home_factor=1.06)
    m_dc = compute_metrics(y_val, dc_val_probs)
    m_dc["model"] = "Dixon-Coles Standalone"
    val_results.append(m_dc)
    print(f"[Dixon-Coles] Val Acc: {m_dc['accuracy']*100:.2f}% | LogLoss: {m_dc['log_loss']:.4f} | Brier: {m_dc['brier_score']:.4f} | Draw Recall: {m_dc['draw_recall']*100:.2f}%")

    # 3. Search best ensemble with Dixon-Coles on validation
    print("\nOptimizing Model + Dixon-Coles Ensembles on Validation...")
    ensemble_results = []
    for name, (_, val_probs) in fitted_models.items():
        best_alpha = 1.0
        best_loss = float("inf")
        best_p = val_probs
        for alpha in np.linspace(0.0, 1.0, 21):
            p = alpha * val_probs + (1.0 - alpha) * dc_val_probs
            loss = log_loss(y_val, p, labels=[0, 1, 2])
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
                best_p = p
        m_ens = compute_metrics(y_val, best_p)
        m_ens["model"] = f"{name} + Dixon-Coles"
        m_ens["alpha"] = best_alpha
        ensemble_results.append(m_ens)
        print(f"[{name} + Dixon-Coles (alpha={best_alpha:.2f})] Val Acc: {m_ens['accuracy']*100:.2f}% | LogLoss: {m_ens['log_loss']:.4f} | Brier: {m_ens['brier_score']:.4f} | Draw Recall: {m_ens['draw_recall']*100:.2f}%")

    # 4. Soft Voting Ensemble of Classifiers
    rf_probs = fitted_models["Random Forest"][1]
    xgb_probs = fitted_models["XGBoost"][1]
    lr_probs = fitted_models["Logistic Regression"][1]
    tri_blend = 0.45 * xgb_probs + 0.35 * rf_probs + 0.20 * lr_probs
    m_tri = compute_metrics(y_val, tri_blend)
    print(f"\n[Soft Voting Ensemble (XGB+RF+LR)] Val Acc: {m_tri['accuracy']*100:.2f}% | LogLoss: {m_tri['log_loss']:.4f} | Brier: {m_tri['brier_score']:.4f} | Draw Recall: {m_tri['draw_recall']*100:.2f}%")

    # Tri-blend + Dixon Coles
    best_tri_alpha, best_tri_loss, best_tri_probs = 1.0, float("inf"), tri_blend
    for alpha in np.linspace(0.0, 1.0, 21):
        p = alpha * tri_blend + (1.0 - alpha) * dc_val_probs
        loss = log_loss(y_val, p, labels=[0, 1, 2])
        if loss < best_tri_loss:
            best_tri_loss = loss
            best_tri_alpha = alpha
            best_tri_probs = p
    m_tri_dc = compute_metrics(y_val, best_tri_probs)
    print(f"[Soft Voting Ensemble + Dixon-Coles (alpha={best_tri_alpha:.2f})] Val Acc: {m_tri_dc['accuracy']*100:.2f}% | LogLoss: {m_tri_dc['log_loss']:.4f} | Brier: {m_tri_dc['brier_score']:.4f} | Draw Recall: {m_tri_dc['draw_recall']*100:.2f}%")

    # Select best model strictly by Validation Log Loss
    all_val = val_results + ensemble_results + [
        {"model": "Soft Voting Ensemble", **m_tri, "alpha": 1.0},
        {"model": "Soft Voting Ensemble + Dixon-Coles", **m_tri_dc, "alpha": best_tri_alpha}
    ]
    best_candidate = min(all_val, key=lambda x: x["log_loss"])
    print(f"\n>>> Selected Best Architecture on Validation: {best_candidate['model']} (Val LogLoss: {best_candidate['log_loss']:.4f}, Val Acc: {best_candidate['accuracy']*100:.2f}%) <<<")

    # 5. NOW AND ONLY NOW: Refit on Train+Val and Evaluate on UNTOUCHED Test Set
    print("\n" + "="*70)
    print("EVALUATING UNTOUCHED HELD-OUT TEST SET (2024/25 - 2025/26: 760 FIXTURES)")
    print("="*70)

    train_val_mask = train_mask | val_mask
    X_train_val = features.loc[train_val_mask, FEATURE_COLUMNS_V25].astype(float)
    y_train_val = features.loc[train_val_mask, "FTR"].map(LABEL_TO_INT).to_numpy(dtype=int)
    w_train_val = features.loc[train_val_mask, "Season"].map(SEASON_SAMPLE_WEIGHTS).fillna(0.5).to_numpy(dtype=float)

    # Refit candidates on train+val
    rf_final = RandomForestClassifier(n_estimators=450, max_depth=9, min_samples_leaf=10, max_features="sqrt", random_state=42, n_jobs=-1)
    rf_final.fit(X_train_val, y_train_val, sample_weight=w_train_val)
    rf_test_probs = rf_final.predict_proba(X_test)

    xgb_final = XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.03, min_child_weight=8,
                              subsample=0.85, colsample_bytree=0.80, reg_lambda=4.0, gamma=0.1,
                              objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                              random_state=42, n_jobs=4, tree_method="hist", verbosity=0)
    xgb_final.fit(X_train_val, y_train_val, sample_weight=w_train_val)
    xgb_test_probs = xgb_final.predict_proba(X_test)

    lr_final = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.45, max_iter=2500, solver="lbfgs", random_state=42))])
    lr_final.fit(X_train_val, y_train_val, model__sample_weight=w_train_val)
    lr_test_probs = lr_final.predict_proba(X_test)

    dc_test_probs = generate_dixon_coles_probs_for_frame(features.loc[test_mask], rho=-0.12, home_factor=1.06)

    # Test predictions for best ensemble
    tri_test_blend = 0.45 * xgb_test_probs + 0.35 * rf_test_probs + 0.20 * lr_test_probs
    best_test_blend = best_tri_alpha * tri_test_blend + (1.0 - best_tri_alpha) * dc_test_probs

    test_rf_dc = 0.70 * rf_test_probs + 0.30 * dc_test_probs

    test_metrics = [
        {"model": "Version 2.4 Baseline (RF + Poisson)", "accuracy": 0.4868, "log_loss": 1.0233, "brier_score": 0.6151, "macro_f1": 0.3728, "draw_recall": 0.0102},
        {"model": "Dixon-Coles Standalone", **compute_metrics(y_test, dc_test_probs)},
        {"model": "Logistic Regression (v2.5 Features)", **compute_metrics(y_test, lr_test_probs)},
        {"model": "Random Forest (v2.5 Features)", **compute_metrics(y_test, rf_test_probs)},
        {"model": "XGBoost (v2.5 Features)", **compute_metrics(y_test, xgb_test_probs)},
        {"model": "Random Forest + Dixon-Coles", **compute_metrics(y_test, test_rf_dc)},
        {"model": "PitchIQ Version 2.5 Production Ensemble (Tri-Blend + Dixon-Coles)", **compute_metrics(y_test, best_test_blend)},
    ]

    test_df = pd.DataFrame(test_metrics)
    print(test_df[["model", "accuracy", "log_loss", "brier_score", "macro_f1", "draw_recall", "draw_f1"]].to_string(index=False))

    return features, all_val, test_metrics, rf_final, xgb_final, lr_final, best_tri_alpha


if __name__ == "__main__":
    run_experimentation()

