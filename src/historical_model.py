"""Leakage-controlled, recency-weighted historical EPL prediction models.

Historical match features are reconstructed immediately before each kickoff.
All fixtures on the same calendar date are scored before any result from that
same date is added to team or H2H histories, preventing same-day leakage.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import math

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    HOME_ADVANTAGE_FACTOR,
    MODELS_DIR,
    RAW_CURRENT_DIR,
    SEASON_SAMPLE_WEIGHTS,
    standardize_team_name,
)
from src.data_loader import DataValidationError, load_historical_matches, load_current_season_results, _read_csv

FEATURE_COLUMNS = [
    "home_form5_ppm", "away_form5_ppm",
    "home_season_ppm", "away_season_ppm",
    "home_season_gf_pm", "away_season_gf_pm",
    "home_season_ga_pm", "away_season_ga_pm",
    "home_season_win_rate", "away_season_win_rate",
    "home_matches_played", "away_matches_played",
    "h2h_home_ppm_10", "h2h_home_goal_diff_10", "h2h_matches_10",
    "league_avg_team_goals",
]
LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
MODEL_FILE = MODELS_DIR / "historical_predictor_v2.joblib"
REPORT_JSON = Path(__file__).resolve().parent.parent / "reports" / "model_evaluation_v2.json"
REPORT_CSV = Path(__file__).resolve().parent.parent / "reports" / "model_comparison_v2.csv"


def _season_from_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("season-", "")
    if len(stem) != 4 or not stem.isdigit():
        raise DataValidationError(f"Cannot infer season from filename: {filename}")
    return f"20{stem[:2]}-{stem[2:]}"


def load_training_matches() -> pd.DataFrame:
    """Load 2015/16–2025/26 and an optional 2026/27 match log if supplied."""
    past = load_historical_matches().copy()
    current = load_current_season_results()
    if not current.empty:
        past = pd.concat([past, current], ignore_index=True, sort=False)
    past = past.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    duplicate = past.duplicated(["Date", "HomeTeam", "AwayTeam"], keep=False)
    if duplicate.any():
        raise DataValidationError(f"Training data contains {int(duplicate.sum())} duplicate fixture rows.")
    if past.empty:
        raise DataValidationError("No completed match results were found for model training.")
    return past


def _summarise(history: list[dict[str, float]], n: int) -> dict[str, float]:
    recent = history[-n:]
    if not recent:
        return {"ppm": 1.35, "gf": 1.45, "ga": 1.45, "win_rate": 0.40}
    count = len(recent)
    return {
        "ppm": float(sum(x["points"] for x in recent) / count),
        "gf": float(sum(x["gf"] for x in recent) / count),
        "ga": float(sum(x["ga"] for x in recent) / count),
        "win_rate": float(sum(x["win"] for x in recent) / count),
    }


def _h2h_summary(history: list[dict[str, float]], home_team: str, limit: int = 10) -> tuple[float, float, int]:
    # Entries are stored from the original home club's perspective.
    selected = history[-limit:]
    if not selected:
        return 1.35, 0.0, 0
    points, goal_diff = 0.0, 0.0
    for item in selected:
        sign = 1.0 if item["home_team"] == home_team else -1.0
        goal_diff += sign * item["goal_diff"]
        points += sign * item["points_home"] if sign > 0 else (3.0 - item["points_home"] if item["points_home"] in (0.0, 3.0) else 1.0)
    return points / len(selected), goal_diff / len(selected), len(selected)


def build_historical_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build features that use only results strictly before the fixture date."""
    required = {"Date", "Season", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    missing = required - set(matches.columns)
    if missing:
        raise DataValidationError(f"Match table missing feature-building columns: {sorted(missing)}")
    data = matches.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data["HomeTeam"] = data["HomeTeam"].map(standardize_team_name)
    data["AwayTeam"] = data["AwayTeam"].map(standardize_team_name)
    data["FTR"] = data["FTR"].astype(str).str.upper()
    data = data[data["FTR"].isin(["H", "D", "A"])].copy()
    data = data.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable")

    team_history: dict[str, list[dict[str, float]]] = defaultdict(list)
    season_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"mp": 0.0, "points": 0.0, "gf": 0.0, "ga": 0.0, "wins": 0.0}
    )
    pair_history: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    total_goals_seen = 0.0
    total_team_appearances = 0.0
    records: list[dict[str, Any]] = []

    # Iterate date groups: no result from a fixture on this date can leak into
    # another fixture played on the same date.
    for date, day in data.groupby("Date", sort=True):
        day_records: list[dict[str, Any]] = []
        for _, row in day.iterrows():
            home, away, season = row["HomeTeam"], row["AwayTeam"], str(row["Season"])
            hs, aw = float(row["FTHG"]), float(row["FTAG"])
            h5 = _summarise(team_history[home], 5)
            a5 = _summarise(team_history[away], 5)
            h10 = _summarise(team_history[home], 10)
            a10 = _summarise(team_history[away], 10)
            ht = season_totals[(home, season)]
            at = season_totals[(away, season)]
            hmp, amp = ht["mp"], at["mp"]
            hseason = {
                "ppm": ht["points"] / hmp if hmp else h10["ppm"],
                "gf": ht["gf"] / hmp if hmp else h10["gf"],
                "ga": ht["ga"] / hmp if hmp else h10["ga"],
                "win_rate": ht["wins"] / hmp if hmp else h10["win_rate"],
            }
            aseason = {
                "ppm": at["points"] / amp if amp else a10["ppm"],
                "gf": at["gf"] / amp if amp else a10["gf"],
                "ga": at["ga"] / amp if amp else a10["ga"],
                "win_rate": at["wins"] / amp if amp else a10["win_rate"],
            }
            h2h_ppm, h2h_gd, h2h_count = _h2h_summary(pair_history[tuple(sorted((home, away)))], home)
            league_avg = total_goals_seen / total_team_appearances if total_team_appearances else 1.35
            day_records.append({
                "Date": date, "Season": season, "HomeTeam": home, "AwayTeam": away,
                "FTR": row["FTR"], "FTHG": hs, "FTAG": aw,
                "home_form5_ppm": h5["ppm"], "away_form5_ppm": a5["ppm"],
                "home_season_ppm": hseason["ppm"], "away_season_ppm": aseason["ppm"],
                "home_season_gf_pm": hseason["gf"], "away_season_gf_pm": aseason["gf"],
                "home_season_ga_pm": hseason["ga"], "away_season_ga_pm": aseason["ga"],
                "home_season_win_rate": hseason["win_rate"], "away_season_win_rate": aseason["win_rate"],
                "home_matches_played": hmp, "away_matches_played": amp,
                "h2h_home_ppm_10": h2h_ppm, "h2h_home_goal_diff_10": h2h_gd,
                "h2h_matches_10": h2h_count, "league_avg_team_goals": league_avg,
            })

        # Update states only after all matches on the date have features assigned.
        for _, row in day.iterrows():
            home, away, season = row["HomeTeam"], row["AwayTeam"], str(row["Season"])
            hs, aw = float(row["FTHG"]), float(row["FTAG"])
            home_points = 3.0 if hs > aw else 1.0 if hs == aw else 0.0
            away_points = 3.0 if aw > hs else 1.0 if hs == aw else 0.0
            home_win = 1.0 if hs > aw else 0.0
            away_win = 1.0 if aw > hs else 0.0
            team_history[home].append({"points": home_points, "gf": hs, "ga": aw, "win": home_win})
            team_history[away].append({"points": away_points, "gf": aw, "ga": hs, "win": away_win})
            for team, points, gf, ga, win in (
                (home, home_points, hs, aw, home_win), (away, away_points, aw, hs, away_win)
            ):
                totals = season_totals[(team, season)]
                totals["mp"] += 1.0
                totals["points"] += points
                totals["gf"] += gf
                totals["ga"] += ga
                totals["wins"] += win
            pair_history[tuple(sorted((home, away)))].append({
                "home_team": home, "points_home": home_points, "goal_diff": hs - aw
            })
            total_goals_seen += hs + aw
            total_team_appearances += 2.0
        records.extend(day_records)

    result = pd.DataFrame(records).sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if result[FEATURE_COLUMNS].isna().any().any():
        bad = result[FEATURE_COLUMNS].columns[result[FEATURE_COLUMNS].isna().any()].tolist()
        raise DataValidationError(f"Historical features contain missing values: {bad}")
    return result


def season_sample_weights(seasons: pd.Series) -> np.ndarray:
    """Map the user's season-priority hierarchy to per-match training weights."""
    return seasons.astype(str).map(SEASON_SAMPLE_WEIGHTS).fillna(0.50).to_numpy(dtype=float)


def _poisson_goal_probs(lam: float, max_goals: int = 10) -> np.ndarray:
    lam = float(np.clip(lam, 0.10, 5.0))
    probs = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(max_goals + 1)], dtype=float)
    return probs / probs.sum()


def poisson_probabilities_from_features(frame: pd.DataFrame) -> np.ndarray:
    """Historical Poisson baseline using only pre-match season-to-date team rates."""
    output = []
    for row in frame.itertuples(index=False):
        league = max(float(row.league_avg_team_goals), 0.75)
        lam_h = float(row.home_season_gf_pm) * float(row.away_season_ga_pm) / league * HOME_ADVANTAGE_FACTOR
        lam_a = float(row.away_season_gf_pm) * float(row.home_season_ga_pm) / league / HOME_ADVANTAGE_FACTOR
        ph, pa = _poisson_goal_probs(lam_h), _poisson_goal_probs(lam_a)
        matrix = np.outer(ph, pa)
        values = np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()], dtype=float)
        output.append(values / values.sum())
    return np.asarray(output, dtype=float)


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, log_loss
    one_hot = np.eye(3)[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, np.argmax(probs, axis=1))),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
    }


def _best_blend_weight(y_true: np.ndarray, ml_probs: np.ndarray, poisson_probs: np.ndarray) -> tuple[float, np.ndarray, float]:
    from sklearn.metrics import log_loss
    best_alpha, best_loss, best_probs = 0.5, float("inf"), None
    for alpha in np.linspace(0.0, 1.0, 21):
        combined = alpha * ml_probs + (1.0 - alpha) * poisson_probs
        loss = log_loss(y_true, combined, labels=[0, 1, 2])
        if loss < best_loss:
            best_alpha, best_loss, best_probs = float(alpha), float(loss), combined
    return best_alpha, best_probs, best_loss


def _candidate_models() -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.65, max_iter=2500, solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=450, max_depth=12, min_samples_leaf=8, max_features="sqrt",
            class_weight=None, random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=450, max_depth=3, learning_rate=0.035, min_child_weight=8,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, gamma=0.05,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        ),
    }


def _fit(model: Any, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any:
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def _predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    ordered = np.zeros((len(X), 3), dtype=float)
    for idx, cls in enumerate(classes):
        ordered[:, int(cls)] = raw[:, idx]
    ordered = np.clip(ordered, 1e-8, 1.0)
    return ordered / ordered.sum(axis=1, keepdims=True)


def train_and_save_model() -> dict[str, Any]:
    """Compare classifiers chronologically, evaluate an untouched test period,
    then fit the chosen classifier on all available historical results.
    """
    matches = load_training_matches()
    features = build_historical_features(matches)
    features = features.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    y = features["FTR"].map(LABEL_TO_INT).to_numpy(dtype=int)
    X = features[FEATURE_COLUMNS].astype(float)
    sample_weights = season_sample_weights(features["Season"])
    train_mask = features["Season"] <= "2021-22"
    val_mask = features["Season"].isin(["2022-23", "2023-24"])
    test_mask = features["Season"].isin(["2024-25", "2025-26"])
    # If a current-season match log is supplied, it is used for final fitting,
    # not for the historical holdout score until a future season is complete.
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise DataValidationError("Need complete historical data through 2025/26 for the configured chronological split.")
    X_train, y_train, w_train = X[train_mask], y[train_mask], sample_weights[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    poisson_val = poisson_probabilities_from_features(features.loc[val_mask])
    poisson_test = poisson_probabilities_from_features(features.loc[test_mask])

    comparison = []
    candidates = {}
    for name, model in _candidate_models().items():
        fitted = _fit(model, X_train, y_train, w_train)
        val_probs = _predict_probabilities(fitted, X_val)
        alpha, blended, blend_loss = _best_blend_weight(y_val, val_probs, poisson_val)
        row = {"model": name, "validation_samples": int(val_mask.sum()),
               "validation_ml_log_loss": _metrics(y_val, val_probs)["log_loss"],
               "validation_blend_log_loss": blend_loss, "blend_ml_weight": alpha,
               **{f"validation_{k}": v for k, v in _metrics(y_val, blended).items()}}
        comparison.append(row)
        candidates[name] = (fitted, alpha, val_probs)
    poisson_val_metrics = _metrics(y_val, poisson_val)
    comparison.append({"model": "Poisson baseline", "validation_samples": int(val_mask.sum()),
                       "validation_ml_log_loss": None, "validation_blend_log_loss": poisson_val_metrics["log_loss"],
                       "blend_ml_weight": 0.0,
                       **{f"validation_{k}": v for k, v in poisson_val_metrics.items()}})

    selected = min((r for r in comparison if r["model"] != "Poisson baseline"), key=lambda r: r["validation_blend_log_loss"])
    selected_name = selected["model"]
    alpha = float(selected["blend_ml_weight"])
    # Refit selected architecture only on train+validation before touching test.
    selected_test_model = _candidate_models()[selected_name]
    train_val_mask = train_mask | val_mask
    _fit(selected_test_model, X[train_val_mask], y[train_val_mask], sample_weights[train_val_mask])
    test_ml = _predict_probabilities(selected_test_model, X_test)
    test_blend = alpha * test_ml + (1.0 - alpha) * poisson_test
    test_rows = [
        {"model": "Poisson baseline", **_metrics(y_test, poisson_test)},
        {"model": f"{selected_name} + Poisson blend", **_metrics(y_test, test_blend)},
        {"model": selected_name, **_metrics(y_test, test_ml)},
    ]

    # Production fit: all completed seasons, weighted with the agreed recency order.
    final_model = _candidate_models()[selected_name]
    _fit(final_model, X, y, sample_weights)
    metadata = {
        "version": "2.0",
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "model_name": selected_name,
        "ensemble_name": f"{selected_name} + Poisson",
        "ml_probability_weight": alpha,
        "poisson_probability_weight": 1.0 - alpha,
        "feature_columns": FEATURE_COLUMNS,
        "training_matches": int(len(features)),
        "training_start": str(features["Date"].min().date()),
        "training_end": str(features["Date"].max().date()),
        "training_seasons": sorted(features["Season"].astype(str).unique().tolist()),
        "season_sample_weights": SEASON_SAMPLE_WEIGHTS,
        "validation_period": ["2022-23", "2023-24"],
        "test_period": ["2024-25", "2025-26"],
        "validation_best_blend_log_loss": float(selected["validation_blend_log_loss"]),
        "test_metrics": test_rows,
        "historical_current_season_matches": int((features["Season"] == "2026-27").sum()),
        "home_advantage_factor": HOME_ADVANTAGE_FACTOR,
        "disclaimer": "Historical backtest is not proof of 2026/27 performance. The current-season team/FPL adjustment is a new snapshot input and needs prospective validation as 2026/27 match results accrue.",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "metadata": metadata}, MODEL_FILE)
    REPORT_JSON.write_text(json.dumps({"metadata": metadata, "validation_comparison": comparison,
                                      "test_comparison": test_rows}, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(REPORT_CSV, index=False)
    pd.DataFrame(test_rows).to_csv(REPORT_CSV.with_name("model_test_results_v2.csv"), index=False)
    return {"metadata": metadata, "validation_comparison": comparison, "test_comparison": test_rows}


def load_or_train_model() -> dict[str, Any]:
    """Load the saved V2 model, retraining if its data or library versions changed."""
    source_matches = load_training_matches()
    source_count = int(len(source_matches))
    source_end = str(pd.to_datetime(source_matches["Date"]).max().date())
    retrain = not MODEL_FILE.exists()
    bundle = None if retrain else joblib.load(MODEL_FILE)
    if bundle is not None:
        metadata = bundle.get("metadata", {})
        retrain = (
            metadata.get("version") != "2.0"
            or metadata.get("training_matches") != source_count
            or metadata.get("training_end") != source_end
            or metadata.get("sklearn_version") != sklearn.__version__
            or metadata.get("xgboost_version") != xgboost.__version__
        )
    if retrain:
        train_and_save_model()
        bundle = joblib.load(MODEL_FILE)
    return bundle


def build_current_fixture_features(home_team_row: pd.Series, away_team_row: pd.Series,
                                   home_form_row: pd.Series, away_form_row: pd.Series,
                                   h2h_matches: pd.DataFrame,
                                   league_avg_team_goals: float | None = None) -> pd.DataFrame:
    """Translate the live 2026/27 snapshot into the historical model's feature contract."""
    home = str(home_team_row.get("Squad", home_team_row.name))
    away = str(away_team_row.get("Squad", away_team_row.name))
    pair = h2h_matches.loc[
        ((h2h_matches["HomeTeam"] == home) & (h2h_matches["AwayTeam"] == away)) |
        ((h2h_matches["HomeTeam"] == away) & (h2h_matches["AwayTeam"] == home))
    ].sort_values("Date", ascending=False).head(10)
    if pair.empty:
        h2h_ppm, h2h_gd, h2h_count = 1.35, 0.0, 0
    else:
        points = 0.0
        goal_diff = 0.0
        for _, item in pair.iterrows():
            if item["HomeTeam"] == home:
                hg, ag = float(item["FTHG"]), float(item["FTAG"])
            else:
                hg, ag = float(item["FTAG"]), float(item["FTHG"])
            points += 3.0 if hg > ag else 1.0 if hg == ag else 0.0
            goal_diff += hg - ag
        h2h_ppm, h2h_gd, h2h_count = points / len(pair), goal_diff / len(pair), len(pair)

    def numeric(row: pd.Series, key: str, default: float = 0.0) -> float:
        value = pd.to_numeric(row.get(key, default), errors="coerce")
        return default if pd.isna(value) else float(value)

    h_mp, a_mp = max(numeric(home_team_row, "MP", 0.0), 1.0), max(numeric(away_team_row, "MP", 0.0), 1.0)
    all_mp = max(numeric(home_team_row, "MP", 0.0) + numeric(away_team_row, "MP", 0.0), 1.0)
    league_avg = league_avg_team_goals if league_avg_team_goals is not None else (numeric(home_team_row, "GF", 0.0) + numeric(away_team_row, "GF", 0.0)) / all_mp
    features = {
        "home_form5_ppm": numeric(home_form_row, "Last_5_PPM", 1.35),
        "away_form5_ppm": numeric(away_form_row, "Last_5_PPM", 1.35),
        "home_season_ppm": numeric(home_form_row, "Season_PPM", 1.35),
        "away_season_ppm": numeric(away_form_row, "Season_PPM", 1.35),
        "home_season_gf_pm": numeric(home_team_row, "GF_per_match", 1.45),
        "away_season_gf_pm": numeric(away_team_row, "GF_per_match", 1.45),
        "home_season_ga_pm": numeric(home_team_row, "GA_per_match", 1.45),
        "away_season_ga_pm": numeric(away_team_row, "GA_per_match", 1.45),
        "home_season_win_rate": numeric(home_team_row, "W", 0.40) / h_mp,
        "away_season_win_rate": numeric(away_team_row, "W", 0.40) / a_mp,
        "home_matches_played": numeric(home_team_row, "MP", 0.0),
        "away_matches_played": numeric(away_team_row, "MP", 0.0),
        "h2h_home_ppm_10": h2h_ppm,
        "h2h_home_goal_diff_10": h2h_gd,
        "h2h_matches_10": float(h2h_count),
        "league_avg_team_goals": max(league_avg, 0.75),
    }
    return pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def predict_classifier_probabilities(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    return _predict_probabilities(bundle["model"], features[bundle["metadata"]["feature_columns"]].astype(float))
