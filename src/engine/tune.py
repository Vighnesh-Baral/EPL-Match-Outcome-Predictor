"""Hyper-parameter tuning on the TUNING window only (2018-19 .. 2021-22).

Selection/ensemble weights are fitted later on a disjoint window, and the final test
window is never read here.  Run:  python -m src.engine.tune
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.data_loader import load_historical_matches
from src.engine.dixon_coles import DixonColesParams, rolling_dixon_coles
from src.engine.features import FEATURE_COLUMNS, FEATURE_GROUPS, RatingParams, build_feature_table
from src.engine.metrics import log_loss
from src.engine.models import ClassifierModel, SeparateGoalModels, logistic_factory, rf_factory, xgb_factory
from src.engine.protocol import TUNING_SEASONS, probs_of, recency_weights, rolling_origin, training_rows

OUT_DIR = REPORTS_DIR / "v25"


def _clf_runner(name, factory, columns, half_life):
    def fit_predict(train, test, season):
        w = np.ones(len(train)) if half_life is None else \
            0.5 ** ((int(season[:4]) - train["Season"].str[:4].astype(int).to_numpy() - 1) / half_life)
        model = ClassifierModel(name, factory, columns).fit(train, train["y"].to_numpy(), w)
        return model.predict_proba(test)
    return fit_predict


def _goal_runner(columns, alpha, half_life):
    def fit_predict(train, test, season):
        w = np.ones(len(train)) if half_life is None else \
            0.5 ** ((int(season[:4]) - train["Season"].str[:4].astype(int).to_numpy() - 1) / half_life)
        model = SeparateGoalModels(columns, alpha=alpha).fit(train, train["FTHG"].to_numpy(), train["FTAG"].to_numpy(), w)
        return model.predict_proba(test)
    return fit_predict


def score(table, runner) -> float:
    oos = rolling_origin(table, TUNING_SEASONS, runner)
    return log_loss(oos["y"].to_numpy(), probs_of(oos))


def run() -> dict:
    t0 = time.time()
    matches = load_historical_matches()
    table, _ = build_feature_table(matches, RatingParams())
    results: dict = {"window": list(TUNING_SEASONS), "rating_params": RatingParams().to_dict()}
    tune_rows = table[table["Season"].isin(TUNING_SEASONS)]
    y = tune_rows["y"].to_numpy()
    # climatology baseline measured the same rolling way
    def clim(train, test, season):
        freq = np.bincount(train["y"], minlength=3) / len(train)
        return np.tile(freq, (len(test), 1))
    results["climatology"] = score(table, clim)

    # Elo-only logistic
    results["elo_only_lr"] = score(table, _clf_runner("elo_lr", logistic_factory(1.0), ["elo_diff"], None))

    # Logistic regression grid: C x recency half-life
    lr_grid = []
    for C, hl in itertools.product([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], [None, 4.0, 8.0]):
        lr_grid.append({"C": C, "half_life": hl, "log_loss": score(table, _clf_runner("lr", logistic_factory(C), FEATURE_COLUMNS, hl))})
    results["lr_grid"] = lr_grid
    best_lr = min(lr_grid, key=lambda r: r["log_loss"])
    results["lr_best"] = best_lr

    # Feature-group ablation with the best LR (drop one group at a time)
    ablation = {"all_features": best_lr["log_loss"]}
    for group, cols in FEATURE_GROUPS.items():
        kept = [c for c in FEATURE_COLUMNS if c not in cols]
        ablation[f"without_{group}"] = score(table, _clf_runner("lr", logistic_factory(best_lr["C"]), kept, best_lr["half_life"]))
    for group, cols in FEATURE_GROUPS.items():
        ablation[f"only_{group}"] = score(table, _clf_runner("lr", logistic_factory(best_lr["C"]), cols, best_lr["half_life"]))
    results["lr_ablation"] = ablation

    # Greedy forward selection of feature groups (tuning window only; C fixed at 0.03)
    def lr_groups_score(groups, C=0.03):
        cols = sum([FEATURE_GROUPS[g] for g in groups], [])
        return score(table, _clf_runner("lr", logistic_factory(C), cols, 8.0))
    chosen, current = ["goal_ratings"], lr_groups_score(["goal_ratings"])
    trace = [{"groups": list(chosen), "log_loss": current}]
    while True:
        trials = [(lr_groups_score(chosen + [g]), g) for g in FEATURE_GROUPS if g not in chosen]
        if not trials:
            break
        best_score, best_group = min(trials)
        if current - best_score > 0.0005:      # require a non-trivial gain
            chosen.append(best_group)
            current = best_score
            trace.append({"groups": list(chosen), "log_loss": current})
        else:
            break
    core_grid = [{"C": C, "log_loss": lr_groups_score(chosen, C)} for C in (0.01, 0.03, 0.1, 0.3)]
    results["lr_core"] = {"groups": chosen, "trace": trace, "grid": core_grid, "best": min(core_grid, key=lambda r: r["log_loss"])}

    # Random forest and XGBoost (small, strongly regularised grids)
    rf_grid = []
    for depth, leaf in itertools.product([3, 4, 6], [10, 20, 40, 80]):
        rf_grid.append({"max_depth": depth, "min_samples_leaf": leaf,
                        "log_loss": score(table, _clf_runner("rf", rf_factory(250, depth, leaf), FEATURE_COLUMNS, 8.0))})
    results["rf_grid"] = rf_grid
    results["rf_best"] = min(rf_grid, key=lambda r: r["log_loss"])
    xgb_grid = []
    for depth, n_est in itertools.product([1, 2, 3], [60, 120, 250]):
        xgb_grid.append({"max_depth": depth, "n_estimators": n_est,
                         "log_loss": score(table, _clf_runner("xgb", xgb_factory(n_est, depth), FEATURE_COLUMNS, 8.0))})
    results["xgb_grid"] = xgb_grid
    results["xgb_best"] = min(xgb_grid, key=lambda r: r["log_loss"])

    # Separate home/away Poisson goal models
    goal_grid = []
    for alpha, hl in itertools.product([0.5, 1.0, 2.0, 5.0, 20.0, 80.0], [None, 8.0]):
        goal_grid.append({"alpha": alpha, "half_life": hl, "log_loss": score(table, _goal_runner(FEATURE_COLUMNS, alpha, hl))})
    results["goal_grid"] = goal_grid
    results["goal_best"] = min(goal_grid, key=lambda r: r["log_loss"])

    # Dixon-Coles
    dc_grid = []
    for half_life, ridge in itertools.product([275, 550, 1100], [0.5, 2.0, 8.0]):
        p = rolling_dixon_coles(matches, tune_rows, DixonColesParams(half_life_days=half_life, ridge=ridge, refit_days=28))
        dc_grid.append({"half_life_days": half_life, "ridge": ridge, "log_loss": log_loss(y, p)})
    results["dc_grid"] = dc_grid
    results["dc_best"] = min(dc_grid, key=lambda r: r["log_loss"])

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tuning_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = run()
    for key in ("climatology", "elo_only_lr", "lr_best", "lr_core", "rf_best", "xgb_best", "goal_best", "dc_best", "elapsed_seconds"):
        print(key, out[key])
    print("ablation:", {k: round(v, 4) for k, v in out["lr_ablation"].items()})
