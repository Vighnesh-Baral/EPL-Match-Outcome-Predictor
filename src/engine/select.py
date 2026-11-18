"""Ensemble / calibration selection on the SELECTION window only (2022-23, 2023-24).

Methods are compared with a nested season-swap (fit on one season, score on the other),
so the chosen method is judged out-of-sample even inside the selection window.
Run:  python -m src.engine.select   (writes reports/v25/selection.json)
"""
from __future__ import annotations

import json

import numpy as np

from src.config import REPORTS_DIR
from src.data_loader import load_historical_matches
from src.engine.backtest import MODEL_ORDER, run_models
from src.engine.ensemble import blend, fit_temperature, fit_weights, temperature
from src.engine.features import RatingParams, build_feature_table
from src.engine.metrics import all_metrics, log_loss
from src.engine.protocol import SELECTION_SEASONS, probs_of

SHRINKS = (0.0, 0.02, 0.1)


def run() -> dict:
    matches = load_historical_matches()
    table, _ = build_feature_table(matches, RatingParams())
    oos = run_models(table, matches, SELECTION_SEASONS)
    ref = oos[MODEL_ORDER[0]]
    y = ref["y"].to_numpy()
    seasons = ref["Season"].to_numpy()
    P = {n: probs_of(oos[n]) for n in MODEL_ORDER}
    result: dict = {"single_models": {n: all_metrics(y, P[n]) for n in MODEL_ORDER}}

    def methods(idx_fit):
        out = {"equal_all": np.full(len(MODEL_ORDER), 1 / len(MODEL_ORDER))}
        for s in SHRINKS:
            out[f"optimised_shrink_{s}"] = fit_weights([P[n][idx_fit] for n in MODEL_ORDER], y[idx_fit], s)
        return out

    nested = {}
    for fit_season, eval_season in (("2022-23", "2023-24"), ("2023-24", "2022-23")):
        fit, ev = seasons == fit_season, seasons == eval_season
        for name, w in methods(fit).items():
            nested.setdefault(name, []).append(log_loss(y[ev], blend([P[n][ev] for n in MODEL_ORDER], w)))
        best_single = min(MODEL_ORDER, key=lambda n: log_loss(y[fit], P[n][fit]))
        nested.setdefault("best_single_by_fit_season", []).append(log_loss(y[ev], P[best_single][ev]))
    result["nested_season_swap_log_loss"] = {k: float(np.mean(v)) for k, v in nested.items()}
    chosen = min((k for k in nested if k != "best_single_by_fit_season"), key=lambda k: np.mean(nested[k]))
    result["chosen_method"] = chosen

    all_idx = np.ones(len(y), bool)
    weights = methods(all_idx)[chosen]
    ens = blend([P[n] for n in MODEL_ORDER], weights)
    result["weights"] = dict(zip(MODEL_ORDER, map(float, weights)))
    result["ensemble_selection_metrics"] = all_metrics(y, ens)

    # temperature scaling, nested the same way
    gains = []
    for fit_season, eval_season in (("2022-23", "2023-24"), ("2023-24", "2022-23")):
        fit, ev = seasons == fit_season, seasons == eval_season
        w = methods(fit)[chosen]
        e_fit = blend([P[n][fit] for n in MODEL_ORDER], w)
        T = fit_temperature(e_fit, y[fit])
        e_ev = blend([P[n][ev] for n in MODEL_ORDER], w)
        gains.append(log_loss(y[ev], e_ev) - log_loss(y[ev], temperature(e_ev, T)))
    T_all = fit_temperature(ens, y)
    result["temperature"] = {"fitted_T": T_all, "nested_mean_gain": float(np.mean(gains)),
                             "apply": bool(np.mean(gains) > 0.001)}
    (REPORTS_DIR / "v25").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "v25" / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    print("nested:", {k: round(v, 4) for k, v in r["nested_season_swap_log_loss"].items()})
    print("chosen:", r["chosen_method"], {k: round(v, 3) for k, v in r["weights"].items()})
    print("ensemble sel ll:", round(r["ensemble_selection_metrics"]["log_loss"], 4), "| temperature:", r["temperature"])
