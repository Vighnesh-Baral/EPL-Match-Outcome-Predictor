"""One-shot final evaluation on the untouched test window (2024-25, 2025-26).

All hyper-parameters (tune.py) and the ensemble recipe (select.py) are frozen before this
runs.  Run:  python -m src.engine.final_eval
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.data_loader import load_historical_matches
from src.engine.backtest import MODEL_ORDER, always_home_runner, climatology_runner, run_models
from src.engine.ensemble import blend
from src.engine.features import RatingParams, build_feature_table
from src.engine.metrics import (all_metrics, bootstrap_ci, per_match_log_loss, paired_bootstrap, reliability_table)
from src.engine.protocol import TEST_SEASONS, probs_of, rolling_origin
from src.engine.v24_reference import v24_features, v24_predict

OUT = REPORTS_DIR / "v25"


def _align_v24(matches: pd.DataFrame, test_rows: pd.DataFrame, rolling: bool) -> np.ndarray:
    feats = v24_features(matches)
    key = feats["Date"].astype(str) + "|" + feats["HomeTeam"] + "|" + feats["AwayTeam"]
    parts = []
    for season in TEST_SEASONS:
        cutoff = "2024-25" if (not rolling or season == "2024-25") else season
        train = feats["Season"] < cutoff
        test = feats["Season"] == season
        probs = v24_predict(feats, train, test)["blend"]
        frame = pd.DataFrame(probs, index=key[test].to_numpy())
        parts.append(frame)
    lookup = pd.concat(parts)
    want = test_rows["Date"].astype(str) + "|" + test_rows["HomeTeam"] + "|" + test_rows["AwayTeam"]
    return lookup.loc[want.to_numpy()].to_numpy()


def run() -> dict:
    matches = load_historical_matches()
    table, _ = build_feature_table(matches, RatingParams())
    ref_oos = {}
    oos = run_models(table, matches, TEST_SEASONS)
    base = rolling_origin(table, TEST_SEASONS, climatology_runner)
    y = base["y"].to_numpy()
    seasons = base["Season"].to_numpy()
    P = {n: probs_of(oos[n]) for n in MODEL_ORDER}
    ensemble = blend([P[n] for n in MODEL_ORDER], np.full(len(MODEL_ORDER), 1 / len(MODEL_ORDER)))
    fixed = run_models(table, matches, TEST_SEASONS, fixed_origin_before="2024-25")
    ens_fixed = blend([probs_of(fixed[n]) for n in MODEL_ORDER], np.full(len(MODEL_ORDER), 1 / len(MODEL_ORDER)))
    v24_fixed = _align_v24(matches, base, rolling=False)
    v24_roll = _align_v24(matches, base, rolling=True)
    always = rolling_origin(table, TEST_SEASONS, always_home_runner)

    entries = {
        "Climatology (train class frequencies)": probs_of(base),
        "Always-home (accuracy reference)": probs_of(always),
        "v2.4 RF+Poisson (as shipped, fixed origin)": v24_fixed,
        "v2.4 RF+Poisson (refit each season)": v24_roll,
        **{f"v2.5 {n}": P[n] for n in MODEL_ORDER},
        "v2.5 ensemble (rolling refit)  <- production recipe": ensemble,
        "v2.5 ensemble (fixed origin, v2.4-equivalent protocol)": ens_fixed,
    }
    result: dict = {"test_seasons": list(TEST_SEASONS), "n_matches": int(len(y)),
                    "metrics": {k: all_metrics(y, v) for k, v in entries.items()}, "per_season": {}, "paired_vs_v24": {}}
    for season in TEST_SEASONS:
        idx = seasons == season
        result["per_season"][season] = {k: all_metrics(y[idx], v[idx]) for k, v in entries.items()
                                         if k.startswith(("v2.4", "v2.5 ensemble", "Climatology"))}
    ll = {k: per_match_log_loss(y, v) for k, v in entries.items()}
    for ours in ("v2.5 ensemble (fixed origin, v2.4-equivalent protocol)", "v2.5 ensemble (rolling refit)  <- production recipe"):
        for theirs in ("v2.4 RF+Poisson (as shipped, fixed origin)", "v2.4 RF+Poisson (refit each season)"):
            result["paired_vs_v24"][f"{ours} vs {theirs}"] = paired_bootstrap(ll[ours], ll[theirs])
    ens_name = "v2.5 ensemble (rolling refit)  <- production recipe"
    result["accuracy_ci95_ensemble"] = bootstrap_ci((ensemble.argmax(1) == y).astype(float))
    result["accuracy_ci95_v24"] = bootstrap_ci((v24_fixed.argmax(1) == y).astype(float))
    result["reliability"] = {c: reliability_table(y, ensemble, i) for i, c in enumerate(("home", "draw", "away"))}
    result["reliability_v24"] = {c: reliability_table(y, v24_fixed, i) for i, c in enumerate(("home", "draw", "away"))}
    # draw diagnostics (no probability manipulation): how well does P(draw) separate draws?
    pd_draw = ensemble[:, 1]
    result["draw_diagnostics"] = {
        "max_p_draw": float(pd_draw.max()), "min_p_draw": float(pd_draw.min()),
        "mean_p_draw_when_draw": float(pd_draw[y == 1].mean()), "mean_p_draw_when_not_draw": float(pd_draw[y != 1].mean()),
        "share_of_matches_p_draw_ge_0.30": float((pd_draw >= 0.30).mean()),
        "observed_draw_rate_when_p_draw_ge_0.30": float((y[pd_draw >= 0.30] == 1).mean()) if (pd_draw >= 0.30).any() else None,
        "observed_draw_rate_when_p_draw_lt_0.24": float((y[pd_draw < 0.24] == 1).mean()) if (pd_draw < 0.24).any() else None,
        "v24_mean_p_draw_when_draw": float(v24_fixed[y == 1, 1].mean()),
        "v24_mean_p_draw_when_not_draw": float(v24_fixed[y != 1, 1].mean()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    preds = base[["Date", "Season", "HomeTeam", "AwayTeam", "y"]].copy()
    for tag, arr in (("v25", ensemble), ("v24", v24_fixed)):
        preds[[f"{tag}_pH", f"{tag}_pD", f"{tag}_pA"]] = arr
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    return result


if __name__ == "__main__":
    r = run()
    for k, m in r["metrics"].items():
        print(f"{k[:58]:58s} acc {m['accuracy']:.4f} ll {m['log_loss']:.4f} brier {m['brier']:.4f} rps {m['rps']:.4f} f1 {m['macro_f1']:.3f} drawR {m['draw_recall']:.3f}")
    print(json.dumps(r["paired_vs_v24"], indent=1))
    print("acc CI v25", r["accuracy_ci95_ensemble"], "v24", r["accuracy_ci95_v24"])
    print(json.dumps(r["draw_diagnostics"], indent=1))
