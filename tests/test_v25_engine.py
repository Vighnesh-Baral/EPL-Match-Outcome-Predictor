"""PitchIQ v2.5 engine tests: leakage safety, metrics, parsers, models, reproducibility."""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.config import REPORTS_DIR
from src.data_loader import load_historical_matches
from src.engine import metrics as M
from src.engine.dixon_coles import DixonColes, DixonColesParams
from src.engine.features import FEATURE_COLUMNS, FeatureEngine, RatingParams, build_feature_table
from src.engine.models import outcome_probs_from_scores, score_matrix
from src.engine.protocol import TEST_SEASONS, TUNING_SEASONS, SELECTION_SEASONS, rolling_origin


@pytest.fixture(scope="module")
def matches():
    return load_historical_matches()


@pytest.fixture(scope="module")
def table(matches):
    return build_feature_table(matches, RatingParams())[0]


# ----------------------------------------------------------------------------- leakage
def test_features_ignore_own_result_and_future(matches):
    """Changing a match's result (or any later result) must not change that row's features."""
    sub = matches[matches["Season"].isin(["2015-16", "2016-17"])].copy()
    base, _ = build_feature_table(sub, RatingParams())
    k = 500
    altered = sub.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True).copy()
    altered.loc[k:, "FTHG"] = altered.loc[k:, "FTHG"] + 3   # corrupt row k and everything after it
    altered.loc[k:, "FTAG"] = 0
    alt, _ = build_feature_table(altered, RatingParams())
    date_k = base.loc[k, "Date"]
    same_or_earlier = base["Date"] <= date_k           # same-day rows may not see row k either
    pd.testing.assert_frame_equal(base.loc[same_or_earlier, FEATURE_COLUMNS], alt.loc[same_or_earlier, FEATURE_COLUMNS])


def test_features_call_does_not_mutate_state(matches):
    _, eng = build_feature_table(matches[matches["Season"] == "2015-16"], RatingParams())
    before = (dict(eng.elo), dict(eng.att), {k: len(v) for k, v in eng.hist.items()}, eng.n_matches)
    date = pd.Timestamp("2016-08-13")
    f1 = eng.features("Arsenal", "Chelsea", date)
    f2 = eng.features("Arsenal", "Chelsea", date)
    after = (dict(eng.elo), dict(eng.att), {k: len(v) for k, v in eng.hist.items()}, eng.n_matches)
    assert f1 == f2 and before == after


def test_same_day_fixtures_do_not_inform_each_other(matches):
    day = matches[matches["Date"] == matches["Date"].iloc[400]]
    assert len(day) >= 2
    base = matches[matches["Date"] < day["Date"].iloc[0]]
    a, _ = build_feature_table(pd.concat([base, day]), RatingParams())
    reordered = pd.concat([base, day.iloc[::-1]])
    b, _ = build_feature_table(reordered, RatingParams())
    key = ["Date", "HomeTeam", "AwayTeam"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])


def test_feature_table_finite_and_complete(table, matches):
    assert len(table) == len(matches) == 4180
    assert np.isfinite(table[FEATURE_COLUMNS].to_numpy()).all()
    assert set(table["y"].unique()) == {0, 1, 2}


def test_missing_shot_stats_are_skipped_not_imputed():
    eng = FeatureEngine(RatingParams())
    eng.start_season("2026-27", ["A", "B"])
    d0 = pd.Timestamp("2026-08-01")
    eng.update(d0, "A", "B", 1, 0, hst=None, ast=None)
    assert math.isnan(eng.hist["A"][-1]["sot"])
    assert eng.n_sot == 0
    row = eng.features("A", "B", pd.Timestamp("2026-08-08"))
    assert math.isfinite(row["h_sot5"])        # padded with the league prior, documented shrinkage
    eng.update(d0 + pd.Timedelta(days=7), "B", "A", 0, 0, hst=6, ast=2)
    assert eng.n_sot == 2


def test_protocol_windows_are_disjoint_and_ordered():
    assert max(TUNING_SEASONS) < min(SELECTION_SEASONS) < min(TEST_SEASONS)
    assert not set(TUNING_SEASONS) & set(SELECTION_SEASONS) and not set(SELECTION_SEASONS) & set(TEST_SEASONS)


def test_rolling_origin_trains_only_on_the_past(table):
    seen = []

    def fit_predict(train, test, season):
        seen.append((train["Date"].max(), test["Date"].min(), season))
        return np.full((len(test), 3), 1 / 3)

    rolling_origin(table, SELECTION_SEASONS, fit_predict)
    assert all(train_max < test_min for train_max, test_min, _ in seen)


# ----------------------------------------------------------------------------- metrics
def test_log_loss_brier_rps_known_values():
    y = np.array([0, 1, 2])
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    assert M.log_loss(y, p) == pytest.approx(-(math.log(0.5) + math.log(0.6) + math.log(0.8)) / 3)
    assert M.brier(y, p) == pytest.approx(np.mean([0.25 + 0.09 + 0.04, 0.04 + 0.16 + 0.04, 0.01 + 0.01 + 0.04]))
    # RPS of a certain, correct forecast is zero; of a certain home forecast on an away result is 1.
    assert M.rps(np.array([0]), np.array([[1.0, 0.0, 0.0]])) == pytest.approx(0.0, abs=1e-9)
    assert M.rps(np.array([2]), np.array([[1.0, 0.0, 0.0]])) == pytest.approx(1.0, abs=1e-6)


def test_metrics_reject_bad_probabilities():
    with pytest.raises(ValueError):
        M.log_loss(np.array([0]), np.array([[0.5, 0.2, 0.2]]))


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.1, 500)
    out = M.paired_bootstrap(a, a + 0.05, n_boot=500)
    assert out["mean_diff"] < 0 and out["ci95_high"] < 0 and out["prob_a_better"] > 0.99


def test_reliability_table_counts_all_rows():
    rng = np.random.default_rng(2)
    p = rng.dirichlet([2, 1, 2], 400)
    y = rng.integers(0, 3, 400)
    assert sum(r["n"] for r in M.reliability_table(y, p, 1)) == 400


# ----------------------------------------------------------------------------- score model / Dixon-Coles
def test_score_matrix_is_a_distribution_and_dc_keeps_normalisation():
    lam_h, lam_a = np.array([1.6, 0.8]), np.array([1.1, 1.4])
    for rho in (0.0, -0.08, 0.05):
        joint = score_matrix(lam_h, lam_a, rho)
        assert np.allclose(joint.sum(axis=(1, 2)), 1.0)
        assert np.allclose(outcome_probs_from_scores(joint).sum(axis=1), 1.0)


def test_dixon_coles_uses_only_matches_before_as_of_and_ranks_strength(matches):
    as_of = pd.Timestamp("2019-08-01")
    model = DixonColes(DixonColesParams()).fit(matches, as_of)
    assert model.n_fit == int((matches["Date"] < as_of).sum())
    with pytest.raises(ValueError):
        DixonColes(DixonColesParams(min_matches=10 ** 6)).fit(matches, as_of)
    strong = model.predict_probs("Manchester City", "Burnley")
    assert strong[0] > strong[2] and strong.sum() == pytest.approx(1.0)
    assert model.hfa > 0          # home advantage is estimated from data, not hand-set


# ----------------------------------------------------------------------------- production engine
@pytest.fixture(scope="module")
def bundle():
    from src.engine.production import load_or_train_model
    return load_or_train_model()


def test_production_predictions_are_valid_and_unmodified(bundle):
    eng = bundle["engine"]
    out = eng.predict("Arsenal", "Liverpool")
    assert out["probs"].sum() == pytest.approx(1.0)
    assert (out["probs"] > 0).all()
    assert set(out["components"]) == set(bundle["metadata"]["ensemble_members"])
    # ensemble = plain mean of members (no manual draw/home reweighting)
    mean = np.mean([out["components"][n] for n in bundle["metadata"]["ensemble_members"]], axis=0)
    assert np.allclose(out["probs"], mean / mean.sum(), atol=1e-6)
    assert bundle["metadata"]["home_advantage_factor"] is None


def test_home_advantage_is_learned_and_symmetric_fixture_favours_home(bundle):
    eng = bundle["engine"]
    pairs = [("Arsenal", "Chelsea"), ("Liverpool", "Everton"), ("Brighton", "Leeds United"), ("Fulham", "Brentford")]
    diffs = []
    for h, a in pairs:
        fwd, rev = eng.predict(h, a)["probs"], eng.predict(a, h)["probs"]
        diffs.append(fwd[0] - rev[2])         # same pairing, venue swapped
    assert np.mean(diffs) > 0


def test_production_is_deterministic(bundle):
    a = bundle["engine"].predict("Tottenham Hotspur", "West Ham")["probs"]
    b = bundle["engine"].predict("Tottenham Hotspur", "West Ham")["probs"]
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------- reproducibility of the stored report
def test_stored_test_results_match_a_fresh_recomputation(matches, table):
    """The reported v2.5 ensemble numbers must be reproducible from code + data."""
    from src.engine.backtest import MODEL_ORDER, run_models
    from src.engine.ensemble import blend
    from src.engine.protocol import probs_of
    stored = json.loads((REPORTS_DIR / "v25" / "test_results.json").read_text())
    oos = run_models(table, matches, TEST_SEASONS)
    ens = blend([probs_of(oos[n]) for n in MODEL_ORDER], np.full(len(MODEL_ORDER), 1 / len(MODEL_ORDER)))
    y = oos["elo_lr"]["y"].to_numpy()
    key = next(k for k in stored["metrics"] if "production recipe" in k)
    assert M.log_loss(y, ens) == pytest.approx(stored["metrics"][key]["log_loss"], abs=2e-4)
    assert len(y) == stored["n_matches"] == 760
