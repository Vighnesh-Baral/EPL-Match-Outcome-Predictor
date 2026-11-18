# Audit of PitchIQ v2.4 (performed before any v2.5 change)

All findings below were checked against the code or reproduced by running it. Nothing was changed
in v2.4 behaviour until after this audit.

## Reproduction
| Metric (760 test matches, 2024-25 + 2025-26) | README / committed artifact | Re-run, scikit-learn 1.9.1 |
|---|---|---|
| RF + Poisson accuracy | 48.55% | 48.68% |
| RF + Poisson log loss | 1.0237 | 1.0233 |
| RF + Poisson Brier | 0.6150 | 0.6151 |
| RF + Poisson draw recall | 0.5% | 1.0% |

The committed model was built with scikit-learn 1.8.0; v2.4 silently retrains and overwrites its artifact when
library versions differ (this happened during the first `pytest` run). Metrics moved by up to 0.7 pp accuracy and
0.5 pp draw recall. The binomial standard error of accuracy at n=760 is about 1.8 pp, so such shifts are noise,
and so are most differences below roughly 3 pp.

## Findings
1. **Train/serve mismatch (most serious).** The backtested "Poisson" branch is a season-to-date goals-for/against
   model. The Poisson branch that actually serves predictions (`src/predict.py`) is a different model: league goal rate
   times hand-weighted attack, defence, midfield, goalkeeper, form and depth multipliers derived from FBref/FPL
   ratings. That serving branch has **never been backtested**, yet the 70/30 blend weight and home factor were tuned
   on the other one. The reported test metrics therefore do not describe production predictions.
2. **Feature skew.** `league_avg_team_goals` is a cumulative average since 2015 in training (`historical_model.py:153`)
   but the current-season average at serving time (`predict.py`).
3. **Home-advantage factor chosen at the edge of its search grid** (1.06 is the largest value tried), so the optimum was
   not located. It is also applied as a multiplier to home goals *and* a divisor to away goals, so the effective
   home/away ratio is about 1.12, not 1.06 as documented.
4. **No baselines.** v2.4 reported no climatology, always-home or Elo reference. On the same test matches climatology
   scores log loss 1.0834 and always-home 41.7% accuracy, so v2.4's gain over a no-information forecast is 0.060 log
   loss and about 7 pp accuracy.
5. **Draws.** Draw recall is about 1%: draw is almost never the largest class probability. This is expected for
   calibrated football forecasts (the maximum P(draw) is about 0.30) and is not by itself a bug.
6. **Hand-set recency weights** ("agreed hierarchy", 0.25 -> 1.15) rather than tuned; the 2026-27 weight is unused.
7. **Sample size.** 11 match-level 2026/27 results; remaining 2026/27 matches exist only as table aggregates.
8. **Dead / misleading code.** `src/evaluate.py` fails on import (`RANDOM_SEED`, `TEST_SEASONS` are not in config);
   `src/feature_engineering.py` (Elo etc.) is not used by production; four legacy model files in `Models/` are loaded by
   nothing. All retained for provenance and documented, not deleted.
9. **No git history** in the delivered archive (v2.5 starts a repository with v2.4 as the baseline commit).
10. **Test window is not pristine.** 2024-25/2025-26 were used to report v2.4 results and may have informed its design.
    v2.5 keeps the same window for like-for-like comparison but never used it for any decision.

## What v2.4 did correctly
- Same-date leakage prevention in the historical feature builder; features use only earlier results.
- Chronological train / validation / test split with a final test window.
- Log loss and Brier were reported, not just accuracy.
- Probabilities are not forced toward any outcome.
