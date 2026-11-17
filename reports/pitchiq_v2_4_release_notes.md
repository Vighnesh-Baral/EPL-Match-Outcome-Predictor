# PitchIQ Version 2.4 — Release and validation notes

## Delivered

- Matchday Predictor endpoints and UI for matchdays 11–38.
- Premier League Table Predictor using Monte Carlo scoreline simulation for all 280 remaining fixtures.
- Same-date leakage control in historical pre-match features.
- Historical training isolated to the complete 4,180-match 2015/16–2025/26 archive.
- Current standings source refreshed to the supplied Matchday 10 table; advanced team/player snapshot date remains disclosed as 4 November 2026.
- Validation-selected home-goal factor and probability blend saved with the active model bundle.
- Standalone FastAPI pages at `/matchday` and `/table`, plus shortcuts from the root application. React source includes integrated workspace navigation for both pages.

## Historical model protocol

- Train: 2015/16–2021/22.
- Validation: 2022/23–2023/24.
- Untouched test: 2024/25–2025/26 (760 fixtures).
- Classifier selected: Random Forest.
- Production blend: 70% Random Forest / 30% Poisson.
- Selected explicit Poisson home-goal multiplier: 1.06 from candidates 1.00, 1.01, 1.02, 1.03, 1.04, 1.06.
- Selection metric: validation log loss.

## Held-out test results

| Model | Accuracy | Log loss | Brier | Macro F1 | Home recall | Draw recall | Away recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poisson baseline (1.06 reference) | 47.76% | 1.0804 | 0.6397 | 0.3774 | 69.09% | 2.54% | 56.50% |
| Random Forest + Poisson | 48.55% | 1.0237 | 0.6150 | 0.3698 | 74.13% | 0.51% | 54.07% |
| Random Forest | 49.34% | 1.0275 | 0.6165 | 0.3740 | 76.66% | 0.51% | 53.25% |

The blend has slightly lower top-class accuracy than the standalone Random Forest, but better log loss and Brier score. It is retained for probability forecasting and Monte Carlo simulation; this is not an accuracy-maximising claim.

## Draw-class investigation

A draw sample-weight multiplier sweep (1.0–2.0) was evaluated on the validation set. It increased draw recall at higher weights, but consistently reduced accuracy and worsened log loss/Brier performance. A post-prediction draw probability multiplier was similarly tested and degraded the validation probability score. No draw boost was shipped. Draw recall remains a major limitation to address in a later model iteration using stronger pre-match features and carefully separated chronological calibration data.

## Validation status

- Automated tests: 14 passed.
- Python compile check: passed.
- Static JS checks for Version 2.4 matchday/table pages: passed.
- FastAPI route smoke test: root page, predictor pages, CSS, JS and APIs returned HTTP 200; invalid Matchday 10 request correctly returns HTTP 400.
- 10,000-run table simulation: 280 fixtures simulated; 20 clubs returned; title probabilities sum to 100%. The first full forecast completed in about 18 seconds; repeat simulations reuse cached fixture forecasts.
- React production build: not completed in this environment. The npm dependency install was blocked by registry DNS errors, leaving `vite` unavailable. The React source has the new screens, and the FastAPI-served standalone pages provide working access without requiring a frontend rebuild.

## Interpretation and remaining limitations

The historic test period is 2024/25–2025/26. It is not a direct backtest of the 2026/27 snapshot-rating pipeline. Current team/goalkeeper/player adjustments are based on incomplete season snapshots and hand-designed transforms. Matchday fixture dates and kick-off times are not supplied by the current schedule file. The table simulator resolves the principal ranking order by points, goal difference and goals scored, with deterministic ordering for exact residual ties rather than a full head-to-head tie-break simulation.
