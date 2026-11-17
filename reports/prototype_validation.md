# Version 2 validation report

Generated from the supplied prototype archive on 2026-11-05.

## Input coverage

- Current 2026/27 league table and FBref team-stat files: 20 clubs.
- Current FPL roster/stat snapshot: 670 players.
- 2026/27 aggregate team table: 9 matches played per club in the supplied snapshot.
- Historical match results: 4,180 completed fixtures across 2015/16–2025/26.
- No fixture-by-fixture 2026/27 results file was present in the uploaded archive.

## Season weighting

Initial fit weights follow the requested recency hierarchy, from 0.25 for 2015/16 up to 1.00 for 2025/26. A future 2026/27 result file receives weight 1.15. These values encode priority; they are not mathematically proven optima.

## Chronological evaluation

- Training: 2015/16–2021/22 (2,660 matches).
- Validation/model selection: 2022/23–2023/24 (760 matches).
- Untouched test: 2024/25–2025/26 (760 matches).

### Validation results

| Candidate | Validation log loss | Validation accuracy | ML share of selected blend |
|---|---:|---:|---:|
| Logistic Regression + Poisson | 0.9748 | 53.55% | 80% |
| Random Forest + Poisson | **0.9713** | 55.13% | **70%** |
| XGBoost + Poisson | 0.9901 | 51.97% | 45% |
| Poisson alone | 1.0179 | 51.97% | 0% |

Random Forest + Poisson was selected by validation log loss.

### Untouched test results

| Candidate | Test log loss | Test Brier score | Test accuracy |
|---|---:|---:|---:|
| Poisson alone | 1.0804 | 0.6397 | 47.76% |
| Random Forest alone | 1.0275 | 0.6165 | 49.34% |
| Random Forest + Poisson | **1.0237** | **0.6150** | 48.55% |

The blended model had the best test log loss and Brier score in this comparison; Random Forest alone had marginally higher test accuracy. This is why probability quality should not be judged by accuracy alone.

## Runtime checks

- All 8 automated tests passed.
- FastAPI smoke tests passed for `/`, `/api/health`, `/api/overview`, `/api/teams`, `/api/h2h`, `/api/predict` and `/docs`.
- All 380 ordered home/away pairs among the 20 current clubs were predicted.
- Each probability triplet was finite, between 0 and 100, and summed to 100.00 after display rounding.
- Same-date fixtures are featurised before any result from that date is added to team/H2H history.

## Interpretation and limitations

1. This is a historical backtest plus a current 2026/27 snapshot adjustment. It is not yet a prospective 2026/27 evaluation.
2. The 2026/27 result file is absent. Current 2026/27 data influences the team ratings and Poisson branch, but it is not part of supervised classifier training yet.
3. The test window is only two seasons and contains a limited number of fixtures; reported metrics can vary by period and competition environment.
4. Sample weights express the user's requested priority order but were not exhaustively tuned. Treat them as transparent initial settings.
5. The Poisson branch uses current team ratings and a fixed 1.06 home factor. Those goal multipliers are explicit assumptions, not fitted on 2026/27 results.
6. Current FPL metrics are not complete player scouting data: individual minutes, starts, xG, goals, assists and defensive actions are absent from the supplied snapshot.
7. Match predictions are probabilistic estimates, not guarantees.
