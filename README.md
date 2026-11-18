# PitchIQ — Premier League Match Predictor (Version 2.5)

PitchIQ estimates Home / Draw / Away probabilities for Premier League fixtures and powers a Match Predictor,
Matchday Predictor, season Table Predictor, Team and Player Intelligence, Head-to-Head and Model Performance views.
It is an independent project, not betting advice.

## Version 2.5 in one paragraph

v2.5 replaces the v2.4 prediction engine with a leakage-safe rating/feature pipeline and a seven-model equal-weight ensemble,
and measures it with rolling chronological backtesting against v2.4 and naive baselines. On the same 760 untouched test matches
the log loss improves from **1.0233 to 1.0091** (Brier 0.6151 -> 0.6046, accuracy 48.7% -> 50.0%). That gap is **not statistically
conclusive** (paired-bootstrap 95% interval [-0.030, +0.002]; accuracy's own 95% interval is about +/-3.5 pp), and draws are still not
predictable. Full tables: `reports/v25/EVALUATION_REPORT.md`. The v2.4 audit is in `reports/v25/AUDIT_V2_4.md`.

## What changed from v2.4

| Area | v2.4 | v2.5 |
|---|---|---|
| Serving vs backtest | Served a hand-weighted, never-backtested Poisson branch; evaluated a different one | One feature engine and one ensemble for both; test report reproducible by a unit test |
| Features | Season-to-date form, H2H, a cumulative league goal rate | Dynamic Elo, online attack/defence ratings, 5/10-match raw and opponent-adjusted form, venue form, shots on target, rest and fixture load, H2H (54 features, 9 groups) |
| Models | Random Forest + Poisson blend | Elo logistic, core logistic, full logistic, Random Forest, XGBoost, home/away Poisson goal models, Dixon-Coles (all compared) |
| Home advantage | Manual factor 1.06 (grid edge) | Learned from results; no manual factor anywhere |
| Draws | Never predicted | Never forced; reported honestly (see below) |
| Tuning / selection | One validation window | Tuning window, separate selection window with nested season-swap, frozen before a single final test |
| Data | 11 match-level 2026/27 rows, no fixture dates | + OpenFootball (cross-validated 4,180/4,180), 2026/27 fixture dates and kick-offs, 60 known 2026/27 results, StatsBomb 2015/16 xG for validation |
| Missing statistics | n/a | Never imputed: NaN is skipped, then padded with a documented league prior |

## Model architecture (`src/engine/`)

- `features.py` — `FeatureEngine`: one stateful, strictly pre-match engine used for training rows *and* live fixtures.
  Fixtures on a date are featurised before that date's results are applied.
- `dixon_coles.py` — time-decayed Dixon-Coles (analytic-gradient MLE, ridge shrinkage, profile-estimated low-score correlation); the
  rolling harness refits at most every 14 days using only earlier matches.
- `models.py` — classifier wrappers, home/away Poisson goal models with a Dixon-Coles score grid.
- `protocol.py`, `backtest.py`, `tune.py`, `select.py`, `final_eval.py`, `report.py` — chronological protocol and its reproducible stages.
- `production.py` — trains the frozen recipe and serves predictions. `v24_reference.py` rebuilds the v2.4 model for comparison.

### Evaluation protocol
| Window | Seasons | Purpose |
|---|---|---|
| Warm-up | 2015-16 | initialise ratings only |
| Tuning | 2018-19 to 2021-22 | hyper-parameters, feature-group selection (rolling-origin) |
| Selection | 2022-23, 2023-24 | ensemble recipe and calibration decision, nested season swap |
| Final test | 2024-25, 2025-26 | scored once, after everything was frozen |

### Headline test results (760 matches; lower is better except accuracy and macro-F1)
| Model | Accuracy | Log loss | Brier | RPS | Draw recall |
|---|---|---|---|---|---|
| Climatology baseline | 41.7% | 1.0834 | 0.6564 | 0.2317 | 0.0% |
| v2.4 RF + Poisson (as shipped) | 48.7% | 1.0233 | 0.6151 | 0.2112 | 1.0% |
| **v2.5 equal-weight ensemble (production)** | **50.0%** | **1.0091** | **0.6046** | **0.2060** | 0.0% |
| v2.5 Dixon-Coles | 49.5% | 1.0148 | 0.6085 | 0.2086 | 0.0% |
| v2.5 home/away Poisson goals | 50.3% | 1.0029 | 0.6005 | 0.2045 | 0.0% |

Single-model results are not significantly different from each other. The goal-Poisson model scored best on the test window, but the ensemble recipe had
been frozen on the selection window and was kept: changing it now would be tuning on the test set.

### What the experiments found
- **Team-strength ratings carry almost all the signal.** Adding rolling form, shots, schedule and H2H groups did not improve out-of-sample log loss;
  heavy regularisation or ensembling was needed so they did not hurt (`reports/v25/tuning_results.json`).
- **Draws are not separable from this data.** Mean P(draw) is 0.235 on draws vs 0.231 on non-draws. Argmax draw recall is therefore ~0% for v2.5 and ~1% for v2.4.
  Probabilities were not inflated to hide this; doing so would worsen log loss and calibration without new information.
- **Calibration.** Temperature scaling was tested on the selection window and not applied (it hurt out of sample).
- **Shots on target are a defensible xG stand-in:** they explain ~52% of real xG variance (StatsBomb 2015/16) and predict next-match goals as well as rolling xG.

## Current-season data and limitations

- 2026/27: 60 of the 100 played matches are known at match level (OpenFootball Matchdays 1-5 + the repository log). The other 40 are **not imputed**; the
  Matchday-10 standings remain the table baseline. Rolling-form features for current-season matches can therefore be a few matches stale.
- **Player-strength features are not part of v2.5 predictions.** Historical per-player data (the MIT-licensed FPL archive) was not integrated, so such features
  could not be validated on past matches. FPL/FBref squad ratings are still displayed in Team and Player Intelligence, but they no longer influence probabilities.
  This is the main unfinished item; the ablation harness in `tune.py` is ready to evaluate it.
- Football-Data.co.uk and FPL live downloads are implemented and tested with mocked responses, but **could not be run live** from the build environment. No bookmaker-odds
  benchmark is reported because the shipped CSVs contain no odds.
- Fixture congestion counts league matches only. No lineup, injury or suspension data is available from the free sources.
- The test window was also used to report v2.4, so it is not pristine with respect to v2.4 development; v2.5 did not use it for any decision.
- With ~760 test matches, differences below roughly 3 pp accuracy or 0.02 log loss should not be over-interpreted.
- `Data/Raw/2026_27/` FBref/FPL snapshots remain at 4 November 2026. Fixture dates come from the cached OpenFootball schedule and must be refreshed on reschedules.

## Reproduce the pipeline

```powershell
python -m src.ingest openfootball --seasons 2026-27     # refresh schedule/results
python -m src.engine.tune                                # tuning window  -> reports/v25/tuning_results.json
python -m src.engine.select                              # selection window -> reports/v25/selection.json
python -m src.engine.final_eval                          # one-shot final test -> reports/v25/test_results.json
python -m src.engine.report                              # renders reports/v25/EVALUATION_REPORT.md
python -m src.engine.xg_validation                       # shots vs real xG (needs Data/Processed/statsbomb_pl_2015_16_xg.csv)
python -m src.train                                      # refit production model -> Models/pitchiq_v25.joblib
```

Datasets, licences and coverage: `DATA_SOURCES.md`. StatsBomb Open Data requires attribution: *Data provided by StatsBomb.*

## New prediction modules

### Matchday Predictor

Select a matchday from 11 to 38. PitchIQ scores every fixture in that round with the same `/api/predict` engine and returns H/D/A probabilities, expected goals, the most likely outcome, a round summary, the highest-confidence fixture and the closest fixture.

### Premier League Table Predictor

The table simulator begins with the supplied Matchday 10 standings and simulates all 280 outstanding fixtures. Each fixture's scorelines are sampled from Poisson goal distributions reweighted so that the simulated H/D/A marginal probabilities match PitchIQ's final match probabilities. This permits goal difference and goals scored to affect final rankings.

The default is 10,000 simulations; the API accepts 1,000–50,000. The 280 fixture predictions are cached in-process and reused when the simulation count changes. Outputs include expected points/records/goals, average finishing position, 10th–90th percentile finishing range, title probability, top-four/top-six probability, relegation probability and position distributions.

The projected table is ordered by expected points, expected goal difference and expected goals scored. The Premier League can use head-to-head criteria for certain exact ties; the current simulator does not resolve those rare residual ties through head-to-head mini-tables and instead uses deterministic club ordering. Interpret projections as distributions, not guaranteed final positions.

## Run locally on Windows

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
python run.py
```

Open `http://127.0.0.1:8000`. Stop the server with `Ctrl+C`.

The application loads `Models/pitchiq_v25.joblib`; if it is missing, corrupt, or its data/library metadata is stale, the API retrains it (about 7 seconds) from the historical archive. To rebuild it explicitly, run `python -m src.train`.

## Frontend

The React + Vite source is in `frontend/`. To run the development frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`; the Vite dev server proxies `/api` to FastAPI at `http://127.0.0.1:8000`.

To rebuild the integrated React workspace for FastAPI:

```powershell
npm run build
```

The production bundle is written to `app/static/`. The Matchday and Table prediction modules are also served directly by FastAPI at `/matchday` and `/table`, and the existing root page has shortcuts to them. These static module pages use the same API endpoints as the React workspace.

## API endpoints

- `GET /api/health`
- `GET /api/overview`
- `GET /api/teams`
- `GET /api/teams/{team_name}`
- `POST /api/predict`
- `GET /api/h2h?home_team=Arsenal&away_team=Liverpool&limit=10`
- `GET /api/matchdays`
- `GET /api/matchday/{matchday}` (11–38)
- `GET /api/table-prediction?simulations=10000` (1,000–50,000)
- `GET /docs`

Prediction request example:

```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool"
}
```

## Tests

```powershell
pytest -q
python -m compileall -q app src tests
```

Tests cover input integrity, leakage safety of the feature engine (a result never changes its own or earlier features; same-day fixtures never inform each other), chronological protocol boundaries, metric correctness, Dixon-Coles fit boundaries, parser strictness and OpenFootball/Football-Data agreement, mocked ingestion for Football-Data/FPL/StatsBomb, unmodified-ensemble and learned-home-advantage checks, reproducibility of the stored test report, probability sums, matchday fixture counts and the 20-team/38-match season simulation.

## Photography and attribution

Football photographs are sourced from Franco Monsalvo's Pexels pages. Source links and licence notes are recorded in `ASSET_CREDITS.md` and `frontend/README.md`. The images are served from the Pexels CDN, so an internet connection is required to display them. PitchIQ is an independent project and is not affiliated with the Premier League or any club.
