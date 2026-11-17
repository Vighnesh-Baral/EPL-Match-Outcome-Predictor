# PitchIQ — Premier League Match Predictor (Version 2.4)

PitchIQ is an independent 2026/27 Premier League forecasting workspace. Version 2.4 combines a chronologically evaluated historical classifier with a current-season Poisson goal model, and adds full-round matchday predictions plus Monte Carlo end-of-season table forecasting.

> **Important:** A backtest is not a guarantee of future performance. The Version 2.4 historical holdout accuracy is approximately 48.6% for the selected probability blend, and draw classification remains a material weakness. The project reports these limitations rather than presenting predictions as certainties.

## Version 2.4 highlights

- Rebuilt and versioned the active historical model using complete 2015/16–2025/26 match archives.
- Selected the classifier/Poisson probability blend and explicit Poisson home-goal factor using a chronological validation set and log loss.
- Excluded the incomplete 2026/27 match-results file from supervised model fitting; it remains available for head-to-head context.
- Refreshed the current standings input through Matchday 10 while disclosing that advanced FBref/FPL inputs remain at the 4 November 2026 snapshot.
- Added **Matchday Predictor** for every scheduled round from Matchday 11 to Matchday 38.
- Added **Table Predictor**, a 10,000-run default Monte Carlo simulation of the remaining 280 fixtures, with projected standings, title/top-four/top-six/relegation probabilities and finishing-position ranges.
- Exposed model metadata, probability components and the data limitations in the API and interface.

## Model architecture

### Historical classifier

The active production classifier is a recency-weighted Random Forest trained on 4,180 complete historical matches from 2015/16 through 2025/26. Match-level features are reconstructed chronologically using only information available before each fixture. All fixtures on the same date are featurised before any results from that date are added to team or head-to-head history.

Features include:

- Last-five form points per match.
- Season-to-date points, goals for, goals against and win rate.
- Matches played before the fixture.
- The previous ten head-to-head meetings, where available.
- A league goal-rate reference.

### Probability blend and home advantage

The selected production blend is **Random Forest + Poisson**, with the current saved version selecting a 70% historical-classifier / 30% Poisson blend. The explicit Poisson home-goal multiplier selected by the validation search is 1.06. Candidate values tested were 1.00, 1.01, 1.02, 1.03, 1.04 and 1.06.

This factor is a multiplier on expected goals in the Poisson branch; it is not a 6-percentage-point addition to home-win probability. The historical classifier also learns home/away differences from its training data, so the validation search evaluates the combined effect rather than assuming that a lower multiplier is automatically better.

### Chronological evaluation

- Training window for model selection: 2015/16–2021/22.
- Validation/model selection: 2022/23–2023/24.
- Untouched historical test: 2024/25–2025/26 (760 fixtures).
- Final production classifier is refitted on the complete 2015/16–2025/26 archive after the validation and test protocol.

The selected blend minimises validation log loss. On the held-out test period it recorded:

| Metric | Random Forest + Poisson |
|---|---:|
| Accuracy | 48.55% |
| Log loss | 1.0237 |
| Multiclass Brier score | 0.6150 |
| Macro F1 | 0.3698 |
| Home recall | 74.13% |
| Draw recall | 0.51% |
| Away recall | 54.07% |

The standalone Random Forest recorded 49.34% accuracy, 1.0275 log loss and 0.6165 Brier score on the same test. The blend was retained because its probability scores (log loss and Brier) were better, even though its top-class accuracy was slightly lower. This is a deliberate probability-quality trade-off, not a claim that the blend maximises accuracy.

**Draw limitation:** the model very rarely selects Draw as the single most likely class. A draw-class weighting / probability boost was tested on validation data; it increased draw recall but reduced accuracy and worsened log loss. It was therefore not applied to the production probabilities. Draw calibration needs more research and prospective results.

Evaluation files are saved in `reports/model_evaluation_v2.json`, `reports/model_comparison_v2.csv` and `reports/model_test_results_v2.csv`.

## Current-season data and limitations

- `Data/Raw/2026_27/current_standings_2026_27.csv` contains the current standings through Matchday 10 and is used as the current league-table baseline.
- Advanced FBref team metrics and FPL player inputs remain at the **4 November 2026** snapshot. They predate the Matchday 10 results and are not retroactively refreshed.
- `Data/Raw/2026_27/match_results_2026_27.csv` is only a partial results log (11 verified rows in this delivery). It is used for H2H context, not supervised model training.
- The current FPL snapshot does not provide complete player minutes, starts, xG, assists, defensive actions, injuries or suspensions. FPL-based values are contribution/depth proxies, not complete player ability estimates or confirmed line-ups.
- Team-rating component weights and the translations from current ratings to expected goals remain transparent heuristics. The historical backtest does not validate every current-season rating adjustment.
- The fixture schedule contains Matchday and home/away assignments but not confirmed dates or kick-off times. It must be refreshed when fixtures are rescheduled.

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

The application loads `Models/historical_predictor_v2.joblib`; if it is missing or its data/library metadata is stale, the API retrains it from the historical archive. To explicitly rebuild the model and evaluation reports, run `python -m src.train`.

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

The production bundle is written to `app/static/`. The two Version 2.4 prediction modules are also served directly by FastAPI at `/matchday` and `/table`, and the existing root page has shortcuts to them. These static module pages use the same API endpoints as the React workspace.

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

Tests cover input integrity, same-date leakage prevention, historical training boundaries, probability sums, matchday fixture counts and the 20-team/38-match season simulation.

## Photography and attribution

Football photographs are sourced from Franco Monsalvo's Pexels pages. Source links and licence notes are recorded in `ASSET_CREDITS.md` and `frontend/README.md`. The images are served from the Pexels CDN, so an internet connection is required to display them. PitchIQ is an independent project and is not affiliated with the Premier League or any club.
