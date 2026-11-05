# PitchIQ â€” EPL 2026/27 Match Predictor (Version 2.3)

A local Premier League predictor combining a recency-weighted historical classifier, a current-season Poisson branch, 2026/27 team/player ratings, and head-to-head context.

## Model design

### 1. Recency-weighted historical learning

The model trains on 2015/16â€“2025/26 fixture results. The agreed priority hierarchy is represented through initial sample weights:

| Season | Weight |
|---|---:|
| 2015/16 | 0.25 |
| 2016/17 | 0.27 |
| 2017/18 | 0.29 |
| 2018/19 | 0.31 |
| 2019/20 | 0.35 |
| 2020/21 | 0.45 |
| 2021/22 | 0.50 |
| 2022/23 | 0.58 |
| 2023/24 | 0.72 |
| 2024/25 | 0.86 |
| 2025/26 | 1.00 |
| 2026/27 (when a result file is added) | 1.15 |

These weights express the intended priority order; they are starting values, not proven optimal coefficients. The model comparison report should be used to revise them only after further validation.

Historical pre-match features are reconstructed chronologically from each team's season-to-date points, goals for/against, recent form and the latest ten prior head-to-head meetings. Matches on the same date are all featurised before any results from that date are added, avoiding same-day leakage.

### 2. Model comparison and validation

The training pipeline compares:

- Multinomial Logistic Regression
- Random Forest
- XGBoost
- Poisson goals baseline

Chronological split:

- Training: 2015/16â€“2021/22
- Validation/model selection: 2022/23â€“2023/24
- Untouched test period: 2024/25â€“2025/26

The validation period selects the classifier and the classifier/Poisson blend weight using log loss. The selected architecture is then tested on 2024/25â€“2025/26 before being refitted on all available historical seasons for production. Reports include accuracy, log loss and multiclass Brier score. Test metrics are not guarantees of 2026/27 accuracy.

### 3. Current 2026/27 inputs

- Current FBref team statistics and league table power the current Attack, Midfield, Defence and Goalkeeping indices.
- Current FPL points and roster information feed position-specific contribution indices and squad depth proxies.
- The current team ratings drive the current-season Poisson branch and match-specific goal expectations.
- Home advantage remains a separate explicit factor (`HOME_ADVANTAGE_FACTOR = 1.06`). The classifier also learns the historical home/away outcome asymmetry from match results.
- H2H appears in the interface and enters the historical classifier as a limited pre-match feature. It is not used as a standalone predictor.

The supplied 2026/27 snapshot shows nine matches played per club, but has aggregate standings/team statistics rather than a fixture-by-fixture result file. Therefore, 2026/27 cannot yet be included in supervised training or prospectively validated. Its current data still has the highest priority in the current-season ratings and Poisson branch.

## Current data limitations

The current FPL snapshot contains player total points, event points, cost, club and position, but not complete player minutes, starts, goals, assists, xG, tackles or interceptions. The player metric remains an **FPL contribution index**, not a complete football ability grade. The 4-3-3 and reserve group remain quality proxies, not predicted matchday lineups; injury/suspension availability is not present.

The team-rating blends and current rating-to-goal multipliers are transparent starting assumptions. The historical blend weight is selected on a chronological validation period, but the current 2026/27 snapshot adjustment still requires prospective evaluation as new fixture results arrive.

## Run locally on Windows

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
python run.py
```

Open **http://127.0.0.1:8000**. Stop the server with `Ctrl+C`.

The app also auto-trains the model once if `Models/historical_predictor_v2.joblib` is missing. Explicitly running `python -m src.train` rebuilds it and refreshes the reports.

## Add 2026/27 match results later

When a completed-match CSV is available, add it to `Data/Raw/2026_27/` using one of these names:

- `match_results_2026_27.csv`
- `season-2627.csv`
- `matches_2026_27.csv`

Required columns: `Date`, `HomeTeam`, `AwayTeam`, `FTR`, `FTHG`, `FTAG`. The training loader will assign those results the 2026/27 weight of 1.15 and refit the production model. Do not insert aggregate table rows as if they were fixture results.

## API endpoints

- `GET /api/health`
- `GET /api/overview`
- `GET /api/teams`
- `GET /api/teams/{team_name}`
- `POST /api/predict`
- `GET /api/h2h?home_team=Arsenal&away_team=Liverpool&limit=10`
- `GET /docs`

Prediction request:

```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool"
}
```

## Rebuild current ratings and run tests

```powershell
python -m src.team_strength
pytest -q
```

Model evaluation outputs are written to `reports/model_evaluation_v2.json`, `reports/model_comparison_v2.csv` and `reports/model_test_results_v2.csv`.

## Version 2.1 update: current-season H2H and prior-season stabilization

- `Data/Raw/2026_27/match_results_2026_27.csv` stores verified completed 2026/27 results. The first verified H2H addition is Brighton 3-0 Arsenal on 2026-09-19. Add further completed fixtures to this same CSV using `Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR`; the loader validates and automatically includes them in H2H and historical features.
- **Coverage note:** this archive currently contains one verified 2026/27 fixture row, not a complete 2026/27 match log. Do not interpret the file as containing all 87/90 completed matches.
- Overall club strength now blends the current 2026/27 rating with a 2025/26 Premier League prior. The prior starts at 15% and decays exponentially as current-season matches accumulate (`0.15 * exp(-MP / 18)`). Clubs without a 2025/26 Premier League record receive a neutral prior of 50 rather than fabricated lower-division performance.
- This 15% prior is an initial transparent regularization choice, not an empirically optimized parameter. It should be tuned using chronological backtests.

## Version 2.3 UI â€” editorial football intelligence workspace

The React + Vite frontend in `frontend/` now has a scroll-led product landing page and a separate analytics workspace. The visual system follows the supplied pastel/editorial reference: warm ivory, charcoal, sage and restrained terracotta, with football photography and purposeful scroll reveals. The workspace uses the same visual language and remains data-led rather than decorative.

### Implemented screens

- **Overview:** API-powered league snapshot, team-strength index, points leaders and model summary.
- **Match predictor:** live `POST /api/predict` request with outcome probabilities, expected goals, current team-rating comparison and model blend details.
- **Team intelligence:** club directory and detail panel from `GET /api/teams/{team_name}`.
- **Player intelligence:** searchable roster and filters; player rows are collected from the existing team-detail endpoints using only available FPL fields.
- **Head to head:** meeting history from `GET /api/h2h`.
- **Model performance:** historical holdout accuracy, log loss and Brier score from the existing overview response.

The landing page and workspace use initials-based club identifiers, not Premier League or club crests. Model metrics are labelled as historical holdout results and are not described as future-performance guarantees.

### Run the frontend in development

1. Start the backend from the project root: `python run.py`.
2. Open a second terminal in `frontend/`.
3. Run `npm install` and then `npm run dev`.
4. Open `http://127.0.0.1:5173`.

Vite proxies `/api` requests to FastAPI at `http://127.0.0.1:8000`, so the two development servers communicate over the API.

### Build for the FastAPI app

From `frontend/`, run `npm run build`. The production bundle is written to `app/static/`; FastAPI serves it at `http://127.0.0.1:8000`.

### Photography

Football photographs are sourced from Franco Monsalvo's Pexels pages. Source links and licence notes are recorded in `ASSET_CREDITS.md` and `frontend/README.md`. Pexels permits free use under its licence, but any third-party rights in people, trademarks or brands depicted in photos remain relevant. The photos are served from the Pexels image CDN, so the browser needs an internet connection to display them.
