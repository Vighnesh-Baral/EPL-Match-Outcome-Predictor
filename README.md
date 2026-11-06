# PitchIQ — EPL 2026/27 Match Predictor (Version 2.3)

PitchIQ is a local Premier League football intelligence platform combining a recency-weighted historical machine learning model, a current-season Poisson prediction branch, 2026/27 team and player ratings, and head-to-head analysis.

## Model Design

### 1. Recency-Weighted Historical Learning

The historical model trains on Premier League match results from 2015/16 to 2025/26. Each season receives a sample weight reflecting its relative importance:

| Season | Sample Weight |
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
| 2026/27 | 1.15 (when verified match results are available) |

These weights represent the intended priority hierarchy. They are initial design choices, not empirically proven optimal values.

Historical pre-match features are reconstructed chronologically using:
- Season-to-date points and goal statistics.
- Recent team form.
- The latest ten head-to-head meetings available before the target fixture.

Matches played on the same date are featurised before any results from that date are added. This prevents results from one match on a given date from leaking into the features of another match on that date.

### 2. Model Comparison and Validation

The training pipeline compares four approaches:

- Multinomial Logistic Regression
- Random Forest
- XGBoost
- Poisson goals baseline

**Chronological data split:**

| Dataset | Seasons | Purpose |
|---|---|---|
| Training | 2015/16–2021/22 | Model training |
| Validation | 2022/23–2023/24 | Model selection and blend-weight tuning |
| Testing | 2024/25–2025/26 | Out-of-sample evaluation |

The validation period is used to select the classifier and determine the classifier/Poisson blending weight using log loss.

The selected architecture is evaluated on the untouched test period before being refitted using all available historical seasons for production.

Evaluation reports include accuracy, log loss and multiclass Brier score. Historical test performance should not be interpreted as a guarantee of future prediction accuracy.

### 3. Current 2026/27 Inputs

The current-season intelligence layer uses:

- **FBref team statistics:** Current team performance and league standings contribute to Attack, Midfield, Defence and Goalkeeping indices.
- **Fantasy Premier League data:** Current player points and squad information contribute to position-specific player ratings and squad-depth proxies.
- **Current team ratings:** These feed the Poisson prediction branch and influence match-specific expected goals.
- **Home advantage:** An explicit factor of `HOME_ADVANTAGE_FACTOR = 1.06` is applied separately. The historical classifier also learns home/away outcome patterns from historical results.
- **Head-to-head:** H2H information is displayed in the interface and included as a limited historical model feature. It is not used as a standalone prediction method.

**2026/27 data coverage:** The supplied league snapshot contains nine matches played per club. However, the current match-results CSV contains only one verified 2026/27 fixture. Consequently, the current-season data cannot yet support comprehensive supervised training or prospective validation.

The current-season team and player statistics still contribute to the ratings and Poisson branch. They should not be confused with a complete fixture-by-fixture results archive.

## Current Data Limitations

The current FPL snapshot contains:
- Player total points.
- Event points.
- Player cost.
- Club.
- Playing position.

It does not provide complete player-level information such as minutes played, starts, goals, assists, expected goals (xG), tackles or interceptions.

Therefore, the player metric is an **FPL contribution index**, not a comprehensive measure of football ability.

The 4-3-3 squad structure and reserve group are quality proxies. They are not predicted matchday lineups. The current dataset also does not contain injury or suspension availability information.

Team-rating blends and current rating-to-goal multipliers are transparent starting assumptions. Although the historical model's blend weight is selected using chronological validation, the current-season rating adjustments still require prospective evaluation as more 2026/27 results become available.

## Installation and Setup (Windows)

### 1. Clone the repository

```powershell
git clone https://github.com/Vighnesh-Baral/EPL-Match-Outcome-Predictor.git
cd EPL-Match-Outcome-Predictor
```

### 2. Create and activate a virtual environment

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the historical model

```powershell
python -m src.train
```

### 5. Start the application

```powershell
python run.py
```

Open the application at:

**http://127.0.0.1:8000**

Stop the server using `Ctrl+C`.

The application can also train the model automatically if `Models/historical_predictor_v2.joblib` is missing. Running `python -m src.train` explicitly rebuilds the model and refreshes the evaluation reports.

## Adding 2026/27 Match Results

To incorporate additional verified 2026/27 results, add them to:

`Data/Raw/2026_27/match_results_2026_27.csv`

Accepted filenames:

- `match_results_2026_27.csv`
- `season-2627.csv`
- `matches_2026_27.csv`

Required columns:

`Date, HomeTeam, AwayTeam, FTR, FTHG, FTAG`

The data loader assigns 2026/27 results a sample weight of 1.15 and includes them in historical feature generation and model training.

Only completed, verified fixture results should be added. Do not treat aggregate league-table statistics as individual match results.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Application and data health |
| GET | `/api/overview` | League and model overview |
| GET | `/api/teams` | Current team ratings |
| GET | `/api/teams/{team_name}` | Individual team information and roster |
| POST | `/api/predict` | Match outcome prediction |
| GET | `/api/h2h` | Head-to-head match history |
| GET | `/docs` | Interactive API documentation |

### Example Prediction Request

```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool"
}
```

The prediction endpoint accepts the home and away club names and returns the model's prediction output.

## Rebuilding Ratings and Running Tests

Rebuild current-season team ratings:

```powershell
python -m src.team_strength
```

Run the automated test suite:

```powershell
pytest -q
```

Model evaluation outputs are stored in:

- `reports/model_evaluation_v2.json`
- `reports/model_comparison_v2.csv`
- `reports/model_test_results_v2.csv`

## Version 2.1 — H2H and Previous-Season Stabilisation

The following improvements were introduced in Version 2.1:

- The 2026/27 match-results file supports verified current-season results in H2H and historical feature generation.
- The first verified current-season H2H addition is Brighton 3–0 Arsenal on 19 September 2026.
- The archive currently contains only one verified 2026/27 fixture row. It must not be interpreted as a complete current-season match log.
- Overall team strength incorporates a 2025/26 Premier League prior alongside the current 2026/27 rating.
- The previous-season prior starts at 15% and decays exponentially as current-season matches accumulate:

  `0.15 × exp(-MP / 18)`

- Clubs without a 2025/26 Premier League record receive a neutral prior of 50 rather than an invented lower-division rating.

The 15% starting contribution is a transparent regularisation choice, not an empirically optimised parameter. It should be evaluated using chronological backtesting.

## Version 2.3 — Editorial Football Intelligence Workspace

Version 2.3 introduces a React + Vite frontend with a scroll-led product landing page and a separate analytics workspace.

The visual system uses:
- Warm ivory backgrounds.
- Charcoal typography.
- Sage green accents.
- Restrained terracotta details.
- Football photography.
- Purposeful scroll-based reveals and interactive elements.

The application uses original initials-based club identifiers rather than official Premier League or club crests.

### Implemented Screens

**1. Overview**
- API-powered league snapshot.
- Team-strength leaderboard.
- Points leaders.
- Model information and performance summary.

**2. Match Predictor**
- Live predictions through `POST /api/predict`.
- Home, draw and away probabilities.
- Expected goals.
- Team-rating comparison.
- Model blending information.

**3. Team Intelligence**
- Club directory.
- Individual team ratings.
- Team statistics and squad information.

**4. Player Intelligence**
- Searchable player roster.
- Position-based filters.
- Available FPL contribution metrics.

**5. Head-to-Head**
- Historical meetings between selected clubs.
- Match history retrieved through `GET /api/h2h`.

**6. Model Performance**
- Historical holdout accuracy.
- Log loss.
- Multiclass Brier score.

Model metrics are presented as historical evaluation results, not guarantees of future performance.

## Running the Frontend in Development Mode

The frontend and backend run as separate development servers.

**Terminal 1 — Backend**

From the project root:

```powershell
python run.py
```

**Terminal 2 — Frontend**

Navigate to the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

Open:

**http://127.0.0.1:5173**

Vite proxies `/api` requests to the FastAPI backend at `http://127.0.0.1:8000`.

## Building the Frontend for Production

From the `frontend/` directory:

```powershell
npm run build
```

The production bundle is generated inside `app/static/`.

FastAPI serves the built frontend at:

**http://127.0.0.1:8000**

## Photography and Asset Credits

Football photographs are sourced from Franco Monsalvo's Pexels pages.

Source links and licence information are documented in:
- `ASSET_CREDITS.md`
- `frontend/README.md`

Pexels permits use of its images under its applicable licence. However, separate rights relating to identifiable people, trademarks and brands depicted in photographs may still apply.

The photographs are served through the Pexels image CDN, so an internet connection is required to display them.
