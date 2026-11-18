# PitchIQ Version 2.4 — Comprehensive System & Predictive Performance Audit

**Audit Date:** November 2026  
**Audited Target:** PitchIQ Version 2.4 (`historical_predictor_v2.joblib`, `src/historical_model.py`, `src/predict.py`, `app/main.py`)  
**Auditor:** Senior ML Engineer & Football Data Scientist  
**Environment:** Python 3.12.4 | scikit-learn 1.9.1 | XGBoost 3.4.1 | FastAPI 0.162.0 | pytest 9.2.0  

---

## Executive Summary

PitchIQ Version 2.4 was audited to establish the baseline for the Version 2.5 upgrade. The system integrates historical supervised classification with a Poisson goal model and provides Matchday predictions and Monte Carlo end-of-season table simulations.

All 14 automated unit and integration tests passed (`14 passed in 33.64s`). The pipeline exhibits strict date-level leakage control, separating same-date fixtures and isolating the historical training set (2015/16–2025/26) from the current season.

However, the audit identified critical predictive and architectural limitations:
1. **Severe Draw Prediction Failure:** Version 2.4 achieved a Draw Recall of only **1.02%** on the held-out test set (only 2 out of 197 draws correctly classified).
2. **Feature Poverty:** The historical classifier uses only 16 basic features. It completely lacks dynamic Elo ratings, opponent-adjusted form, venue-specific splits, shot metrics (shots on target, conversion rates), available xG, and fixture congestion/rest days.
3. **Early-Season Noise:** Season-to-date metrics reset at the start of each season, causing high variance and reliance on uncalibrated defaults (1.35 PPM, 1.45 GF) for the first 5–10 rounds.
4. **Heuristic Disconnection:** The 2026/27 team-strength ratings (FBref + FPL) are mapped into Poisson parameters via arbitrary linear heuristics (`0.75 + 0.50 * rating/100`) rather than learned empirical mappings.
5. **No Correlation in Scorelines:** Independent Poisson under-predicts 0–0 and 1–1 draws because it lacks the bivariate Dixon-Coles dependence parameter ($\tau$).

---

## 1. System Architecture & Component Inventory

### 1.1 Data Assets
- **Historical Matches Archive:** 4,180 completed Premier League fixtures spanning 11 seasons (2015/16 to 2025/26) stored in `Data/Raw/Historical/`.
- **2026/27 Matchday Results:** 11 verified early-season fixtures used strictly for Head-to-Head (H2H) context and excluded from training.
- **2026/27 Current League Standings:** Refreshed through Matchday 10 (20 clubs, 10 matches played each) in `Data/Raw/2026_27/current_standings_2026_27.csv`.
- **2026/27 Squad & Player Snapshots:** FBref squad standard, shooting, goalkeeping, defensive, and passing tables, plus 670 FPL players with contribution and squad depth metrics.
- **Scheduled Fixtures:** Complete 280 remaining fixtures for Matchdays 11–38 in `Data/Raw/2026_27/fixtures_matchday_11_38.csv`.

### 1.2 Model Architecture (Version 2.4)
- **Model Paradigm:** Two-branch hybrid ensemble:
  $$\hat{P}(y) = \alpha \hat{P}_{\text{ML}}(y) + (1 - \alpha) \hat{P}_{\text{Poisson}}(y)$$
- **Classifier Branch:** Recency-weighted Random Forest (450 estimators, `max_depth=12`, `min_samples_leaf=8`).
- **Poisson Branch:** Bounded Poisson goal model with grid-searched venue factor ($\text{multiplier} = 1.06$).
- **Selected Weights (Validation Min Log Loss):** $\alpha = 0.70$ Random Forest, $1 - \alpha = 0.30$ Poisson.

---

## 2. Reproduction of Baseline Evaluation Metrics

The historical backtesting protocol was verified:
- **Training Period:** 2015/16 to 2021/22 (2,660 matches) with exponential season recency weights (0.25 to 0.50).
- **Validation Period:** 2022/23 to 2023/24 (760 matches).
- **Held-Out Test Period:** 2024/25 to 2025/26 (760 matches, completely untouched during tuning).

### Validation Set Comparison (2022/23–2023/24, 760 matches)
| Model Architecture | Blend Weight ($\alpha$) | Log Loss | Brier Score | Accuracy | Macro F1 | Home Recall | Draw Recall | Away Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Poisson Baseline (1.06)** | 0.00 | 1.0125 | 0.5978 | 51.58% | 0.3920 | 69.92% | 2.96% | 64.66% |
| **Logistic Regression + Poisson** | 0.80 | 0.9748 | 0.5796 | 53.55% | 0.3940 | 74.09% | 0.00% | 60.78% |
| **XGBoost + Poisson** | 0.45 | 0.9899 | 0.5880 | 52.24% | 0.3956 | 68.80% | 1.18% | 63.79% |
| **Random Forest + Poisson (v2.4)** | **0.70** | **0.9699** | **0.5754** | **55.26%** | **0.4090** | **74.65%** | **0.00%** | **65.52%** |

### Held-Out Test Set Performance (2024/25–2025/26, 760 matches)
| Model | Accuracy | Log Loss | Brier Score | Macro F1 | Home Recall | Draw Recall | Away Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Poisson Baseline (1.06 ref)** | 47.76% | 1.0804 | 0.6397 | 0.3774 | 69.09% | 2.54% | 56.50% |
| **Random Forest Standalone** | 50.00% | 1.0272 | 0.6168 | 0.3824 | 78.86% | 1.52% | 51.63% |
| **Random Forest + Poisson (v2.4 Production)** | **48.68%** | **1.0233** | **0.6151** | **0.3728** | **74.76%** | **1.02%** | **53.25%** |

---

## 3. Data Leakage and Integrity Audit

### Verified Protections in Version 2.4:
- **Same-Date Anti-Leakage:** All matches on calendar date $T$ are featurized strictly against the state of the world at $T - \epsilon$. No match on date $T$ influences any other match played on date $T$.
- **Strict Separation of Incomplete Season:** The 2026/27 11-match partial log is excluded from model fitting to prevent sample distortion from an unrepresentative 11-match sample.
- **Left-Joined FBref & FPL Datasets:** No silent row-dropping across the 20 Premier League clubs. All 670 players are preserved.

### Identified Latent Leakage & Inconsistency Risks:
1. **Cumulative League Average:** `league_avg_team_goals` in `build_historical_features()` is computed as cumulative goals across all seen matches. Early in the dataset (2015/16), it starts at 0 before stabilizing.
2. **H2H Sign Orientation:** The head-to-head aggregation logic contains a sign-inversion edge case when calculating points from the away team's perspective.
3. **No Explicit Split for Promoted Teams:** Promoted clubs with no Premier League history in the preceding 1–3 seasons revert directly to neutral priors (1.35 PPM) rather than receiving Championship-adjusted strength baselines.

---

## 4. In-Depth Root Cause Analysis: The Draw Dilemma

In the test set of 760 matches, the empirical outcome distribution is:
- **Home Wins:** 317 (41.71%)
- **Draws:** 197 (25.92%)
- **Away Wins:** 246 (32.37%)

Yet, the Version 2.4 model predicted only **4 draws** across all 760 matches, getting only 2 right (Precision: 50.0%, Recall: 1.02%).

### Root Causes:
1. **Multiclass Argmax Thresholding:**
   The average predicted draw probability across all matches was 24.8%. Because draw probabilities peaked at ~34% in close games while Home or Away win probabilities regularly exceeded 35–45%, the `argmax` operator almost never selects Draw as the modal class.
2. **Independent Poisson Draw Under-Estimation:**
   Under independent Poisson distributions, the probability of equal goals $P(X = Y)$ is:
   $$P(\text{Draw}) = \sum_{k=0}^{\infty} e^{-(\lambda_H + \lambda_A)} \frac{(\lambda_H \lambda_A)^k}{(k!)^2} = e^{-(\lambda_H + \lambda_A)} I_0(2\sqrt{\lambda_H \lambda_A})$$
   In football, low-scoring draws (0–0 and 1–1) occur significantly more often than independent Poisson predicts due to defensive tactical conservatism when scores are level.
3. **Absence of Calibrated Probability Thresholds:**
   No temperature scaling, Platt scaling, or utility-optimizing decision thresholds were tuned on validation data.

---

## 5. Upgrade Roadmap for PitchIQ Version 2.5

To significantly improve match prediction accuracy, probability calibration, and draw detection, Version 2.5 will implement:

1. **Rich Multi-Source Historical Data Ingestion:**
   - Ingest Football-Data.co.uk full match statistics: Shots (HS, AS), Shots on Target (HST, AST), Corners (HC, AC), Fouls (HF, AF), and historical betting odds benchmarks (B365H, B365D, B365A).
   - Ingest available historical and current xG/xGA from reliable open repositories.
2. **Advanced Pre-Match Feature Engineering (Strictly Leakage-Free):**
   - Dynamic Elo Rating system with margin-of-victory and home-court advantage adjustment.
   - Opponent-adjusted rolling form (PPM and GD adjusted by the opponent's Elo at match time).
   - Venue-specific rolling performance (Home team at Home vs. Away team Away).
   - Shot dominance and shooting efficiency indices (SoT ratio, conversion rate).
   - Fixture congestion & rest days calculated from exact calendar dates.
3. **Dixon-Coles Bivariate Goal Model:**
   - Replace standard independent Poisson with a Dixon-Coles model incorporating the parameter $\tau(\lambda_H, \lambda_A, \rho)$ for low scorelines (0–0, 1–0, 0–1, 1–1).
4. **Calibrated Multi-Model Ensemble:**
   - Train and cross-validate Random Forest, XGBoost, CatBoost/HistGradientBoosting, Calibrated Logistic Regression, and Dixon-Coles.
   - Apply isotonic or Platt probability calibration using chronological validation folds.
5. **Full Application & UI Integration:**
   - Update all backend routes (`/api/predict`, `/api/matchday/{id}`, `/api/table-prediction`, `/api/overview`) to expose the upgraded Version 2.5 features, expected goals, and calibrated distributions without breaking any existing dashboard features.

