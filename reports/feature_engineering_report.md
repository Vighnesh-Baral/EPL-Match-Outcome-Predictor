# Leakage-Safe Feature Engineering Report
**EPL Match Outcome Predictor â€” 2026/27**

## 1. Principles of Time-Aware Feature Engineering

Machine learning in sports forecasting is particularly vulnerable to data leakage. If end-of-season statistics, future match outcomes, or retroactively computed metrics leak into historical training records, model evaluation becomes falsely optimistic and fails in production.

### Strict Leakage-Free Rules Implemented:
1. **Strict Chronological Sequencing:**
   Matches across 11 seasons (2015/16 through 2025/26) are sorted strictly by `Date` and match occurrence.
2. **Pre-Kickoff State Only:**
   For match $M_{i}$ on date $D_{i}$, only matches $M_{j}$ where $D_{j} < D_{i}$ are accessible. No statistics from match $M_{i}$ or future matches exist in the feature set.
3. **No Retroactive 2026/27 Application:**
   The current 2026/27 snapshot is never retroactively projected into 2015â€“2026 historical match features. For historical matches, all strength and form indicators are dynamically reconstructed from prior results.

---

## 2. Feature Specifications

### 2.1 Dynamic Elo Ratings
- **Base Formulation:**
  $$\hat{P}(\text{Home Win}) = \frac{1}{1 + 10^{(\text{Elo}_{\text{Away}} - (\text{Elo}_{\text{Home}} + 65)) / 400}}$$
- **Update Rule:**
  $$\text{Elo}_{\text{new}} = \text{Elo}_{\text{old}} + K \times M \times (S_{\text{actual}} - \hat{P})$$
  where $K = 25$, $S \in \{1.0, 0.5, 0.0\}$, and $M$ is the margin-of-victory multiplier ($M = 1.0$ for 1-goal margins, $1.5$ for 2 goals, $1.75 + (\Delta G - 3)/8$ for larger margins).
- **Off-Season Regression:**
  At the onset of each new campaign, team ratings regress 20% toward the league mean ($1500.0$) to capture summer transfer activity, managerial changes, and natural regression:
  $$\text{Elo}_{\text{new season}} = 0.80 \times \text{Elo}_{\text{prior}} + 0.20 \times 1500.0$$

### 2.2 Chronological Form Indicators
- **Last 3 Matches:** `Home_Form_Last3_PPG`, `Away_Form_Last3_PPG`, `Form_Last3_PPG_Diff`
- **Last 5 Matches:** `Home_Form_Last5_PPG`, `Away_Form_Last5_PPG`, `Form_Last5_PPG_Diff`
- **Last 10 Matches:** `Home_Form_Last10_PPG`, `Away_Form_Last10_PPG`, `Form_Last10_PPG_Diff`
- **Goals & Goal Differentials:** `Home_GF_Last5`, `Away_GF_Last5`, `Home_GA_Last5`, `Away_GA_Last5`, `GD_Last5_Diff`
- **Shots & Shots on Target:** `Home_SoT_Last5`, `Away_SoT_Last5`, `SoT_Last5_Diff`, `Home_SoTA_Last5`, `Away_SoTA_Last5`

### 2.3 Venue-Specific Form
Home and away environments exert distinct psychological and tactical pressures.
- `HomeTeam_Home_PPG_Last5`: Points per match earned by the home club exclusively in their last 5 home fixtures.
- `AwayTeam_Away_PPG_Last5`: Points per match earned by the away club exclusively in their last 5 away fixtures.
- `Venue_Form_Diff`: Difference between home performance at home and away performance on the road.

### 2.4 Rest Days & Schedule Congestion
- `Home_Rest_Days`: Days elapsed since previous competitive fixture (clipped to $[1, 30]$).
- `Away_Rest_Days`: Days elapsed since previous fixture.
- `Rest_Days_Diff`: Rest advantage differential.

---

## 3. Incremental Feature Group Design (Models A, B, C, D)

To systematically measure the value of each component as required by Part 9 of the project architecture:

- **Model A (Team Strength Only):**
  12 features capturing attacking strength, defensive strength, net ratings, and shots on target differentials.
- **Model B (Team Strength + Recent Form):**
  28 features adding rolling 3, 5, and 10-match PPG, goal differentials, and recent scoring trends.
- **Model C (Strength + Form + Historical Strength):**
  34 features adding pre-match dynamic Elo ratings and rest day differentials.
- **Model D (Full Architecture: Strength + Form + Historical + Home Advantage):**
  41 features adding venue-specific home/away form, venue goal differentials, home advantage indicators, and prior league home win rates.

All 4,180 historical matches were populated without nulls or infinite values and exported to `Data/Processed/match_features.csv`.
