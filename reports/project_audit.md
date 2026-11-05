# Initial Project & Data Audit Report
**English Premier League Match Outcome Predictor â€” 2026/27 Season**
**Auditor:** Senior Machine Learning Engineer & Football Data Scientist
**Date:** November 2026
**Environment:** Python 3.12.4 | Pandas 3.1.0 | Scikit-Learn 1.9.1 | NumPy 2.5.4 | SciPy 1.18.1

---

## 1. Executive Summary

This comprehensive audit inspects the workspace, repository structure, raw datasets, previous experimental artifacts, and dependency environments for the Premier League Match Predictor project.

The objective is to establish an end-to-end, leakage-free machine learning architecture that forecasts **Home Win (H)**, **Draw (D)**, and **Away Win (A)** probabilities summing to 100% for fixtures in the ongoing **2026/27 English Premier League** season.

---

## 2. Workspace Structure Inspection

The physical project workspace was systematically scanned:

```
EPL Match Predictor/
â”œâ”€â”€ .gitignore
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ app/                          # Application deployment layer (empty / future scope)
â”œâ”€â”€ Data/
â”‚   â”œâ”€â”€ Raw/
â”‚   â”‚   â”œâ”€â”€ 2026_27/              # Current-season 2026/27 snapshots (FBref & FPL)
â”‚   â”‚   â”œâ”€â”€ additional_statistics/# 2025/26 comprehensive player performance records
â”‚   â”‚   â”œâ”€â”€ match_results/        # 11 historical seasons (2015/16 through 2025/26)
â”‚   â”‚   â””â”€â”€ player_statistics/    # Auxiliary FPL player stats snapshot
â”‚   â””â”€â”€ Processed/                # Exported processed datasets and feature tables
â”œâ”€â”€ Models/                       # Serialized model artifacts, preprocessors & configs
â”œâ”€â”€ Notebooks/
â”‚   â”œâ”€â”€ 01_dataset_exploration.ipynb   # Historical match exploration & RF baseline
â”‚   â””â”€â”€ 02_team_strength_engine.ipynb  # Player rating & squad aggregation prototypes
â”œâ”€â”€ reports/                      # Audit, quality, strength, and evaluation reports
â”œâ”€â”€ src/                          # Modular production Python package
â””â”€â”€ venv/                         # Python virtual environment
```

---

## 3. Comprehensive Dataset Audit

| Dataset Category | File Path | Shape (Rows, Cols) | Season Coverage | Missing Values | Duplicate Records | Team / Player Coverage | Actual Performance vs Supplementary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Results** | `Data/Raw/match_results/season-1516.csv` to `season-2526.csv` (11 files) | 4,180 total (380 Ã— 11) | 2015/16 to 2025/26 | 0 missing in common columns | 0 duplicates | 34 historical clubs across 11 seasons | Actual match performance (FTHG, FTAG, FTR, shots, fouls, cards, corners) |
| **2026/27 Standings** | `Data/Raw/2026_27/premier_league_2026-27_table.csv` | (20, 16) | 2026/27 (9 matchdays) | Notes: 20 nulls | 0 duplicates | 20 current EPL clubs | Actual league table performance (MP, W, D, L, GF, GA, GD, Pts, Last 5) |
| **2026/27 Squad Standard** | `Data/Raw/2026_27/squad_standard_statistics_2026_27.csv` | (21, 21) | 2026/27 (current) | 0 missing | 0 duplicates | 20 clubs + league average | Actual performance (Goals, Assists, Possession%, Per 90 rates) |
| **2026/27 Shooting** | `Data/Raw/2026_27/squad_shooting_statistics_2026_27-v3.csv` | (21, 13) | 2026/27 (current) | 0 missing | 0 duplicates | 20 clubs + league average | Actual performance (Shots, SoT, SoT%, G/Sh, G/SoT, PKs) |
| **2026/27 Goalkeeping** | `Data/Raw/2026_27/squad_goalkeeping_statistics_2026_27.csv` | (21, 21) | 2026/27 (current) | Penalty kicks: 10 nulls | 0 duplicates | 20 clubs + league average | Actual performance (GA90, Save%, Clean Sheet%, Saves, SoTA) |
| **2026/27 Playing Time** | `Data/Raw/2026_27/squad_playing_time_statistics_2026_27-v4.csv` | (21, 19) | 2026/27 (current) | 0 missing | 0 duplicates | 20 clubs + league average | Actual performance (Squad depth, Minutes%, Starts, Subs, +/-) |
| **2026/27 Misc Stats** | `Data/Raw/2026_27/squad_miscellaneous_statistics_2026_27.csv` | (21, 15) | 2026/27 (current) | PKwon/con: 20 nulls | 0 duplicates | 20 clubs + league average | Actual performance (Tackles won, Interceptions, Crosses, Cards) |
| **2026/27 FPL Players** | `Data/Raw/2026_27/fpl_players-selected-columns.csv` | (670, 10) | 2026/27 (current) | news_added: 307 nulls | 0 duplicates | 670 players across 20 clubs | Supplementary / Squad affiliation (Name, Position, Club, Cost) |
| **2026/27 FPL Stats** | `Data/Raw/2026_27/fpl_player_statistics-selected-columns.csv` | (670, 10) | 2026/27 (current) | 0 missing | 0 duplicates | 670 players across 20 clubs | Supplementary / Squad affiliation (Name, Club, Points, Cost) |
| **2025/26 Player Stats** | `Data/Raw/additional_statistics/premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv` | (526, 106) | 2025/26 (GW 35) | xG: 88, xA: 5, GP: 489 | 0 duplicates | 526 players across 20 clubs | Actual historical individual player performance |

---

## 4. Verification of Previous Context Figures

1. **Historical Matches:** Confirmed exactly **4,180 matches** across 11 complete seasons (380 matches/season from 2015/16 through 2025/26).
2. **Current 2026/27 Teams:** Confirmed exactly **20 clubs** in the ongoing campaign.
3. **FPL Player Records:** Confirmed exactly **670 unique players** across 20 clubs in 2026/27.
4. **2025/26 Player Performance Records:** Confirmed exactly **526 player records** in the primary dataset (`premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv`).
5. **Baseline Benchmark:** Confirmed the experimental Random Forest baseline recorded in Notebook 1 of **45.72%** test accuracy.

---

## 5. Datasets Eligible for Merging

1. **2026/27 FBref Squad Datasets:**
   - Standard, Shooting, Goalkeeping, Playing Time, Miscellaneous, and League Table merge cleanly on `Squad` (using canonical team standardization) to form a unified 20-team, 96-metric 2026/27 team performance matrix.
2. **2026/27 FPL Player Metadata & Statistics:**
   - `fpl_players` and `fpl_player_statistics` merge on `id` / `player_id` with 100% 1-to-1 match rate, defining official 2026/27 club affiliations and rosters.
3. **2025/26 Player Performance & 2026/27 Roster:**
   - Merged using fuzzy token-matching and position constraints to map historical ability priors to current players, following transferred players to their new clubs.
