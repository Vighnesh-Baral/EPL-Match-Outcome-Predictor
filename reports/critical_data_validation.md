# Critical Data Validation Report
## EPL Match Outcome Predictor — 2026/27 Season

**Generated:** November 2026
**Validation Scope:** Club reconciliation, transfer audit, FPL integrity, merge correctness, missing statistics

---

## 1. Club Reconciliation — Three-Way Cross-Check

Three independent data sources were compared to establish the authoritative list of 2026/27 Premier League clubs.

### Sources Inspected

| Source | File | Clubs Found |
|--------|------|-------------|
| Official League Table | `Data/Raw/2026-27/epl_table_2026_27.csv` | 20 |
| FPL Player Metadata | `Data/Raw/2026-27/fpl_players_2026_27.csv` | 20 |
| Generated Team Strength | `Data/Processed/team_strength_2026_27.csv` | 20 |

### Authoritative 2026/27 Club List (20 Clubs)

| # | Club | Promoted / Relegated Status |
|---|------|-----------------------------|
| 1 | Arsenal | Established PL club |
| 2 | Aston Villa | Established PL club |
| 3 | Bournemouth | Established PL club |
| 4 | Brentford | Established PL club |
| 5 | Brighton & Hove Albion | Established PL club |
| 6 | Chelsea | Established PL club |
| 7 | Coventry City | **Promoted from Championship 2025/26** |
| 8 | Crystal Palace | Established PL club |
| 9 | Everton | Established PL club |
| 10 | Fulham | Established PL club |
| 11 | Hull City | **Promoted from Championship 2025/26** |
| 12 | Ipswich Town | **Promoted from Championship 2025/26** |
| 13 | Leeds United | Established PL club |
| 14 | Liverpool | Established PL club |
| 15 | Manchester City | Established PL club |
| 16 | Manchester United | Established PL club |
| 17 | Newcastle United | Established PL club |
| 18 | Nottingham Forest | Established PL club |
| 19 | Sunderland | Established PL club |
| 20 | Tottenham Hotspur | Established PL club |

### Relegated at End of 2025/26 (NOT in 2026/27)

| Club | Status |
|------|--------|
| Burnley | Relegated 2025/26 |
| West Ham United | Relegated 2025/26 |
| Wolverhampton Wanderers | Relegated 2025/26 |

> [!IMPORTANT]
> An earlier audit report incorrectly claimed "Hull City, Ipswich Town, Coventry City" were suspicious entries. These clubs ARE correctly in 2026/27 — they are the three promoted clubs. The report also incorrectly referenced "Burnley, West Ham, Wolves" as 2026/27 clubs — these were relegated at the end of 2025/26 and do NOT appear in any 2026/27 dataset. The raw datasets are correct; the prior audit report was wrong.

### Three-Way Reconciliation Table

All 20 clubs matched perfectly across all three sources. No discrepancies detected.

| Club | League Table | FPL Metadata | Team Strength | Status |
|------|:---:|:---:|:---:|--------|
| Arsenal | ✅ | ✅ | ✅ | Perfect match |
| Aston Villa | ✅ | ✅ | ✅ | Perfect match |
| Bournemouth | ✅ | ✅ | ✅ | Perfect match |
| Brentford | ✅ | ✅ | ✅ | Perfect match |
| Brighton | ✅ | ✅ | ✅ | Perfect match |
| Chelsea | ✅ | ✅ | ✅ | Perfect match |
| Coventry City | ✅ | ✅ | ✅ | Perfect match (promoted) |
| Crystal Palace | ✅ | ✅ | ✅ | Perfect match |
| Everton | ✅ | ✅ | ✅ | Perfect match |
| Fulham | ✅ | ✅ | ✅ | Perfect match |
| Hull City | ✅ | ✅ | ✅ | Perfect match (promoted) |
| Ipswich Town | ✅ | ✅ | ✅ | Perfect match (promoted) |
| Leeds United | ✅ | ✅ | ✅ | Perfect match |
| Liverpool | ✅ | ✅ | ✅ | Perfect match |
| Manchester City | ✅ | ✅ | ✅ | Perfect match |
| Manchester United | ✅ | ✅ | ✅ | Perfect match |
| Newcastle United | ✅ | ✅ | ✅ | Perfect match |
| Nottingham Forest | ✅ | ✅ | ✅ | Perfect match |
| Sunderland | ✅ | ✅ | ✅ | Perfect match |
| Tottenham Hotspur | ✅ | ✅ | ✅ | Perfect match |

**Export:** `Data/Processed/club_reconciliation_2026_27.csv` (23 rows: 20 active + 3 relegated)

---

## 2. Transfer Audit

### Summary Statistics

| Category | Count |
|----------|------:|
| Verified same-club (player matched, club unchanged) | 235 |
| **Verified transfers (player changed club)** | **46** |
| Player without sufficient historical info (<450 min played) | 75 |
| Unmatched player (new to PL, no prior record) | 206 |
| New PL player from promoted clubs (Coventry, Hull, Ipswich) | 108 |
| Ambiguous match (forced match without confidence) | **0** |
| **Total players** | **670** |

### Key Verified Transfers

| Player | From Club | To Club | Confidence |
|--------|-----------|---------|------------|
| Ezri Konsa | Aston Villa | Arsenal | High |
| Bruno Guimarães | Newcastle United | Arsenal | High |
| Morgan Rogers | Aston Villa | Chelsea | High |
| Emiliano Martínez | Aston Villa | Chelsea | High |
| (+ 42 further verified transfers) | | | |

**Full transfer list:** `Data/Processed/player_mapping_2026_27.csv`

### Transfer Methodology

- **Historical stats follow the player** to their new club. A player transferred from Club A to Club B contributes their prior-season performance metrics to Club B's ratings.
- **No zero-ability penalization** for unmatched or promoted players. A Bayesian prior of 50.0 was assigned.
- **No ambiguous matches forced.** Where match confidence was insufficient, the player was classified as "Unmatched" rather than risking an incorrect assignment.

> [!NOTE]
> 0 ambiguous matches is a valid and correct outcome. It means no match was made where confidence fell below the threshold — these cases were instead classified as Unmatched (category 4) with a Bayesian prior, which is the correct conservative approach.

---

## 3. FPL Data Integrity

| Check | Result |
|-------|--------|
| Total players in 2026/27 FPL dataset | 670 |
| Duplicate player IDs | **0** |
| Clubs covered | 20 (all 20 PL clubs) |
| Players per club (average) | 33.5 |
| Missing club entries | **0** |
| Join type used | LEFT JOIN (anchored on FPL metadata) |
| Data loss from join | **0 players lost** |

The FPL dataset was loaded with a LEFT JOIN on the authoritative club list, ensuring no silent data loss. All 670 players retained.

---

## 4. Inner-Join Fix Confirmation

### Problem Identified

The original `load_2026_27_fbref_team_stats()` function used INNER JOINs when merging supplementary FBref datasets (shooting, goalkeeping, defensive, passing). An inner join silently drops clubs absent from any supplementary dataset, potentially reducing coverage from 20 to fewer clubs.

### Fix Applied

**All inner joins were replaced with LEFT JOINs anchored on the authoritative league table.**

```python
# BEFORE (incorrect - silently drops clubs):
merged = merged.merge(shooting_df, on="Squad", how="inner")

# AFTER (correct - preserves all 20 clubs):
merged = league_table_df.merge(squad_df, on="Squad", how="left")
merged = merged.merge(shooting_df, on="Squad", how="left")
```

### Post-Fix Validation

Pre-merge validation was added to detect duplicates and missing clubs in each supplementary source before merging:

| Supplementary File | Clubs Found | Duplicates | Missing Clubs |
|--------------------|:-----------:|:----------:|:-------------:|
| Squad Standard Stats | 20 | 0 | 0 |
| Shooting Stats | 20 | 0 | 0 |
| Goalkeeping Stats | 20 | 0 | 0 |
| Defensive Stats | 20 | 0 | 0 |
| Passing Stats | 20 | 0 | 0 |

**Result:** All 5 supplementary files have exactly 20 clubs, 0 duplicates, 0 missing. The LEFT join fix results in zero additional data loss vs. the inner join approach for this dataset.

---

## 5. Missing Statistics

| Statistic Type | Missing Clubs | Action Taken |
|----------------|:-------------:|--------------|
| Squad Standard Stats | 0 | No action needed |
| Shooting Stats | 0 | No action needed |
| Goalkeeping Stats | 0 | No action needed |
| Defensive Stats | 0 | No action needed |
| Passing Stats | 0 | No action needed |
| FPL Player Stats | 0 | No action needed |

**No missing statistics for any club in any supplementary source.**
Null values in historical training features were handled by median imputation in the preprocessing pipeline.

---

## 6. Feature Leakage Check

All 50 features used in model training were verified to be strictly chronological — computed only from information available **before** the target match date.

| Feature Group | Leakage Risk | Status |
|---------------|:------------:|--------|
| Rolling Elo (dynamic, updated per match) | None | ✅ Safe |
| Rolling Form PPG (Last 3/5/10) | None | ✅ Safe |
| Rolling Goals Scored/Conceded | None | ✅ Safe |
| Rolling Shots on Target | None | ✅ Safe |
| Rest Days | None | ✅ Safe |
| Home Advantage (constant) | None | ✅ Safe |
| Home/Away Venue Form (rolling) | None | ✅ Safe |
| Squad Depth (historical proxy) | None | ✅ Safe* |

> [!NOTE]
> *Squad Depth in historical training features is an **approximated historical proxy** (function of match history length × multiplier), NOT the 2026/27 snapshot rating. The 2026/27 Team Strength snapshot is used exclusively at prediction time, not injected into historical training rows.

---

## 7. Acceptance Checklist

| Criterion | Status |
|-----------|--------|
| All 20 verified 2026/27 clubs accounted for | ✅ |
| No duplicate clubs in any dataset | ✅ |
| FPL player metadata preserved, no duplicate IDs | ✅ |
| Transfer mappings audited and categorized | ✅ 46 transfers, 0 ambiguous |
| Missing statistics reported | ✅ 0 missing |
| No silent inner-join data loss | ✅ Fixed to LEFT JOIN |
| Current-season stats contribute to Team Strength | ✅ |
| Depth scores validated | ✅ Corrected formula |
| Leakage-safe chronological features | ✅ |
| Chronological train/val/test splits respected | ✅ |
| Test set untouched during model selection | ✅ |
| Draw prediction explicitly evaluated | ✅ |
| Calibration assessed | ✅ Sigmoid calibration applied |
| Prediction function returns probabilities summing to 1 | ✅ Verified |
