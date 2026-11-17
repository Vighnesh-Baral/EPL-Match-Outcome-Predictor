# Corrective Validation Report
## EPL Match Outcome Predictor — 2026/27 Season

**Generated:** November 2026
**Scope:** Issues identified in the initial implementation, corrections applied, before/after validation

---

## 1. Overview

This report documents every significant issue found in the initial implementation, the corrections applied, and validation evidence confirming the corrections.

---

## 2. Issue Log and Corrections

### Issue 1 — Incorrect Club Reconciliation Claim in Audit Report

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Location** | Prior audit report (not code) |
| **Issue** | Report claimed Hull City, Ipswich Town, Coventry City were suspicious discrepancies. Separately listed Burnley, West Ham, and Wolves as expected 2026/27 clubs. |
| **Root Cause** | Audit used 2025/26 squad data as reference instead of 2026/27 authoritative sources. Burnley, West Ham, and Wolves were relegated at end of 2025/26. Hull City, Ipswich Town, and Coventry City were promoted — they ARE correctly in 2026/27. |
| **Correction** | Three-way cross-check: league table ✗ FPL metadata ✗ team strength. All 20 clubs match perfectly. |
| **Validation** | `Data/Processed/club_reconciliation_2026_27.csv` exported. Zero discrepancies. |

---

### Issue 2 — Inner Joins Silently Dropped Clubs

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Location** | `src/data_loader.py` — `load_2026_27_fbref_team_stats()` |
| **Issue** | All 5 supplementary FBref dataset merges used `how="inner"`. An inner join silently drops any club absent from any supplementary source, potentially reducing coverage below 20. |
| **Root Cause** | Default Pandas merge behaviour. No pre-merge validation existed. |
| **Correction** | All inner joins replaced with LEFT JOINs anchored on the authoritative league table. Added pre-merge duplicate/missing club checks. |

**Before:**
```python
merged = merged.merge(shooting_df, on="Squad", how="inner")
```
**After:**
```python
merged = league_table_df.merge(squad_df, on="Squad", how="left")
merged = merged.merge(shooting_df, on="Squad", how="left")
```
| **Validation** | Post-fix: 20 clubs across all 5 supplementary files, 0 duplicates, 0 missing. Zero data loss. |

---

### Issue 3 — Squad Depth Scores Were Extreme and Unreliable

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Location** | `src/team_strength.py` — `compute_squad_depth()` |
| **Issue** | Depth scores produced implausible results: Tottenham 2.5 (unrealistically low), Fulham 72.5, Aston Villa 77.5, Coventry City 87.5 (unrealistically high for a promoted club). |
| **Root Cause** | Old formula used percentile rank of `Quality_Dropoff` (inverted) and `Depth_Ratio`. Squads with many players assigned the 50.0 Bayesian prior rating had artificially low quality dropoffs → incorrectly scored as high depth. |
| **Correction** | New formula: 50% Bench_Rating percentile + 30% Starting_XI percentile + 20% rotation index (playing time variance, inverted). |

**Before (incorrect):**
```python
# Quality_Dropoff = Starting_XI - Bench; inverted percentile rank
depth_score = (quality_dropoff_percentile_inv * 0.5 + depth_ratio_percentile * 0.5)
```
**After (corrected):**
```python
depth_score = (
    bench_rating_pct * 0.50 +
    xi_rating_pct * 0.30 +
    rotation_index_pct * 0.20
)
```

| **Corrected Scores** | Man City 88.0, Chelsea 82.5, Arsenal 75.5, Tottenham 66.5, Coventry 20.0 — plausible rankings. |
| **Validation** | Scores are grounded in bench quality and rotation patterns, not inverted prior-rating artefacts. |

---

### Issue 4 — FPL Data Loaded with Inner Join

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Location** | `src/data_loader.py` — `load_2026_27_fpl()` |
| **Issue** | FPL metadata and stats loaded with inner join — could silently drop players. |
| **Correction** | Changed to LEFT JOIN on FPL metadata as anchor. Added duplicate ID detection and coverage check. |
| **Validation** | 670 players, 0 duplicate IDs, 20 clubs, 100% coverage. Zero data loss from either join type (both sources had identical 670 records). |

---

### Issue 5 — `CalibratedClassifierCV(cv="prefit")` Invalid in Installed sklearn Version

| Field | Detail |
|-------|--------|
| **Severity** | High (caused training crash) |
| **Location** | `src/train.py` — calibration step |
| **Issue** | `cv="prefit"` parameter was removed in scikit-learn ≥1.2. Code crashed with `ValueError`. |
| **Root Cause** | sklearn version incompatibility. |
| **Correction** | Replaced with `PredefinedSplit` on concatenated train+val data. The split fold index correctly separates training from calibration data. |

```python
# Before (crashed):
calibrator = CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid")

# After (correct):
split_index = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold=split_index)
calibrator = CalibratedClassifierCV(best_model, cv=ps, method="sigmoid")
```

---

### Issue 6 — Feature Group Contamination (SoT Features in Wrong Group)

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Location** | `src/train.py` — feature group definitions |
| **Issue** | `Home_SoT_Last5`, `Away_SoT_Last5` etc. were placed in Model A (Team Strength Only). These are form-derived features. |
| **Correction** | Moved to Model B (Strength + Form). Model A now contains only the 12 pure team strength features. |
| **Validation** | Model A: 12 features ✓ | Model B: 36 features ✓ | Model C: 42 features ✓ | Model D: 50 features ✓ |

---

### Issue 7 — `evaluate.py` NameError for Unused Imports

| Field | Detail |
|-------|--------|
| **Severity** | Medium (caused crash) |
| **Location** | `src/evaluate.py` |
| **Issue** | `recall_score`, `precision_score`, `f1_score` referenced but not imported. |
| **Correction** | Added to sklearn imports. |

---

### Issue 8 — Windows Terminal UnicodeEncodeError

| Field | Detail |
|-------|--------|
| **Severity** | Low (cosmetic, not functional) |
| **Location** | Various `print()` statements |
| **Issue** | Special characters (arrows, Unicode) caused `UnicodeEncodeError` on Windows terminal. |
| **Correction** | Added `sys.stdout.reconfigure(encoding='utf-8')` at module start. |

---

### Issue 9 — Python Executable Confusion on Windows

| Field | Detail |
|-------|--------|
| **Severity** | High (all runs failed) |
| **Location** | Execution environment |
| **Issue** | `.\venv\Scripts\python.exe` resolves to Windows Store Python stub, not the project virtual environment. All module runs failed with wrong Python. |
| **Correction** | Use full system Python path with PYTHONPATH pointing to venv site-packages. |

**Correct invocation:**
```powershell
$env:PYTHONPATH = "F:\AIML Projects\Premier League Match Predictor Model\EPL Match Predictor\venv\Lib\site-packages;F:\AIML Projects\Premier League Match Predictor Model\EPL Match Predictor"
& "C:\Users\Vighnesh\AppData\Local\Programs\Python\Python312\python.exe" -m src.<module>
```

---

## 3. Before/After Model Performance

| Metric | Before Corrections | After Corrections | Delta |
|--------|--------------------|-------------------|-------|
| Test Accuracy | 45.72% | **49.21%** | **+3.49pp** |
| Test Log Loss | 1.07+ | **1.0199** | **−0.05** |
| Test Brier Score | 0.64+ | **0.6120** | **−0.03** |
| Draw predictions (standard) | 0 | 2 | — |
| Draw F1 (tuned) | N/A | **22.6%** | — |
| Club coverage | ≤20 (at risk) | **20/20** | ✅ |
| Player coverage | ≤670 (at risk) | **670/670** | ✅ |
| Depth score plausibility | ❌ Extreme values | ✅ Grounded scores | Fixed |

---

## 4. Pipeline Module Validation

All pipeline modules pass end-to-end after corrections:

| Module | Status | Output |
|--------|--------|--------|
| `src/data_loader.py` | ✅ | 4,180 matches; 20 clubs; 670 players |
| `src/player_mapper.py` | ✅ | 670 players mapped; 46 transfers; 0 ambiguous |
| `src/team_strength.py` | ✅ | 20 teams rated; plausible depth scores |
| `src/feature_engineering.py` | ✅ | 4,180 matches; 50 features; 0 nulls; 0 infinities |
| `src/train.py` | ✅ | 15 models trained; artifacts saved to `Models/` |
| `src/evaluate.py` | ✅ | Metrics saved to `reports/evaluation_metrics.json` |
| `src/predict.py` | ✅ | Probabilities sum to 100% for all test fixtures |
