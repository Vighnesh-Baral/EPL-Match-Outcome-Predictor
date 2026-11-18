# Data sources, licences and coverage (v2.5)

| Source | What is used | Licence / terms | Status in this build |
|---|---|---|---|
| **Football-Data.co.uk** | 4,180 historical results 2015-16 to 2025-26 with shots, shots on target, corners, fouls, cards (already in `Data/Raw/match_results/`) | Free for non-commercial use; credit requested | Local files used. `src/data_sources/football_data.py` can download seasons and store bookmaker odds separately; **not run live** (host unreachable from the build sandbox), covered by mocked tests. The shipped CSVs contain no odds, so no bookmaker benchmark is reported. |
| **OpenFootball** (`openfootball/england`) | Season files 2015-16 to 2026-27 | CC0 / public domain | Downloaded and cached in `Data/Raw/openfootball/`. Cross-validated against Football-Data: **4,180 of 4,180 results and dates identical**. Supplies 2026/27 fixture dates, kick-off times and 50 results (Matchdays 1-5). |
| **StatsBomb Open Data** | Real xG for Premier League 2015-16 (380 matches) | StatsBomb Open Data User Agreement: **attribution required** ("Data provided by StatsBomb") | Downloaded and aggregated (`Data/Processed/statsbomb_pl_2015_16_xg.csv`). Only one of the 11 seasons has xG, so it is used only to validate shots-based features (`reports/v25/xg_validation.json`), not as a model input. |
| **FPL API** (`fantasy.premierleague.com`) | Current squad/player data (existing snapshot in `Data/Raw/2026_27/`) | Free public endpoint; unofficial use | Existing 4 Nov 2026 snapshot displayed in Team/Player Intelligence. `src/data_sources/fpl.py` parses live payloads; **not run live** (unreachable). |
| FPL historical archive (`vaastav/Fantasy-Premier-League`, MIT) | Would enable backtestable player-strength features | MIT | **Not integrated.** See limitations. |

## Coverage and gaps (stated, not hidden)
- 2026/27: 60 of the 100 played matches are known at match level (OpenFootball Matchdays 1-5 plus the repository log,
  deduplicated; no conflicts, none exceed the standings totals). The other 40 are **not imputed**. Ratings are primed with
  the 60 known results; the Matchday-10 standings remain the source for the table.
- Shots-on-target for 2026/27 results are unavailable; those values are skipped in rolling averages, then padded with the
  running league average (documented shrinkage), never filled with invented per-match numbers.
- Schedule load covers league fixtures only (no cups or European football in the free sources).
- No lineup, injury or suspension data exists in any free source used here.
