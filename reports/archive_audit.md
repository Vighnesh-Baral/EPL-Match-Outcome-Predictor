# Input archive audit — Version 2

## Files inspected

The uploaded `EPL_Match_Predictor_Prototype(1).zip` was extracted into a separate Version 2 working copy. The uploaded ZIP was not modified.

## Verified source datasets

| Input | Coverage | Version 2 use |
|---|---:|---|
| 2026/27 league table | 20 clubs; 9 matches played per club | Current team metrics, form and current Poisson branch |
| 2026/27 FBref team statistics | 20 clubs per file | Current attack/midfield/defence/goalkeeper component ratings |
| 2026/27 FPL metadata | 670 unique player IDs | Current squad composition and position |
| 2026/27 FPL player statistics | 670 unique player IDs | Current FPL points and event points |
| Historical match results | 4,180 fixtures, 2015/16–2025/26 | Recency-weighted supervised training and H2H features/display |

## Missing input

No fixture-by-fixture 2026/27 result file was present. The current 2026/27 table and FBref files are aggregate snapshots. They are used for current team ratings and the Poisson branch, not misrepresented as individual match rows for classifier training.

## Data limitations

The current FPL snapshot contains total points, event points, cost, club and position, but no complete player minutes, starts, goals, assists, xG, tackles or interceptions. Current player contribution values are not complete football ability ratings. The quality-ranked starting XI/reserve selections are proxies, not actual team lineups or availability predictions.

## Legacy-model policy

The V2 historical model is newly constructed from the supplied 2015/16–2025/26 match-result CSVs with chronological features and the season-weight hierarchy described in the README. It does not reuse any legacy model artifacts from the original user's earlier project.
