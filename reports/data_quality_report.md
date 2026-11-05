# Data Quality & Player Transfer Integration Report
**EPL Match Outcome Predictor — 2026/27**

## 1. Data Integrity and Cleaning Protocols

### 1.1 Team Name Standardization
Across the raw datasets, significant nomenclature variance existed:
- Match results: `"Man City"`, `"Man United"`, `"Nott'm Forest"`, `"Spurs"`, `"Hull"`, `"Ipswich"`, `"Leeds"`
- FBref 2026/27: `"Manchester City"`, `"Manchester Utd"`, `"Nottingham"`, `"Tottenham"`, `"Hull City"`, `"Ipswich Town"`, `"Leeds United"`
- FPL 2026/27: `"Man City"`, `"Man Utd"`, `"Nott'm Forest"`, `"Spurs"`, `"Leeds"`
- 2025/26 Stats: `"Liverpool FC"`, `"Brighton &amp; Hove Albion"`, `"Manchester United"`, `"Tottenham Hotspur"`

A canonical dictionary (`CANONICAL_TEAM_NAMES` in `src/config.py`) maps all variants to authoritative canonical representations, eliminating join mismatches across all 20 active 2026/27 clubs and 34 historical clubs.

### 1.2 Text Normalization & Encoding Mojibake Resolution
Raw player records contained HTML entities (`&amp;`) and Latin-1/UTF-8 mojibake (`Gabriel Magalh\ufffdes`, `Mart\ufffdn Zubimendi`, `Jurri\ufffdn Timber`).
The cleaning pipeline:
1. Decodes HTML entities via `html.unescape`.
2. Repairs character mojibake via Unicode NFKD normalization and ASCII transliteration.
3. Removes punctuation, hyphens, and accents.
4. Generates lowercased alphabetic tokens for robust token-subset comparison.

---

## 2. Transfer-Aware Player Integration

### 2.1 Principle of Transfer Invariance
A player who competed for Club A in 2025/26 and transferred to Club B for 2026/27 must carry their historical performance priors exclusively to Club B. Attributing their historical output to Club A's current strength would introduce severe squad attribution error.

### 2.2 Matching Methodology
- **Step 1: Squad Definition:** 2026/27 FPL dataset (670 players) establishes official 2026/27 squad membership.
- **Step 2: Position Filtering:** Strict positional alignment (`GKP` ↔ `G`, `DEF` ↔ `D`, `MID` ↔ `M`, `FWD` ↔ `F`).
- **Step 3: Multi-Stage Name Matching:**
  - Stage A: Exact normalized full display name.
  - Stage B: Exact normalized `first_name + second_name`.
  - Stage C: Unambiguous token subset matching (accommodating Portuguese/Spanish compound patronymics, e.g., *Bruno Borges Fernandes* ↔ *Bruno Fernandes*, *David Raya Martín* ↔ *David Raya*, *Ezri Konsa Ngoyo* ↔ *Ezri Konsa*).
- **Step 4: Ambiguity Quarantine:** If a player matches multiple records with equal token compatibility across different players, matching is quarantined (0 false force-matches permitted).

### 2.3 Empirical Bayesian Prior for Missing Historical Data
Players lacking 2025/26 Premier League statistics (e.g. transfers from foreign leagues, academy promotions, and promoted clubs such as Coventry City, Hull City, and Sunderland) are **never assigned zero ability**. They are assigned an empirical Bayesian prior:
- Outfield: Median rating of active Premier League players in their position (~50.0).
- Goalkeepers: Median goalkeeper rating (~50.0).

---

## 3. Transfer Matching Results

- **Total 2026/27 Squad Members:** 670 players
- **2025/26 Historical Records Matched:** 358 players
- **Detected Club-to-Club Transfers:** 46 verified transfers
- **Ambiguous Matches Quarantined:** 0
- **Bayesian Empirical Priors Assigned:** 314 players (Championship promotions and overseas arrivals)

### Notable Verified Transfers Carried to 2026/27 Squads:
| Player Name | 2025/26 Former Club | 2026/27 Current Club | Position | 2025/26 Minutes | Prior Rating Carried Over |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Morgan Rogers** | Aston Villa | Chelsea | M | 3,105 | 68.4 |
| **Ezri Konsa** | Aston Villa | Arsenal | D | 2,856 | 62.0 |
| **Emiliano Martínez** | Aston Villa | Chelsea | G | 2,655 | 64.5 |
| **Joao Gomes** | Wolves | Aston Villa | M | 2,664 | 58.9 |
| **Pascal Struijk** | Leeds United | Brighton | D | 2,791 | 68.7 |
| **Marcos Senesi** | Bournemouth | Tottenham Hotspur | D | 3,020 | 66.2 |
| **Aaron Wan-Bissaka** | West Ham | Aston Villa | D | 1,931 | 54.2 |
| **Callum Wilson** | West Ham | Brentford | F | 1,110 | 56.8 |
| **Harvey Elliott** | Aston Villa | Liverpool | M | 1,420 | 62.1 |
| **Christian Nørgaard** | Arsenal | Everton | M | 1,840 | 58.3 |

All 46 transfers have been integrated into `Data/Processed/player_mapping_2026_27.csv`.
