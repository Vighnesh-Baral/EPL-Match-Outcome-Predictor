# 2026/27 Team Strength Engine & Squad Depth Report
**EPL Match Outcome Predictor â€” 2026/27**

## 1. Architectural Philosophy

The 2026/27 Team Strength Rating is the foundation of our match prediction system (Priority 1 in our hierarchy).

Rather than simply averaging all registered players or relying on FPL popularity/price, the Team Strength Engine synthesizes:
1. **Current 2026/27 Season Performance (65% Weight):**
   - Actual matchday data from the completed 9 rounds of 2026/27.
   - Goals scored, shooting accuracy, conversion rates, possession, pass progression, tackles won, goals conceded, clean sheets, and save percentages.
2. **Transfer-Aware Squad Quality & Depth (35% Weight):**
   - Starting XI tactical selection (1 GK, 4 DEF, 3 MID, 3 FWD).
   - Tactical role scores incorporating individual attribute profiles.
   - Bench depth quality and drop-off analysis.

---

## 2. Component Construction

### 2.1 Attacking Strength
- **Current Performance Metrics:** Goals scored per match (`GF / MP`), Shots on Target per 90 (`shoot_SoT/90`), Shot Conversion (`shoot_G/Sh`), Total Shots per 90 (`shoot_Sh/90`).
- **Squad Unit:** Top 3 attacking forwards/midfielders Attack Role score (`0.55*Attack + 0.35*Creativity + 0.10*Defending`).
- **Composite:** 65% Performance + 35% Squad Attacking Quality.

### 2.2 Midfield Strength
- **Current Performance Metrics:** Possession percentage (`std_Poss`), Assists per match (`std_Ast / MP`), Tackles won in midfield (`misc_TklW / MP`), Press resistance / fouls drawn (`misc_Fld / MP`).
- **Squad Unit:** Top 3 starting midfielders Midfield Role score (`0.25*Attack + 0.50*Creativity + 0.25*Defending`).
- **Composite:** 65% Performance + 35% Squad Midfield Quality.

### 2.3 Defensive Strength
- **Current Performance Metrics:** Goals conceded per match (`GA / MP`, inverted), Shots on target allowed (`gk_SoTA / MP`, inverted), Clean sheet percentage (`gk_CS%`), Interceptions per match (`misc_Int / MP`).
- **Squad Unit:** Top 4 starting defenders Defense Role score (`0.10*Attack + 0.20*Creativity + 0.70*Defending`).
- **Composite:** 65% Performance + 35% Squad Defensive Quality.

### 2.4 Goalkeeping Strength
- **Current Performance Metrics:** Save percentage (`gk_Save%`), Goals against per 90 (`gk_GA90`, inverted), Clean sheet percentage (`gk_CS%`). (Raw save volume is deliberately excluded to avoid rewarding teams with porous defenses).
- **Squad Unit:** Primary goalkeeper overall rating.
- **Composite:** 65% Performance + 35% Goalkeeper Quality.

### 2.5 Squad Depth & Balance
- **Starting XI:** Selected by tactical unit from highest-rated players.
- **Bench:** Next 7 highest-rated squad players across positions (1 GK, 2 DEF, 2 MID, 2 FWD).
- **Quality Drop-Off:** $\max(0, \text{Starting XI} - \text{Bench})$.
- **Depth Ratio:** $\frac{\text{Bench Rating}}{\text{Starting XI Rating}}$.
- Teams with high bench quality retain higher resilience against injuries and fixture congestion.

---

## 3. 2026/27 Premier League Team Strength Leaderboard

| Rank | Club | Overall Strength | Attacking Strength | Midfield Strength | Defensive Strength | Goalkeeping Strength | Squad Depth Score | Starting XI Rating |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Manchester City** | **73.6** | 91.7 | 81.3 | 55.4 | 59.4 | 55.5 | 63.7 |
| 2 | **Brighton** | **72.6** | 86.3 | 74.3 | 64.9 | 63.6 | 37.5 | 60.0 |
| 3 | **Arsenal** | **70.7** | 63.9 | 67.8 | 85.0 | 79.5 | 42.5 | 61.4 |
| 4 | **Chelsea** | **65.1** | 78.4 | 74.1 | 48.4 | 49.3 | 50.5 | 62.9 |
| 5 | **Liverpool** | **62.6** | 56.7 | 68.6 | 62.2 | 65.8 | 62.5 | 60.2 |
| 6 | **Brentford** | **59.9** | 64.5 | 50.8 | 66.8 | 58.7 | 51.5 | 62.5 |
| 7 | **Bournemouth** | **58.0** | 67.6 | 49.2 | 56.6 | 59.0 | 50.5 | 58.2 |
| 8 | **Tottenham Hotspur** | **57.0** | 52.8 | 57.8 | 77.3 | 48.4 | 2.5 | 63.1 |
| 9 | **Leeds United** | **51.8** | 44.5 | 51.5 | 64.0 | 48.4 | 50.5 | 61.9 |
| 10 | **Everton** | **50.0** | 33.9 | 43.1 | 75.3 | 66.9 | 18.0 | 60.0 |
| 11 | **Manchester United** | **48.4** | 53.6 | 55.9 | 34.1 | 63.1 | 32.5 | 61.2 |
| 12 | **Newcastle United** | **47.6** | 54.3 | 45.4 | 45.2 | 37.7 | 56.5 | 62.4 |
| 13 | **Hull City** | **45.5** | 45.6 | 45.2 | 47.9 | 40.8 | 44.5 | 53.9 |
| 14 | **Ipswich Town** | **44.9** | 59.1 | 39.4 | 33.6 | 47.8 | 47.5 | 56.2 |
| 15 | **Fulham** | **43.1** | 33.2 | 43.1 | 56.8 | 32.5 | 72.5 | 58.1 |
| 16 | **Aston Villa** | **38.6** | 35.6 | 40.5 | 36.3 | 37.9 | 77.5 | 58.9 |
| 17 | **Sunderland** | **35.1** | 24.3 | 31.8 | 56.0 | 33.6 | 7.5 | 52.9 |
| 18 | **Nottingham Forest** | **34.7** | 23.4 | 43.5 | 45.0 | 25.1 | 38.5 | 56.8 |
| 19 | **Crystal Palace** | **32.8** | 36.8 | 37.4 | 19.9 | 38.0 | 60.0 | 57.0 |
| 20 | **Coventry City** | **29.3** | 28.1 | 24.0 | 33.5 | 13.9 | 87.5 | 50.1 |

---

## 4. Analytical Observations

1. **Top Tier (Arsenal, Man City, Brighton):**
   - Arsenal displays dominant defensive ratings (85.0 Defensive Strength, only 5 goals conceded in 9 matches).
   - Manchester City and Brighton lead in attacking metrics (91.7 and 86.3 Attacking Strength), reflecting explosive early-season goal generation (20 and 24 goals respectively).
2. **Mid-Table Congestion (50â€“60 range):**
   - Chelsea, Liverpool, Brentford, Bournemouth, and Tottenham show competitive parity, separated by defensive consistency and squad depth.
3. **Newly Promoted & Relegation Contenders:**
   - Coventry City and Sunderland show low performance metrics across early fixtures, despite respectable depth resilience from Coventry.
