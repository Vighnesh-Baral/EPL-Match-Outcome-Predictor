"""Leakage-safe, single-code-path feature engine (PitchIQ v2.5).

The same ``FeatureEngine`` object builds features for historical training rows and for
live fixtures, which removes the train/serve skew present in v2.4.

Protocol (enforced and unit-tested):
  1. ``features(...)`` reads state only; it never mutates it.
  2. All fixtures on a calendar date are featurised *before* any of that date's
     results are applied with ``update(...)``.
  3. Results are applied strictly in date order.

State maintained (all from past results only):
  * Elo ratings (home-advantage in rating points, goal-margin multiplier, season
    regression, explicit promoted-club prior).
  * Online attack/defence ratings: a Poisson "goals-Elo" (SGD on the Poisson
    log-likelihood).  Home advantage enters through running home/away goal means; no
    hand-set multiplier is used.
  * Rolling 5/10-match form, goals, shots-on-target, venue-specific form,
    opponent-adjusted performance (result vs Elo expectation, goal difference vs
    rating expectation), rest days and fixture congestion, and head-to-head.

Short histories are padded with the running league average (documented shrinkage;
never a silent fill of an unavailable *statistic*).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

SEED_PTS_PER_MATCH = 1.37          # only used until real matches accumulate
SEED_GOALS_HOME, SEED_GOALS_AWAY = 1.50, 1.20
SEED_SOT, SEED_SHOTS = 4.3, 12.5


@dataclass
class RatingParams:
    elo_k: float = 12.0
    elo_hfa: float = 40.0                 # Elo points added to the home side's rating
    elo_season_regression: float = 0.20   # fraction pulled back to 1500 each new season
    elo_promoted_delta: float = -120.0    # start rating offset for clubs new to the league
    elo_mov: bool = True                  # goal-margin multiplier
    goal_eta: float = 0.015               # online attack/defence learning rate
    goal_season_regression: float = 0.10
    prior_strength: float = 30.0          # pseudo-matches for running league means

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_COLUMNS = [
    # Elo
    "elo_home", "elo_away", "elo_diff", "elo_exp_home",
    # online attack/defence goal ratings
    "att_home", "def_home", "att_away", "def_away", "lam_home", "lam_away", "lam_diff",
    # raw form (windows of 5 and 10)
    "h_ppm5", "a_ppm5", "h_ppm10", "a_ppm10",
    "h_gf5", "a_gf5", "h_ga5", "a_ga5", "h_gf10", "a_gf10", "h_ga10", "a_ga10",
    # opponent-adjusted form
    "h_perf5", "a_perf5", "h_perf10", "a_perf10",
    "h_gdx5", "a_gdx5", "h_gdx10", "a_gdx10",
    # venue-specific form
    "h_home_ppm5", "a_away_ppm5", "h_home_gd5", "a_away_gd5",
    # shots on target (for / against) over 5 and 10
    "h_sot5", "a_sot5", "h_sota5", "a_sota5", "h_sot10", "a_sot10", "h_sota10", "a_sota10",
    # schedule load (league fixtures only; see limitations)
    "h_rest", "a_rest", "h_n14", "a_n14", "h_n30", "a_n30",
    # sample-size context
    "h_mp_season", "a_mp_season",
    # head to head (shrunk toward the league average)
    "h2h_ppm", "h2h_gd", "h2h_n",
]

# Candidate feature groups used by the ablation study.
FEATURE_GROUPS = {
    "elo": ["elo_home", "elo_away", "elo_diff", "elo_exp_home"],
    "goal_ratings": ["att_home", "def_home", "att_away", "def_away", "lam_home", "lam_away", "lam_diff"],
    "raw_form": ["h_ppm5", "a_ppm5", "h_ppm10", "a_ppm10", "h_gf5", "a_gf5", "h_ga5", "a_ga5",
                 "h_gf10", "a_gf10", "h_ga10", "a_ga10"],
    "opp_adjusted": ["h_perf5", "a_perf5", "h_perf10", "a_perf10", "h_gdx5", "a_gdx5", "h_gdx10", "a_gdx10"],
    "venue": ["h_home_ppm5", "a_away_ppm5", "h_home_gd5", "a_away_gd5"],
    "shots": ["h_sot5", "a_sot5", "h_sota5", "a_sota5", "h_sot10", "a_sot10", "h_sota10", "a_sota10"],
    "schedule": ["h_rest", "a_rest", "h_n14", "a_n14", "h_n30", "a_n30"],
    "context": ["h_mp_season", "a_mp_season"],
    "h2h": ["h2h_ppm", "h2h_gd", "h2h_n"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURE_COLUMNS)


def _padded_mean(values: list[float], window: int, prior: float) -> float:
    """Mean of the last ``window`` values; missing slots filled with ``prior``."""
    recent = values[-window:]
    missing = window - len(recent)
    return (sum(recent) + missing * prior) / window


class FeatureEngine:
    def __init__(self, params: RatingParams | None = None):
        self.params = params or RatingParams()
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)
        self.att: dict[str, float] = defaultdict(float)
        self.dfn: dict[str, float] = defaultdict(float)
        self.hist: dict[str, list[dict]] = defaultdict(list)
        self.venue_hist: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.pair_hist: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.season: str | None = None
        self.season_teams: set[str] = set()
        self.prev_teams: set[str] = set()
        self.season_mp: dict[str, int] = defaultdict(int)
        self.season_gf: dict[str, float] = defaultdict(float)
        self.season_ga: dict[str, float] = defaultdict(float)
        self.was_promoted: set[str] = set()
        self.promoted_cohort: list[tuple[float, float]] = []   # (att, def) of promoted clubs' first seasons
        # running league sums (prior results only)
        self.n_matches = 0
        self.sum_home_goals = 0.0
        self.sum_away_goals = 0.0
        self.sum_pts = 0.0
        self.sum_sot = 0.0
        self.sum_shots = 0.0
        self.n_team_matches = 0

    # ------------------------------------------------------------------ league priors
    def league_means(self) -> dict[str, float]:
        k = self.params.prior_strength
        n = self.n_matches
        nt = self.n_team_matches
        return {
            "goals_home": (self.sum_home_goals + k * SEED_GOALS_HOME) / (n + k),
            "goals_away": (self.sum_away_goals + k * SEED_GOALS_AWAY) / (n + k),
            "ppm": (self.sum_pts + k * SEED_PTS_PER_MATCH) / (nt + k),
            "sot": (self.sum_sot + k * SEED_SOT) / (nt + k),
            "shots": (self.sum_shots + k * SEED_SHOTS) / (nt + k),
        }

    # ------------------------------------------------------------------ season handling
    def _finalise_season(self) -> None:
        if self.season is None:
            return
        total_gf = sum(self.season_gf.values())
        total_mp = sum(self.season_mp.values())
        if total_mp == 0:
            return
        league_rate = total_gf / total_mp
        for team in self.was_promoted:
            mp = self.season_mp.get(team, 0)
            if mp >= 10 and league_rate > 0:
                att = math.log(max(self.season_gf[team] / mp, 0.2) / league_rate)
                dfn = -math.log(max(self.season_ga[team] / mp, 0.2) / league_rate)
                self.promoted_cohort.append((att, dfn))

    def start_season(self, season: str, teams: Iterable[str]) -> None:
        teams = set(teams)
        p = self.params
        first_ever = self.season is None
        self._finalise_season()
        prior_att = float(np.mean([a for a, _ in self.promoted_cohort])) if self.promoted_cohort else 0.0
        prior_def = float(np.mean([d for _, d in self.promoted_cohort])) if self.promoted_cohort else 0.0
        self.was_promoted = set()
        for team in teams:
            if first_ever:
                continue
            if team in self.prev_teams:
                self.elo[team] = (1 - p.elo_season_regression) * self.elo[team] + p.elo_season_regression * 1500.0
                self.att[team] *= (1 - p.goal_season_regression)
                self.dfn[team] *= (1 - p.goal_season_regression)
            else:  # promoted, or returning after at least one season away: stale ratings are discarded
                self.elo[team] = 1500.0 + p.elo_promoted_delta
                self.att[team] = prior_att
                self.dfn[team] = prior_def
                self.was_promoted.add(team)
        self.prev_teams = teams
        self.season_teams = teams
        self.season = season
        self.season_mp = defaultdict(int)
        self.season_gf = defaultdict(float)
        self.season_ga = defaultdict(float)

    # ------------------------------------------------------------------ expectations
    def elo_expected_home(self, home: str, away: str) -> float:
        diff = self.elo[home] - self.elo[away] + self.params.elo_hfa
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def goal_lambdas(self, home: str, away: str) -> tuple[float, float]:
        lm = self.league_means()
        lam_h = lm["goals_home"] * math.exp(self.att[home] - self.dfn[away])
        lam_a = lm["goals_away"] * math.exp(self.att[away] - self.dfn[home])
        return float(np.clip(lam_h, 0.15, 5.0)), float(np.clip(lam_a, 0.15, 5.0))

    # ------------------------------------------------------------------ features (read-only)
    def features(self, home: str, away: str, date: pd.Timestamp) -> dict[str, float]:
        lm = self.league_means()
        hh, ah = self.hist[home], self.hist[away]
        hv, av = self.venue_hist[(home, "H")], self.venue_hist[(away, "A")]
        lam_h, lam_a = self.goal_lambdas(home, away)
        e_home = self.elo_expected_home(home, away)

        def roll(hist: list[dict], key: str, n: int, prior: float) -> float:
            return _padded_mean([r[key] for r in hist], n, prior)

        def sched(hist: list[dict]) -> tuple[float, float, float]:
            if not hist:
                return 14.0, 0.0, 0.0
            rest = min((date - hist[-1]["date"]).days, 14)
            n14 = sum(1 for r in hist[-6:] if (date - r["date"]).days <= 14)
            n30 = sum(1 for r in hist[-10:] if (date - r["date"]).days <= 30)
            return float(max(rest, 1)), float(n14), float(n30)

        h_rest, h_n14, h_n30 = sched(hh)
        a_rest, a_n14, a_n30 = sched(ah)

        pair = self.pair_hist[tuple(sorted((home, away)))][-10:]
        k = 3.0
        pts = gd = 0.0
        for m in pair:
            if m["home"] == home:
                hg, ag = m["hg"], m["ag"]
            else:
                hg, ag = m["ag"], m["hg"]
            pts += 3.0 if hg > ag else 1.0 if hg == ag else 0.0
            gd += hg - ag
        n_pair = len(pair)
        h2h_ppm = (pts + k * lm["ppm"]) / (n_pair + k)
        h2h_gd = gd / (n_pair + k)

        row = {
            "elo_home": self.elo[home], "elo_away": self.elo[away],
            "elo_diff": self.elo[home] - self.elo[away], "elo_exp_home": e_home,
            "att_home": self.att[home], "def_home": self.dfn[home],
            "att_away": self.att[away], "def_away": self.dfn[away],
            "lam_home": lam_h, "lam_away": lam_a, "lam_diff": lam_h - lam_a,
            "h_ppm5": roll(hh, "pts", 5, lm["ppm"]), "a_ppm5": roll(ah, "pts", 5, lm["ppm"]),
            "h_ppm10": roll(hh, "pts", 10, lm["ppm"]), "a_ppm10": roll(ah, "pts", 10, lm["ppm"]),
            "h_gf5": roll(hh, "gf", 5, 1.35), "a_gf5": roll(ah, "gf", 5, 1.35),
            "h_ga5": roll(hh, "ga", 5, 1.35), "a_ga5": roll(ah, "ga", 5, 1.35),
            "h_gf10": roll(hh, "gf", 10, 1.35), "a_gf10": roll(ah, "gf", 10, 1.35),
            "h_ga10": roll(hh, "ga", 10, 1.35), "a_ga10": roll(ah, "ga", 10, 1.35),
            "h_perf5": roll(hh, "perf", 5, 0.0), "a_perf5": roll(ah, "perf", 5, 0.0),
            "h_perf10": roll(hh, "perf", 10, 0.0), "a_perf10": roll(ah, "perf", 10, 0.0),
            "h_gdx5": roll(hh, "gdx", 5, 0.0), "a_gdx5": roll(ah, "gdx", 5, 0.0),
            "h_gdx10": roll(hh, "gdx", 10, 0.0), "a_gdx10": roll(ah, "gdx", 10, 0.0),
            "h_home_ppm5": roll(hv, "pts", 5, lm["ppm"]), "a_away_ppm5": roll(av, "pts", 5, lm["ppm"]),
            "h_home_gd5": roll(hv, "gd", 5, 0.0), "a_away_gd5": roll(av, "gd", 5, 0.0),
            "h_sot5": roll(hh, "sot", 5, lm["sot"]), "a_sot5": roll(ah, "sot", 5, lm["sot"]),
            "h_sota5": roll(hh, "sota", 5, lm["sot"]), "a_sota5": roll(ah, "sota", 5, lm["sot"]),
            "h_sot10": roll(hh, "sot", 10, lm["sot"]), "a_sot10": roll(ah, "sot", 10, lm["sot"]),
            "h_sota10": roll(hh, "sota", 10, lm["sot"]), "a_sota10": roll(ah, "sota", 10, lm["sot"]),
            "h_rest": h_rest, "a_rest": a_rest, "h_n14": h_n14, "a_n14": a_n14, "h_n30": h_n30, "a_n30": a_n30,
            "h_mp_season": float(self.season_mp[home]), "a_mp_season": float(self.season_mp[away]),
            "h2h_ppm": h2h_ppm, "h2h_gd": h2h_gd, "h2h_n": float(n_pair),
        }
        return row

    # ------------------------------------------------------------------ update (mutating)
    def update(self, date: pd.Timestamp, home: str, away: str, hg: float, ag: float,
               hst: float | None = None, ast: float | None = None,
               hs: float | None = None, as_: float | None = None) -> None:
        p = self.params
        lm = self.league_means()
        lam_h, lam_a = self.goal_lambdas(home, away)
        e_home = self.elo_expected_home(home, away)
        e_away = 1.0 - e_home
        s_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        s_away = 1.0 - s_home
        pts_h = 3.0 if hg > ag else 1.0 if hg == ag else 0.0
        pts_a = 3.0 if ag > hg else 1.0 if hg == ag else 0.0
        sot_h = lm["sot"] if hst is None or pd.isna(hst) else float(hst)
        sot_a = lm["sot"] if ast is None or pd.isna(ast) else float(ast)

        rec_h = {"date": date, "pts": pts_h, "gf": hg, "ga": ag, "gd": hg - ag,
                 "perf": s_home - e_home, "gdx": (hg - ag) - (lam_h - lam_a), "sot": sot_h, "sota": sot_a}
        rec_a = {"date": date, "pts": pts_a, "gf": ag, "ga": hg, "gd": ag - hg,
                 "perf": s_away - e_away, "gdx": (ag - hg) - (lam_a - lam_h), "sot": sot_a, "sota": sot_h}
        self.hist[home].append(rec_h)
        self.hist[away].append(rec_a)
        self.venue_hist[(home, "H")].append(rec_h)
        self.venue_hist[(away, "A")].append(rec_a)
        self.pair_hist[tuple(sorted((home, away)))].append({"home": home, "hg": hg, "ag": ag})

        # Elo
        margin = abs(hg - ag)
        mult = 1.0
        if p.elo_mov:
            mult = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11.0 + margin) / 8.0
        self.elo[home] += p.elo_k * mult * (s_home - e_home)
        self.elo[away] += p.elo_k * mult * (s_away - e_away)

        # online attack / defence (Poisson SGD, capped residuals)
        err_h = float(np.clip(hg - lam_h, -3.0, 3.0))
        err_a = float(np.clip(ag - lam_a, -3.0, 3.0))
        self.att[home] += p.goal_eta * err_h
        self.dfn[away] -= p.goal_eta * err_h
        self.att[away] += p.goal_eta * err_a
        self.dfn[home] -= p.goal_eta * err_a

        # running league sums and season tallies
        self.n_matches += 1
        self.sum_home_goals += hg
        self.sum_away_goals += ag
        self.n_team_matches += 2
        self.sum_pts += pts_h + pts_a
        self.sum_sot += sot_h + sot_a
        self.sum_shots += (SEED_SHOTS if hs is None or pd.isna(hs) else float(hs)) + \
                          (SEED_SHOTS if as_ is None or pd.isna(as_) else float(as_))
        for team, gf, ga in ((home, hg, ag), (away, ag, hg)):
            self.season_mp[team] += 1
            self.season_gf[team] += gf
            self.season_ga[team] += ga


REQUIRED_COLUMNS = {"Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def build_feature_table(matches: pd.DataFrame, params: RatingParams | None = None,
                        season_teams: dict[str, Iterable[str]] | None = None,
                        engine: FeatureEngine | None = None) -> tuple[pd.DataFrame, FeatureEngine]:
    """Featurise every match in chronological order; returns the table and the final engine.

    ``season_teams`` optionally overrides the per-season club set (needed when a season's
    results are partial).  Otherwise it is derived from the season's own fixtures.
    """
    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(f"Matches missing columns: {sorted(missing)}")
    data = matches.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    eng = engine or FeatureEngine(params)
    if season_teams is None:
        season_teams = {s: set(g["HomeTeam"]) | set(g["AwayTeam"]) for s, g in data.groupby("Season")}
    rows: list[dict] = []
    for date, day in data.groupby("Date", sort=True):
        season = str(day["Season"].iloc[0])
        if eng.season != season:
            eng.start_season(season, season_teams[season])
        # 1) features for every fixture of the day, before any state change
        for r in day.itertuples(index=False):
            feats = eng.features(r.HomeTeam, r.AwayTeam, date)
            feats.update({"Date": date, "Season": season, "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
                          "FTHG": float(r.FTHG), "FTAG": float(r.FTAG)})
            rows.append(feats)
        # 2) only then apply the day's results
        for r in day.itertuples(index=False):
            eng.update(date, r.HomeTeam, r.AwayTeam, float(r.FTHG), float(r.FTAG),
                       getattr(r, "HST", None), getattr(r, "AST", None),
                       getattr(r, "HS", None), getattr(r, "AS", None))
    table = pd.DataFrame(rows)
    table["FTR"] = np.where(table["FTHG"] > table["FTAG"], "H", np.where(table["FTHG"] < table["FTAG"], "A", "D"))
    table["y"] = table["FTR"].map({"H": 0, "D": 1, "A": 2}).astype(int)
    if not np.isfinite(table[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in the feature table.")
    return table, eng
