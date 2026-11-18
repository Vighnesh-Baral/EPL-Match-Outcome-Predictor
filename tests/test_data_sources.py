"""Data-source parser tests (OpenFootball layouts, strictness, cross-source agreement)."""
import pandas as pd
import pytest

from src.data_loader import load_historical_matches
from src.data_sources.openfootball import load_openfootball_season, parse_openfootball_text

LAYOUT_V = """= Premier League 2026/27
# Matches 3

▪ Matchday 1
  Sat Aug 22
    17:30  Arsenal FC v Chelsea FC  2-1 (1-0)
  Sun Aug 23
    16:30  Cardiff City v Brighton & Hove Albion  0-0
▪ Matchday 2
  Sat Aug 29
    15:00  Liverpool FC v Everton FC
"""

LAYOUT_MID = """= Premier League 2019/20
# Matches 3

▪ Matchday 1
Sat Jun 20
  15:00  Arsenal          1-0 (0-0)  Norwich City
Wed Jul 1
  19:00  Chelsea          2-2 (1-1)  Everton
Thu Jan 2
  20:00  Liverpool        3-0 (1-0)  Cardiff City
"""


def test_layout_v_parses_scores_unscored_and_aliases():
    df = parse_openfootball_text(LAYOUT_V, season="2026-27")
    assert len(df) == 3
    assert list(df["Played"]) == [True, True, False]
    assert df.loc[2, "FTHG"] != df.loc[2, "FTHG"]          # NaN for an unplayed fixture, never imputed
    assert df.loc[0, "FTR"] == "H" and df.loc[1, "FTR"] == "D" and df.loc[2, "FTR"] is None
    assert "Cardiff" in set(df["HomeTeam"])                 # 'Cardiff City' alias to the Football-Data name
    assert df.loc[0, "KickOff"] == "17:30"


def test_layout_mid_handles_year_rollover_and_july_restart():
    df = parse_openfootball_text(LAYOUT_MID, season="2019-20")
    assert [d.year for d in df["Date"]] == [2019, 2019, 2020] or [d.year for d in df["Date"]] == [2019, 2019, 2019 + 1]
    # July after June stays in the same calendar year; January after December/July rolls over.
    assert df.loc[1, "Date"] == pd.Timestamp("2019-07-01")
    assert df.loc[2, "Date"] == pd.Timestamp("2020-01-02")


def test_count_guard_raises_when_fixtures_are_skipped():
    bad = LAYOUT_V.replace("# Matches 3", "# Matches 4")
    with pytest.raises(ValueError):
        parse_openfootball_text(bad, season="2026-27")


def test_openfootball_agrees_with_football_data_for_every_historical_match():
    fd = load_historical_matches()
    total = 0
    for y in range(2015, 2026):
        season = f"{y}-{str(y + 1)[2:]}"
        of = load_openfootball_season(season)
        merged = of.merge(fd[fd["Season"] == season][["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]],
                          on=["HomeTeam", "AwayTeam"], suffixes=("_of", "_fd"))
        assert len(merged) == 380, season
        assert (merged["FTHG_of"] == merged["FTHG_fd"]).all() and (merged["FTAG_of"] == merged["FTAG_fd"]).all()
        assert (merged["Date_of"] == merged["Date_fd"]).all()
        total += len(merged)
    assert total == 4180


def test_current_season_schedule_is_complete_and_consistent():
    sched = load_openfootball_season("2026-27")
    assert len(sched) == 380 and sched.groupby("Matchday").size().eq(10).all()
    played = sched[sched["Played"]]
    assert played["Matchday"].max() <= 10 and played["FTHG"].notna().all()


# ----------------------------------------------------------------------------- Football-Data / FPL / StatsBomb (mocked network)
class _Resp:
    def __init__(self, text="", payload=None):
        self.text, self._payload = text, payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.resp


def test_football_data_download_splits_results_and_odds(tmp_path):
    from src.data_sources import football_data as fd
    csv = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,B365H,B365D,B365A\n"
           "E0,15/08/2025,Liverpool,Bournemouth,4,2,H,22,10,9,4,1.3,5.5,9.0\n"
           "E0,16/08/2025,Aston Villa,Newcastle,0,0,D,12,9,3,3,2.6,3.4,2.8\n")
    session = _Session(_Resp(text=csv))
    paths = fd.download_season("2025-26", session, tmp_path / "res", tmp_path / "odds")
    assert session.calls == ["https://www.football-data.co.uk/mmz4281/2526/E0.csv"]
    res, odds = pd.read_csv(paths["results"]), pd.read_csv(paths["odds"])
    assert "B365H" not in res.columns and "B365H" in odds.columns and len(res) == 2


def test_football_data_rejects_malformed_files():
    from src.data_sources import football_data as fd
    with pytest.raises(ValueError):
        fd.parse_csv("Date,HomeTeam\n01/01/2025,A\n")
    with pytest.raises(ValueError):
        fd.parse_csv("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/2025,A,B,1,0,X\n")
    assert fd.season_code("2025-26") == "2526" and fd.season_code("1999-00") == "9900"


def test_fpl_bootstrap_parsing_and_validation():
    from src.data_sources import fpl
    payload = {"teams": [{"id": 1, "name": "Arsenal"}],
               "elements": [{"id": 7, "web_name": "Saka", "team": 1, "element_type": 3, "minutes": 900,
                             "total_points": 80, "goals_scored": 5, "assists": 6, "now_cost": 101, "status": "a"}]}
    frame = fpl.bootstrap_to_players(payload)
    assert frame.loc[0, "club"] == "Arsenal" and frame.loc[0, "position"] == "MID" and frame.loc[0, "now_cost_m"] == 10.1
    with pytest.raises(ValueError):
        fpl.bootstrap_to_players({"teams": []})


def test_statsbomb_aggregation_excludes_shootouts_and_other_events():
    from src.data_sources.statsbomb import aggregate_events
    events = [
        {"type": {"name": "Shot"}, "period": 1, "team": {"name": "Chelsea"}, "shot": {"statsbomb_xg": 0.30}},
        {"type": {"name": "Shot"}, "period": 2, "team": {"name": "Chelsea"}, "shot": {"statsbomb_xg": 0.05}},
        {"type": {"name": "Shot"}, "period": 2, "team": {"name": "Arsenal"}, "shot": {"statsbomb_xg": 0.76}},
        {"type": {"name": "Shot"}, "period": 5, "team": {"name": "Arsenal"}, "shot": {"statsbomb_xg": 0.78}},
        {"type": {"name": "Pass"}, "period": 1, "team": {"name": "Arsenal"}},
    ]
    out = aggregate_events(events, "Chelsea", "Arsenal")
    assert out["home_xg"] == pytest.approx(0.35) and out["away_xg"] == pytest.approx(0.76)
    assert out["home_shots"] == 2 and out["away_shots"] == 1
