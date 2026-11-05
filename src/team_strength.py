"""2026/27-only team strength indices from current FBref and FPL data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TEAMS_2026_27
from src.data_loader import load_current_players, load_current_team_data, load_historical_matches
from src.player_mapper import build_current_player_ratings


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = _numeric(series)
    median = values.median()
    if pd.isna(median):
        return pd.Series(50.0, index=series.index)
    values = values.fillna(median)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = values.rank(method="average", pct=True) * 100.0
    return ranks if higher_is_better else (100.0 - ranks)


def _mean_scores(frame: pd.DataFrame, columns: list[tuple[str, bool]]) -> pd.Series:
    scores = [percentile_score(frame[col], higher) for col, higher in columns]
    return pd.concat(scores, axis=1).mean(axis=1)


def _select_squad(players: pd.DataFrame) -> dict:
    """Create a flexible 4-3-3 quality proxy plus a distinct seven-player reserve group.

    When a club has fewer than three FPL-listed forwards, unused midfielders can
    occupy attacking slots. The selection is a quality/depth proxy, not a predicted
    matchday lineup: the supplied snapshot has no minutes, starts or availability.
    """
    by_pos = {pos: players[players["Position"] == pos].sort_values("FPL_Contribution_Rating", ascending=False).copy()
              for pos in ["GK", "DEF", "MID", "FWD"]}
    gk_xi = by_pos["GK"].head(1)
    def_xi = by_pos["DEF"].head(4)
    mid_xi = by_pos["MID"].head(3)
    used_ids = set(pd.concat([gk_xi, def_xi, mid_xi])["Player_ID"].tolist())
    attack_pool = pd.concat([by_pos["FWD"], by_pos["MID"]], ignore_index=True)
    attack_pool = attack_pool[~attack_pool["Player_ID"].isin(used_ids)].sort_values("FPL_Contribution_Rating", ascending=False)
    attack_xi = attack_pool.head(3)
    xi = pd.concat([gk_xi, def_xi, mid_xi, attack_xi], ignore_index=True)
    used_ids = set(xi["Player_ID"].tolist())

    # Reserves: one different goalkeeper, then the six highest-rated unused outfielders.
    reserve_gk = by_pos["GK"][~by_pos["GK"]["Player_ID"].isin(used_ids)].head(1)
    used_ids.update(reserve_gk["Player_ID"].tolist())
    reserve_outfield = players[players["Position"].isin(["DEF", "MID", "FWD"]) & ~players["Player_ID"].isin(used_ids)]
    reserve_outfield = reserve_outfield.sort_values("FPL_Contribution_Rating", ascending=False).head(6)
    bench = pd.concat([reserve_gk, reserve_outfield], ignore_index=True)

    xi_mean = float(xi["FPL_Contribution_Rating"].mean()) if not xi.empty else 50.0
    bench_mean = float(bench["FPL_Contribution_Rating"].mean()) if not bench.empty else 50.0
    counts = {pos: int(len(by_pos[pos])) for pos in by_pos}
    coverage_ratio = np.mean([
        min(counts["GK"] / 2.0, 1.0),
        min(counts["DEF"] / 6.0, 1.0),
        min((counts["MID"] + counts["FWD"]) / 10.0, 1.0),
    ])
    depth = 0.75 * bench_mean + 0.25 * (coverage_ratio * 100.0)
    complete = len(xi) == 11 and len(bench) == 7 and xi["Player_ID"].is_unique and bench["Player_ID"].is_unique and not set(xi["Player_ID"]) & set(bench["Player_ID"])
    attack_candidates = pd.concat([by_pos["FWD"].head(3), by_pos["MID"].sort_values("FPL_Contribution_Rating", ascending=False).head(3)], ignore_index=True)
    return {
        "Squad_Size": int(len(players)),
        "GK_Count": counts["GK"], "DEF_Count": counts["DEF"],
        "MID_Count": counts["MID"], "FWD_Count": counts["FWD"],
        "Starting_XI_FPL_Rating": round(xi_mean, 2),
        "Bench_FPL_Rating": round(bench_mean, 2),
        "Squad_Depth_Score": round(float(depth), 2),
        "Squad_Selection_Complete": bool(complete),
        "FPL_Attack_Rating": round(float(attack_candidates["FPL_Contribution_Rating"].mean()), 2),
        "FPL_Midfield_Rating": round(float(by_pos["MID"].head(3)["FPL_Contribution_Rating"].mean()), 2),
        "FPL_Defence_Rating": round(float(by_pos["DEF"].head(4)["FPL_Contribution_Rating"].mean()), 2),
        "FPL_GK_Rating": round(float(by_pos["GK"].head(1)["FPL_Contribution_Rating"].mean()), 2),
    }


def compute_team_strength() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = load_current_team_data()
    raw_players = load_current_players()
    players = build_current_player_ratings(raw_players)

    if set(teams["Squad"]) != set(TEAMS_2026_27):
        raise ValueError("Team statistics do not cover the configured 20 clubs.")
    if set(players["Club"]) != set(TEAMS_2026_27):
        raise ValueError("FPL player data do not cover the configured 20 clubs.")

    squad_rows = []
    for team in TEAMS_2026_27:
        squad_rows.append({"Squad": team, **_select_squad(players[players["Club"] == team])})
    squad = pd.DataFrame(squad_rows)
    df = teams.merge(squad, on="Squad", how="left", validate="one_to_one")

    # Performance indicators are all 2026/27 club-level observations.
    mp = df["MP"].replace(0, np.nan)
    df["GF_per_match"] = df["GF"] / mp
    df["GA_per_match"] = df["GA"] / mp
    df["TklW_per_match"] = _numeric(df["misc_TklW"]) / mp
    df["Int_per_match"] = _numeric(df["misc_Int"]) / mp
    df["Ast_per_match"] = _numeric(df["std_Ast"]) / mp
    df["SoTA_per_match"] = _numeric(df["gk_SoTA"]) / mp

    perf_attack = _mean_scores(df, [
        ("GF_per_match", True), ("shoot_SoT/90", True), ("shoot_Sh/90", True), ("shoot_G/Sh", True)
    ])
    perf_mid = _mean_scores(df, [
        ("std_Poss", True), ("Ast_per_match", True), ("TklW_per_match", True), ("Int_per_match", True)
    ])
    perf_def = _mean_scores(df, [
        ("GA_per_match", False), ("SoTA_per_match", False), ("gk_CS%", True), ("Int_per_match", True)
    ])
    perf_gk = _mean_scores(df, [
        ("gk_Save%", True), ("gk_GA90", False), ("gk_CS%", True)
    ])

    # Equal 50/50 blending is an explicit starting heuristic: measured team output
    # and current FPL contribution are both 2026/27 data. It is not fitted yet.
    df["Attacking_Strength"] = (0.50 * perf_attack + 0.50 * df["FPL_Attack_Rating"]).round(2)
    df["Midfield_Strength"] = (0.50 * perf_mid + 0.50 * df["FPL_Midfield_Rating"]).round(2)
    df["Defensive_Strength"] = (0.50 * perf_def + 0.50 * df["FPL_Defence_Rating"]).round(2)
    df["Goalkeeping_Strength"] = (0.50 * perf_gk + 0.50 * df["FPL_GK_Rating"]).round(2)

    # Current-season rating is primary, but a small 2025/26 Premier League
    # prior prevents nine matches from completely erasing established strength.
    # The prior is based on last season's points, goal difference and goal rates.
    component_cols = ["Attacking_Strength", "Midfield_Strength", "Defensive_Strength", "Goalkeeping_Strength", "Squad_Depth_Score"]
    df["Current_Season_Strength"] = df[component_cols].mean(axis=1)

    previous = load_historical_matches()
    previous = previous[previous["Season"].astype(str) == "2025-26"].copy()
    if not previous.empty:
        rows = []
        for club in TEAMS_2026_27:
            home = previous[previous["HomeTeam"] == club]
            away = previous[previous["AwayTeam"] == club]
            mp = len(home) + len(away)
            gf = float(home["FTHG"].sum() + away["FTAG"].sum())
            ga = float(home["FTAG"].sum() + away["FTHG"].sum())
            points = float((home["FTR"] == "H").sum() * 3 + (home["FTR"] == "D").sum() +
                           (away["FTR"] == "A").sum() * 3 + (away["FTR"] == "D").sum())
            rows.append({"Squad": club, "Prior_MP_2025_26": mp, "Prior_PPM_2025_26": points / mp if mp else np.nan,
                         "Prior_GDPM_2025_26": (gf - ga) / mp if mp else np.nan,
                         "Prior_GFPM_2025_26": gf / mp if mp else np.nan,
                         "Prior_GAPM_2025_26": ga / mp if mp else np.nan})
        prior = pd.DataFrame(rows)
        prior["Prior_Season_Strength"] = (
            0.40 * percentile_score(prior["Prior_PPM_2025_26"]) +
            0.30 * percentile_score(prior["Prior_GDPM_2025_26"]) +
            0.15 * percentile_score(prior["Prior_GFPM_2025_26"]) +
            0.15 * percentile_score(prior["Prior_GAPM_2025_26"], higher_is_better=False)
        ).round(2)
        df = df.merge(prior, on="Squad", how="left", validate="one_to_one")
    else:
        df["Prior_MP_2025_26"] = 0
        df["Prior_Season_Strength"] = np.nan

    # Clubs without 2025/26 Premier League records (promoted clubs) receive a
    # neutral prior, not invented Championship performance. Prior influence
    # starts at 15% and fades with current-season matches played.
    df["Prior_Season_Strength"] = pd.to_numeric(df["Prior_Season_Strength"], errors="coerce").fillna(50.0)
    df["Prior_Season_Weight"] = (0.15 * np.exp(-pd.to_numeric(df["MP"], errors="coerce").fillna(0) / 18.0)).round(4)
    df["Overall_Team_Strength"] = (
        (1.0 - df["Prior_Season_Weight"]) * df["Current_Season_Strength"] +
        df["Prior_Season_Weight"] * df["Prior_Season_Strength"]
    ).round(2)

    # Parse current table's Last 5 form string (e.g. "L W W W W").
    def points_from_form(value: object) -> tuple[int, str]:
        tokens = str(value).strip().upper().replace(",", " ").split()
        tokens = [t for t in tokens if t in {"W", "D", "L"}][-5:]
        points = sum(3 if t == "W" else 1 if t == "D" else 0 for t in tokens)
        return points, " ".join(tokens)

    form_rows = []
    for _, row in df.iterrows():
        points, form_string = points_from_form(row.get("Last 5", ""))
        form_rows.append({
            "Squad": row["Squad"], "Last_5": form_string,
            "Last_5_Points": points, "Last_5_PPM": points / max(len(form_string.split()), 1),
            "Season_PPM": float(row["Pts"]) / float(row["MP"]),
            "Goals_Scored_Per_Match": float(row["GF_per_match"]),
            "Goals_Conceded_Per_Match": float(row["GA_per_match"]),
            "Goal_Difference_Per_Match": float(row["GD"]) / float(row["MP"]),
        })
    form = pd.DataFrame(form_rows)
    form["Form_Rating_Last_5"] = (percentile_score(form["Last_5_PPM"])).round(2)

    out_cols = [
        "Squad", "Rk", "MP", "W", "D", "L", "Pts", "GF", "GA", "GD", "GF_per_match", "GA_per_match",
        "Attacking_Strength", "Midfield_Strength", "Defensive_Strength", "Goalkeeping_Strength",
        "Starting_XI_FPL_Rating", "Bench_FPL_Rating", "Squad_Depth_Score", "Current_Season_Strength", "Prior_Season_Strength", "Prior_Season_Weight", "Prior_MP_2025_26", "Overall_Team_Strength",
        "FPL_Attack_Rating", "FPL_Midfield_Rating", "FPL_Defence_Rating", "FPL_GK_Rating",
        "Squad_Size", "GK_Count", "DEF_Count", "MID_Count", "FWD_Count", "Squad_Selection_Complete",
    ]
    team_strength = df[out_cols].sort_values("Overall_Team_Strength", ascending=False).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(PROCESSED_DATA_DIR / "player_ratings_2026_27.csv", index=False)
    team_strength.to_csv(PROCESSED_DATA_DIR / "team_strength_2026_27.csv", index=False)
    form.to_csv(PROCESSED_DATA_DIR / "team_recent_form_2026_27.csv", index=False)
    return team_strength, form, players


if __name__ == "__main__":
    strength, form, players = compute_team_strength()
    print(f"Built ratings for {len(strength)} clubs and {len(players)} current FPL players.")
    print(strength[["Squad", "Attacking_Strength", "Midfield_Strength", "Defensive_Strength", "Goalkeeping_Strength", "Squad_Depth_Score", "Overall_Team_Strength"]].to_string(index=False))
