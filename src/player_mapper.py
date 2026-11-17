"""Current-season player contribution ratings; no 2025/26 player priors are used."""
from __future__ import annotations
import pandas as pd


def build_current_player_ratings(players: pd.DataFrame) -> pd.DataFrame:
    """Rank current FPL points within position to form a relative contribution index.

    The supplied FPL snapshot does not include minutes, goals, xG, tackles or other
    granular player metrics. This score is therefore explicitly an FPL contribution
    percentile, not a complete scouting/ability grade.
    """
    required = {"id", "player_name", "team_name", "position_group", "total_points", "event_points"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"Current FPL player table is missing columns: {sorted(missing)}")
    result = players.copy()
    result["FPL_Contribution_Rating"] = (
        result.groupby("position_group")["total_points"].rank(method="average", pct=True) * 100.0
    ).round(2)
    result["FPL_Event_Points"] = result["event_points"].astype(float)
    result["Player_ID"] = result["id"].astype(int)
    result = result.rename(columns={"team_name": "Club", "position_group": "Position"})
    return result.sort_values(["Club", "Position", "FPL_Contribution_Rating"], ascending=[True, True, False]).reset_index(drop=True)
