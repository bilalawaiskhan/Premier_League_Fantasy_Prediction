"""Load + clean raw FPL data (vaastav/Fantasy-Premier-League, 4 seasons with xG/xA)."""
import pandas as pd, numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
SUM_COLS = ["total_points", "minutes", "starts", "goals_scored", "assists", "bonus", "bps",
            "clean_sheets", "goals_conceded", "saves", "yellow_cards", "red_cards",
            "expected_goals", "expected_assists", "expected_goal_involvements",
            "expected_goals_conceded", "creativity", "influence", "threat", "ict_index"]


def load_raw() -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        d = pd.read_csv(DATA / f"gw_{s}.csv", low_memory=False)
        tm = pd.read_csv(DATA / f"teams_{s}.csv").set_index("id")
        d["season"] = s
        d["opp_name"] = d["opponent_team"].map(tm["name"])
        for c in ["strength_attack_home", "strength_attack_away", "strength_defence_home",
                  "strength_defence_away", "strength_overall_home", "strength_overall_away"]:
            d["opp_" + c] = d["opponent_team"].map(tm[c])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["GW"] = df["GW"].astype(int)
    return df


def to_player_gw(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season, GW). Double gameweeks are summed; blank GWs have no row."""
    df = df.copy()
    df["was_home"] = df["was_home"].astype(int)
    df["opp_att_str"] = np.where(df.was_home == 1, df.opp_strength_attack_away, df.opp_strength_attack_home)
    df["opp_def_str"] = np.where(df.was_home == 1, df.opp_strength_defence_away, df.opp_strength_defence_home)
    df["opp_ovr_str"] = np.where(df.was_home == 1, df.opp_strength_overall_away, df.opp_strength_overall_home)
    g = df.groupby(["name", "season", "GW"], sort=False)
    agg = g[SUM_COLS].sum()
    agg["n_fixtures"] = g.size()
    for c in ["was_home", "opp_att_str", "opp_def_str", "opp_ovr_str", "value", "selected"]:
        agg[c] = g[c].mean()
    for c in ["position", "team", "opp_name"]:
        agg[c] = g[c].first()
    agg["xP"] = g["xP"].sum()
    return agg.reset_index().sort_values(["name", "season", "GW"]).reset_index(drop=True)
