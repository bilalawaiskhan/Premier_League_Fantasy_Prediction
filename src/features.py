"""Feature engineering. GOLDEN RULE: every feature for GW t is built only from rows < t (shift(1) first)."""
import pandas as pd, numpy as np
from data import SEASONS

ROLL_BASE = ["total_points", "minutes", "starts", "goals_scored", "assists", "bonus", "bps",
             "expected_goals", "expected_assists", "expected_goal_involvements",
             "clean_sheets", "goals_conceded", "saves", "ict_index", "threat", "creativity", "influence"]
WINDOWS = [3, 5, 10]
SEASON_IDX = {s: i for i, s in enumerate(SEASONS)}


def _order(df):
    df = df.copy()
    df["t"] = df["season"].map(SEASON_IDX) * 100 + df["GW"]
    return df.sort_values(["name", "t"]).reset_index(drop=True)


def add_player_features(df):
    df = _order(df)
    g = df.groupby("name", sort=False)
    prev = g[ROLL_BASE].shift(1)                      # <- shift(1): only the past is visible
    prev_g = prev.groupby(df["name"], sort=False)
    out = {"points_last_1": prev["total_points"]}
    for w in WINDOWS:
        r = prev_g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        for c in ROLL_BASE:
            out[f"{c}_last_{w}"] = r[c]
    out["points_ewm"] = prev_g["total_points"].transform(lambda s: s.ewm(alpha=0.4).mean())
    out["minutes_ewm"] = prev_g["minutes"].transform(lambda s: s.ewm(alpha=0.4).mean())
    out["points_std_last_10"] = prev_g["total_points"].rolling(10, min_periods=3).std().reset_index(level=0, drop=True)
    for k in range(1, 6):                              # raw lags, used by the baselines
        out[f"p_lag{k}"] = g["total_points"].shift(k)
    out["n_prev_apps"] = g.cumcount()
    out["price_change_1"] = df["value"] - g["value"].shift(1)
    out["points_per_million"] = out["total_points_last_5"] / (df["value"] / 10)
    return pd.concat([df, pd.DataFrame(out)], axis=1)


def add_team_features(df):
    """Team attack/defence form from team-level xG & xGC per GW, shifted by one GW."""
    tg = (df.groupby(["team", "season", "GW"])
            .agg(team_xg=("expected_goals", "sum"), team_xgc=("expected_goals_conceded", "max"),
                 team_goals=("goals_scored", "sum"), team_conceded=("goals_conceded", "max")).reset_index())
    tg["t"] = tg["season"].map(SEASON_IDX) * 100 + tg["GW"]
    tg = tg.sort_values(["team", "t"])
    gg = tg.groupby("team", sort=False)
    for c in ["team_xg", "team_xgc", "team_goals", "team_conceded"]:
        tg[c + "_last_5"] = gg[c].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    keep = ["team", "season", "GW"] + [c for c in tg.columns if c.endswith("_last_5")]
    t1 = tg[keep]
    df = df.merge(t1, on=["team", "season", "GW"], how="left")
    opp = t1.rename(columns={"team": "opp_name", **{c: "opp_" + c for c in t1.columns if c.endswith("_last_5")}})
    return df.merge(opp, on=["opp_name", "season", "GW"], how="left")


def build_dataset(pgw):
    df = add_team_features(add_player_features(pgw))
    df["pos_code"] = df["position"].map({"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "AM": 2, "FWD": 3})
    df["price"] = df["value"] / 10
    df["target"] = df["total_points"]
    return df


def feature_columns(df):
    raw = set(ROLL_BASE) | {"expected_goals_conceded", "yellow_cards", "red_cards"}
    skip = {"name", "season", "GW", "t", "team", "opp_name", "position", "target", "value", "selected", "xP"} | raw
    skip |= {c for c in df.columns if c.startswith("p_lag")}
    return [c for c in df.columns if c not in skip]
