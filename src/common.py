import pandas as pd, numpy as np
from pathlib import Path
from data import load_raw, to_player_gw
from features import build_dataset, feature_columns

OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG, TAB = OUT / "figures", OUT / "tables"
TRAIN_S, VAL_S, TEST_S = ["2022-23", "2023-24"], ["2024-25"], ["2025-26"]


def get_dataset():
    return build_dataset(to_player_gw(load_raw()))


def eligible(df, min_apps=3):
    """Rows we evaluate/train on: player has >=3 prior appearances (history exists)."""
    return df[df.n_prev_apps >= min_apps].copy()


def regulars(df):
    """'Relevant' players: averaged >=30 min over their last 5 appearances (realistic transfer candidates)."""
    return df[df.minutes_last_5 >= 30]
