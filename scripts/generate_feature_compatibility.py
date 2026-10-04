"""Emit an evidence-first inventory of the model's required features."""
import csv
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from data import load_raw,to_player_gw
from features import build_dataset,feature_columns

dataset=build_dataset(to_player_gw(load_raw()))
features=feature_columns(dataset)
writer=csv.writer(sys.stdout)
writer.writerow(["feature","historical source / transformation","current FPL source candidate","availability / compatibility","missing-data policy","leakage consideration"])
for feature in features:
    if feature.endswith("_last_3") or feature.endswith("_last_5") or feature.endswith("_last_10") or feature.endswith("_ewm") or feature.startswith("points_last_"):
        hist="Past player Gameweek stats; shifted rolling mean or exponential weighting in src/features.py"
        live="element-summary history, after a season/player-identity mapping is validated"
    elif feature.startswith("opp_") or feature.startswith("team_"):
        hist="Historical team/opponent aggregates and strength fields, shifted over prior Gameweeks"
        live="bootstrap teams plus fixtures/history; matching semantics and pre-deadline cut-off unverified"
    elif feature in {"price","price_change_1","points_per_million"}:
        hist="Historical player value and derived change / value ratio"
        live="bootstrap element value; price history and exact transformation unverified"
    elif feature in {"n_fixtures","was_home","pos_code"}:
        hist="Historical fixture aggregation and player metadata"
        live="fixtures/bootstrap metadata; double/blank Gameweek aggregation needs mapping"
    else:
        hist="Past player/team Gameweek statistics, engineered by src/features.py"
        live="element-summary history where present; field names and aggregation unverified"
    writer.writerow([feature,hist,live,"UNVERIFIED: no current-season compatibility adapter","Mark prediction unavailable; do not silently fill with zero","Only records available before the target Gameweek may be used; verify shift/cut-off before inference"])
