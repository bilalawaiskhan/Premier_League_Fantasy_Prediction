"""Build next-event features from official FPL history and the preserved research data.

The three opponent-strength inputs are omitted from the live model because the
official API currently returns zero for its attack/defence strengths; those are
not equivalent to the 900-1300 vaastav training values. The remaining 70 fields
retain the original feature engineering definitions.
"""
from __future__ import annotations

import asyncio
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import SUM_COLS, load_raw, to_player_gw
import features as original_features
from features import build_dataset, feature_columns

OMITTED_FEATURES = {"opp_att_str", "opp_def_str", "opp_ovr_str"}
_today=datetime.now(timezone.utc)
_season_start=_today.year if _today.month>=7 else _today.year-1
SEASON=f"{_season_start}-{str(_season_start+1)[-2:]}"
HISTORY_FIELDS = {
    "total_points": "total_points", "minutes": "minutes", "starts": "starts",
    "goals_scored": "goals_scored", "assists": "assists", "bonus": "bonus", "bps": "bps",
    "expected_goals": "expected_goals", "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements", "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded", "saves": "saves", "ict_index": "ict_index",
    "threat": "threat", "creativity": "creativity", "influence": "influence",
    "expected_goals_conceded": "expected_goals_conceded", "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def api_history_to_raw(history: list[dict[str, Any]], player: dict[str, Any], teams: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Map official element-summary match records to vaastav raw row semantics."""
    name = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
    team = teams.get(int(player.get("team") or 0), {}).get("name", "Unknown")
    rows=[]
    for event in history:
        opponent = teams.get(int(event.get("opponent_team") or 0), {}).get("name", "Unknown")
        row={"name":name,"season":SEASON,"GW":int(event.get("round") or 0),"team":team,
             "opp_name":opponent,"opponent_team":event.get("opponent_team"),
             "was_home":int(bool(event.get("was_home"))),"value":_number(event.get("value")),
             "opp_strength_attack_home":0.0,"opp_strength_attack_away":0.0,
             "opp_strength_defence_home":0.0,"opp_strength_defence_away":0.0,
             "opp_strength_overall_home":0.0,"opp_strength_overall_away":0.0,
             "selected":_number(event.get("selected")),"xP":0.0}
        row.update({target:_number(event.get(source)) for target,source in HISTORY_FIELDS.items()})
        rows.append(row)
    return rows


def _candidate_raw(players: list[dict[str, Any]], fixtures: list[dict[str, Any]], gameweek: int,
                   teams: dict[int, dict[str, Any]], position_names: dict[int, str]) -> list[dict[str, Any]]:
    event_fixtures=[f for f in fixtures if f.get("event")==gameweek]
    by_team: dict[int,list[dict[str,Any]]]={}
    for f in event_fixtures:
        by_team.setdefault(int(f["team_h"]),[]).append(f)
        by_team.setdefault(int(f["team_a"]),[]).append(f)
    rows=[]
    for p in players:
        team_id=int(p.get("team") or 0); team=teams.get(team_id,{}).get("name","Unknown")
        entries=by_team.get(team_id) or [None]  # blank-gameweek players receive an explicit zero fixture count
        for fx in entries:
            opponent_id=(fx.get("team_a") if fx["team_h"]==team_id else fx.get("team_h")) if fx else 0
            row={"name":f"{p.get('first_name','')} {p.get('second_name','')}".strip(),"season":SEASON,
                 "GW":gameweek,"team":team,"position":position_names.get(int(p.get("element_type") or 0),"Unknown"),
                 "opp_name":teams.get(int(opponent_id or 0),{}).get("name","Unknown"),"opponent_team":opponent_id,
                 "was_home":int(bool(fx and fx.get("team_h")==team_id)),"value":_number(p.get("now_cost")),"selected":0,"xP":0.0,
                 "opp_strength_attack_home":0.0,"opp_strength_attack_away":0.0,
                 "opp_strength_defence_home":0.0,"opp_strength_defence_away":0.0,
                 "opp_strength_overall_home":0.0,"opp_strength_overall_away":0.0}
            row.update({c:0.0 for c in SUM_COLS})
            rows.append(row)
    return rows


def explain_prediction(row: pd.Series, contributions: np.ndarray, columns: list[str], player: dict[str,Any], fixture_text: str) -> dict[str,Any]:
    """Explain using only finite feature values and model contribution numbers."""
    effects=[]
    for name,value,effect in zip(columns,row[columns].to_numpy(dtype=float),contributions):
        if not np.isfinite(value) or not np.isfinite(effect):
            continue
        effects.append((abs(float(effect)),name,float(value),float(effect)))
    effects.sort(reverse=True)
    def display(item: tuple[float,str,float,float]) -> str:
        _,name,value,effect=item
        if name.endswith("_last_3") or name.endswith("_last_5") or name.endswith("_last_10"):
            base,window=name.rsplit("_last_",1)
            metric={"minutes":"minutes per appearance","starts":"starts per appearance","total_points":"points per appearance",
                    "expected_goals":"xG per appearance","expected_assists":"xA per appearance","expected_goal_involvements":"xGI per appearance",
                    "team_xg":"team xG per match","team_xgc":"team xGC per match","team_goals":"team goals per match",
                    "team_conceded":"team goals conceded per match","opp_team_xg":"opponent xG per match","opp_team_xgc":"opponent xGC per match",
                    "opp_team_goals":"opponent goals per match","opp_team_conceded":"opponent goals conceded per match"}.get(base,base.replace("_"," "))
            label=f"{metric} over the last {window} appearances/matches = {value:.2f}"
        elif name=="n_fixtures": label=f"scheduled fixtures this Gameweek = {value:.0f}"
        elif name=="price": label=f"price = £{value/10:.1f}m"
        elif name=="price_change_1": label=f"price change since last appearance = £{value/10:+.1f}m"
        elif name=="n_prev_apps": label=f"previous appearances in training history = {value:.0f}"
        elif name=="pos_code": label=f"position code = {value:.0f}"
        else: label=f"{name.replace('_',' ')} = {value:.2f}"
        direction="raised" if effect>=0 else "lowered"
        return f"{label}; the model {direction} this estimate by {abs(effect):.2f} points."
    reasons=[display(e) for e in effects[:2]]
    if len(reasons)<2:
        reasons=[f"{fixture_text}; the model has limited feature evidence for this estimate.",
                 f"Availability chance = {_number(player.get('chance_of_playing_next_round'),100):.0f}%; estimates are probabilistic."]
    chance=player.get("chance_of_playing_next_round")
    risk=player.get("news") or (f"Chance of playing next round: {chance}%" if chance is not None and chance<100 else None)
    return {"headline":fixture_text,"reasons":reasons,"risk":risk,
            "confidence":"Probabilistic estimate for guidance; outcomes are uncertain."}


async def build_live_predictions(bootstrap: dict[str,Any], fixtures: list[dict[str,Any]], history_getter) -> dict[str,Any]:
    events=bootstrap.get("events",[]); target=next((e for e in events if e.get("is_next")),None)
    if target is None:
        target=next((e for e in events if e.get("is_current") and not e.get("finished")),None)
    if target is None:
        return {"gameweek":None,"items":[],"prediction_status":"No upcoming Gameweek is currently published."}
    gw=int(target["id"]); teams={int(t["id"]):t for t in bootstrap.get("teams",[])}
    players=bootstrap.get("elements",[]); positions={int(p["id"]):p.get("singular_name_short","Unknown") for p in bootstrap.get("element_types",[])}
    semaphore=asyncio.Semaphore(8)
    async def history(p):
        # No current-season appearance means there is no current-season event history
        # to fetch; prior years are already seeded from the local training data.
        if _number(p.get("minutes"))<=0:
            return []
        async with semaphore:
            payload=await history_getter(int(p["id"]))
            return api_history_to_raw(payload.get("history",[]),p,teams)
    history_rows=await asyncio.gather(*(history(p) for p in players))
    historical_raw=load_raw()
    historical_raw=pd.concat([historical_raw,pd.DataFrame([r for group in history_rows for r in group])],ignore_index=True,sort=False)
    candidates=_candidate_raw(players,fixtures,gw,teams,positions)
    combined=pd.concat([historical_raw,pd.DataFrame(candidates)],ignore_index=True,sort=False)
    original_features.SEASON_IDX.setdefault(SEASON,max(original_features.SEASON_IDX.values())+1)
    frame=build_dataset(to_player_gw(combined))
    frame=frame[(frame.season==SEASON)&(frame.GW==gw)].copy()
    names={f"{p.get('first_name','')} {p.get('second_name','')}".strip():p for p in players}
    frame["n_fixtures"]=[int(sum(1 for f in fixtures if f.get("event")==gw and int(names.get(str(name),{}).get("team") or 0) in (int(f["team_h"]),int(f["team_a"])))) for name in frame.name]
    # The model build is intentionally limited to the 70 inputs whose definitions
    # can be reproduced from event history/fixtures; old strength columns are never substituted.
    model_path=Path(__file__).resolve().parents[3]/"outputs"/"xgb_live_model.joblib"
    if not model_path.is_file():
        return {"gameweek":gw,"items":[],"prediction_status":"Reduced live model has not been trained and validated."}
    import joblib
    artifact=joblib.load(model_path); model=artifact["model"]; columns=list(artifact["features"])
    expected=[c for c in feature_columns(frame) if c not in OMITTED_FEATURES]
    if columns!=expected:
        raise RuntimeError("Live feature order does not match the evaluated reduced model.")
    matrix=frame[columns]
    raw_predictions=model.predict(matrix)
    try:
        import xgboost as xgb
        contribution_matrix=model.get_booster().predict(xgb.DMatrix(matrix,feature_names=columns),pred_contribs=True)
    except Exception:
        contribution_matrix=np.zeros((len(frame),len(columns)+1))
    player_map={int(p["id"]):p for p in players}; fixture_labels={}
    team_fixture_counts={}
    for f in fixtures:
        if f.get("event")==gw:
            team_fixture_counts[int(f["team_h"])]=team_fixture_counts.get(int(f["team_h"]),0)+1
            team_fixture_counts[int(f["team_a"])]=team_fixture_counts.get(int(f["team_a"]),0)+1
    # Full names key historical rolling features. Resolve ids by the same name used in raw data.
    items=[]
    for idx,(_,row) in enumerate(frame.iterrows()):
        player=names.get(row["name"],{})
        team_name=str(row["team"]); matching=[f for f in fixtures if f.get("event")==gw and team_name in (teams.get(int(f["team_h"]),{}).get("name"),teams.get(int(f["team_a"]),{}).get("name"))]
        if matching:
            f=matching[0]; home=teams[int(f["team_h"])]["name"]==team_name; opponent=teams[int(f["team_a"] if home else f["team_h"])]["name"]
            fixture_text=f"{row['name']} ({row['team']}) vs {opponent} ({'H' if home else 'A'})"
        else:
            opponent="Blank Gameweek"; home=False; fixture_text=f"{row['name']} ({row['team']}) has no fixture in GW{gw}"
        chance=player.get("chance_of_playing_next_round")
        chance_factor=1.0 if chance is None else max(0.0,min(1.0,float(chance)/100))
        adjusted=0.0 if not matching else max(0.0,float(raw_predictions[idx])*chance_factor)
        fixture_text+=f", predicted {adjusted:.1f} pts"
        explanation=explain_prediction(row,contribution_matrix[idx,:-1],columns,player,fixture_text)
        pos=positions.get(int(player.get("element_type") or 0),str(row.get("position","Unknown")))
        if pos=="GKP":
            pos="GK"  # normalize the official API's label to the builder/optimizer schema
        items.append({"id":int(player.get("id",0)),"name":row["name"],"team":row["team"],"team_short":teams.get(int(player.get("team") or 0),{}).get("short_name",""),
                      "position":pos,"price":_number(player.get("now_cost"))/10,"ownership":_number(player.get("selected_by_percent")),
                      "availability_percent":chance,"news":player.get("news") or "","form":_number(player.get("form")),
                      "fixture_count":int(team_fixture_counts.get(int(player.get("team") or 0),0)),"opponent":opponent,"was_home":home,
                      "predicted_points":round(adjusted,3),"model_points":round(float(raw_predictions[idx]),3),
                      "features":{c:None if not np.isfinite(row[c]) else float(row[c]) for c in columns},"explanation":explanation})
    return {"gameweek":gw,"model":"xgb_live_model.joblib","model_scope":"Live 70-feature model, three unavailable opponent-strength inputs omitted; validation report linked in docs/live_model_validation.json.",
            "input_count":len(columns),"omitted_features":sorted(OMITTED_FEATURES),"items":items,"prediction_status":"Live model estimate; probabilistic guidance, not a guarantee."}
