"""API for historical FPL research and clearly-labeled live data."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import asyncio
import math
from functools import lru_cache
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app import database
from app.services.fpl_client import get_json as fpl_get_json, cache_state as fpl_cache_state, close_clients as close_fpl_clients
from app.services.live_predictions import build_live_predictions

ROOT = Path(os.getenv("FPL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "outputs"
DATA = ROOT / "data"
app = FastAPI(title="FPL AI", version="1.0.0", description="Historical model research and current FPL data. Historical projections are never presented as live forecasts.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","), allow_methods=["*"], allow_headers=["*"])

class PlayerInput(BaseModel):
    id: int
    name: str
    team: str
    position: str
    price: float = Field(ge=0)
    projected_points: float

class OptimizeRequest(BaseModel):
    players: list[PlayerInput]
    budget: float = Field(default=100, gt=0, le=200)

class TransferRequest(BaseModel):
    outgoing: PlayerInput
    candidates: list[PlayerInput]
    budget: float = Field(ge=0, le=200)

class SquadSaveRequest(BaseModel):
    name: str = Field(default="My squad", min_length=1, max_length=80)
    players: list[dict[str, Any]]
    budget: float = Field(default=100, gt=0, le=200)

_live_predictions: dict[str, Any] | None = None
_live_predictions_at: float = 0.0
_live_predictions_lock = asyncio.Lock()

@app.on_event("startup")
def startup(): database.init_db()

@app.on_event("shutdown")
async def shutdown():
    await close_fpl_clients()

def predictions() -> pd.DataFrame:
    path = OUT / "test_predictions.csv"
    if not path.exists():
        raise HTTPException(503, "Historical test prediction file is unavailable.")
    return pd.read_csv(path)

def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas / NumPy scalars and NaNs to strict JSON-safe values."""
    result=[]
    for raw in frame.to_dict("records"):
        row={}
        for key,value in raw.items():
            if value is None or (not isinstance(value,(list,dict)) and pd.isna(value)):
                row[key]=None
            elif hasattr(value,"item"):
                row[key]=value.item()
            else: row[key]=value
        result.append(row)
    return result

@lru_cache(maxsize=1)
def load_saved_artifact():
    """Load the preserved model once per worker; restart after replacing the artifact."""
    import joblib
    artifact=OUT/"xgb_model.joblib"
    if not artifact.is_file(): raise FileNotFoundError("Saved XGBoost artifact is missing.")
    return joblib.load(artifact)

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status":"ok", "model_artifact_available": (OUT / "xgb_model.joblib").is_file(),
            "live_model_available":(OUT/"xgb_live_model.joblib").is_file(),
            "historical_predictions_available": (OUT / "test_predictions.csv").is_file(),
            "dataset_seasons":[p.stem.removeprefix("gw_") for p in sorted(DATA.glob("gw_*.csv"))],
            "prediction_scope":"2025-26 historical evaluation; 2026-27 live-compatible reduced model"}

@app.get("/api/squads")
def get_squads(): return {"items":database.list_squads()}

@app.post("/api/squads")
def create_squad(req: SquadSaveRequest):
    try: return database.save_squad(req.name,req.players,req.budget)
    except Exception as exc: raise HTTPException(400,"Squad could not be saved. Check that all player data is valid JSON.") from exc

@app.put("/api/squads/{squad_id}")
def update_squad(squad_id: int, req: SquadSaveRequest):
    row=database.save_squad(req.name,req.players,req.budget,squad_id)
    if row is None: raise HTTPException(404,"Squad not found.")
    return row

@app.delete("/api/squads/{squad_id}")
def remove_squad(squad_id: int):
    if not database.delete_squad(squad_id): raise HTTPException(404,"Squad not found.")
    return {"deleted":True,"id":squad_id}

@app.get("/api/players")
def players(search: str = "", position: str = "", team: str = "", sort: str = "points", limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    df = predictions()
    df = df[df.season == "2025-26"]
    latest = int(df.GW.max())
    df = df[df.GW == latest].copy()
    if search: df = df[df.name.str.contains(search, case=False, na=False)]
    if position: df = df[df.position.str.upper() == position.upper()]
    if team: df = df[df.team.str.contains(team, case=False, na=False)]
    sort_col = {"points":"m_XGBoost", "price":"price", "name":"name", "form":"minutes_last_5"}.get(sort)
    if not sort_col: raise HTTPException(400, "sort must be points, price, name, or form")
    df = df.sort_values(sort_col, ascending=sort_col == "name")
    total = len(df)
    rows = df.iloc[offset:offset+limit].rename(columns={"m_XGBoost":"historical_model_prediction", "GW":"gameweek"})
    return {"scope":"Historical test data (2025-26); not current projections", "gameweek":latest, "total":total, "items":json_records(rows[["name","team","position","price","historical_model_prediction","minutes_last_5","gameweek"]])}

@app.get("/api/players/{player_name}/history")
def player_history(player_name: str):
    df = predictions()
    df = df[df.name.str.casefold() == player_name.casefold()]
    if df.empty: raise HTTPException(404, "Player not found in historical test predictions.")
    return {"scope":"Historical test data (2025-26)","name":player_name,"items":json_records(df[["GW","target","m_XGBoost","price","minutes_last_5"]].rename(columns={"GW":"gameweek","target":"actual_points","m_XGBoost":"historical_model_prediction"}))}

@app.get("/api/model-performance")
def model_performance():
    path=OUT/"tables"/"model_comparison.csv"
    if not path.exists(): raise HTTPException(503,"Existing model evaluation report is unavailable.")
    frame=pd.read_csv(path,header=[0,1],index_col=[0,1])
    rows=[]
    for (period,model),values in frame.iterrows():
        rows.append({"period":period,"model":model,**{str(k):None if pd.isna(v) else float(v) for k,v in values.items()}})
    return {"source":"Existing evaluation report; no retraining or recalculation", "items":rows}

@app.get("/api/feature-importance")
def feature_importance():
    path=OUT/"tables"/"xgb_feature_importance.csv"
    if not path.exists(): raise HTTPException(503,"Saved feature importance report is unavailable.")
    df=pd.read_csv(path)
    return {"source":"Existing XGBoost feature importance; association, not causation", "items":df.to_dict("records")}

@app.get("/api/model-compatibility")
def model_compatibility():
    path=ROOT/"docs"/"model_feature_compatibility.csv"
    if not path.exists(): raise HTTPException(503,"Generated 73-feature compatibility report is unavailable.")
    frame=pd.read_csv(path)
    return {"model_features":len(frame),"current_live_inference":"unverified; do not display current model projections","items":json_records(frame)}

@app.get("/api/live/model-validation")
def live_model_validation():
    path=ROOT/"docs"/"live_model_validation.json"
    if not path.is_file(): raise HTTPException(503,"Live-model validation report is missing.")
    import json
    return json.loads(path.read_text(encoding="utf-8"))

@app.post("/api/predict/prepared")
def predict_prepared(rows: list[dict[str, Any]]):
    """Run saved artifact inference only on fully prepared, feature-compatible rows."""
    if not rows: raise HTTPException(422,"At least one prepared row is required.")
    artifact=OUT/"xgb_model.joblib"
    if not artifact.is_file(): raise HTTPException(503,"Saved model artifact is unavailable.")
    try:
        saved=load_saved_artifact()
        model=saved["model"]; features=list(saved["features"])
        frame=pd.DataFrame(rows)
        missing=[col for col in features if col not in frame.columns]
        if missing: raise HTTPException(422,{"message":"Required model features are missing.","missing":missing})
        ignored=sorted(set(frame.columns)-set(features))
        X=frame.loc[:,features].apply(pd.to_numeric,errors="coerce")
        if X.isna().any().any(): raise HTTPException(422,"Required feature values must be numeric and non-missing.")
        preds=model.predict(X)
        return {"scope":"Saved model inference on caller-prepared rows", "model_version":"saved xgb_model.joblib", "features_used":features,"ignored_extra_columns":ignored,"predictions":[float(x) for x in preds]}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(503,f"Saved model inference failed: {type(exc).__name__}") from exc

@app.get("/api/predictions/historical-inference")
def historical_inference(limit: int = Query(50, ge=1, le=500)):
    """Rebuild features from existing 2025-26 historic rows and replay the saved model."""
    artifact=OUT/"xgb_model.joblib"
    if not artifact.is_file(): raise HTTPException(503,"Saved model artifact is unavailable.")
    src=ROOT/"src"
    if not src.is_dir(): raise HTTPException(503,"Original feature engineering source is unavailable.")
    try:
        sys.path.insert(0,str(src))
        from common import eligible
        from data import load_raw, to_player_gw
        from features import build_dataset
        frame=eligible(build_dataset(to_player_gw(load_raw())))
        frame=frame[frame.season=="2025-26"].sort_values(["GW","name"]).tail(limit)
        saved=load_saved_artifact(); model=saved["model"]; features=saved["features"]
        missing=[c for c in features if c not in frame.columns]
        if missing: raise HTTPException(503,{"message":"Rebuilt historic features do not match artifact.","missing":missing})
        result=frame[["name","team","position","season","GW","target"]].copy()
        result["new_saved_model_inference"]=model.predict(frame[features])
        recorded_path=OUT/"test_predictions.csv"
        if recorded_path.is_file():
            recorded=pd.read_csv(recorded_path)
            if {"name","season","GW","m_XGBoost"}.issubset(recorded.columns):
                keyed=recorded.set_index(["name","season","GW"])["m_XGBoost"]
                result["existing_test_prediction"]=[keyed.get((r.name,r.season,r.GW)) for r in result.itertuples()]
                result["inference_minus_recorded"]=result.new_saved_model_inference-result.existing_test_prediction
        return {"scope":"Newly generated historical inference on 2025-26 data; not a current prediction", "comparison":"Existing test predictions are a separate recorded output", "items":json_records(result.rename(columns={"GW":"gameweek","target":"actual_points"}))}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(503,f"Historical inference could not be completed: {type(exc).__name__}: {exc}") from exc
    finally:
        if str(src) in sys.path: sys.path.remove(str(src))

@app.get("/api/live/bootstrap")
async def live_bootstrap():
    """Fetch genuine live player/team metadata; deliberately makes no ML forecasts."""
    try:
        data=await fpl_get_json("bootstrap-static/",ttl=300)
        for key in ("events","teams","element_types","elements"):
            if not isinstance(data.get(key),list): raise ValueError(f"unexpected response: {key} is not a list")
        return {"scope":"Live official FPL metadata", "events":data.get("events",[]),"teams":data.get("teams",[]),"positions":data.get("element_types",[]),"players":data.get("elements",[]),"prediction_status":"Uses the separately validated 70-feature live model; three unavailable opponent-strength features are omitted.","data_status":fpl_cache_state("bootstrap-static/")}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(502, f"Official FPL data could not be retrieved: {type(exc).__name__}") from exc

@app.get("/api/gameweeks/current")
async def current_gameweek():
    try:
        data=await fpl_get_json("bootstrap-static/",ttl=300)
        events=data.get("events",[])
        current=next((e for e in events if e.get("is_current")),None)
        upcoming=next((e for e in events if e.get("is_next")),None)
        return {"scope":"Live official FPL calendar", "current":current,"next":upcoming,"data_status":fpl_cache_state("bootstrap-static/")}
    except (RuntimeError,ValueError) as exc: raise HTTPException(502,"Current official FPL Gameweek data is unavailable.") from exc

@app.get("/api/live/players")
async def live_players(search: str = "", position: str = "", team: str = "", sort: str = "points", limit: int = Query(50,ge=1,le=200), offset: int = Query(0,ge=0)):
    try:
        data=await fpl_get_json("bootstrap-static/",ttl=300)
        teams={t["id"]:t.get("name","Unknown") for t in data["teams"]}
        positions={p["id"]:p.get("singular_name_short","Unknown") for p in data["element_types"]}
        items=[]
        for p in data["elements"]:
            pos=positions.get(p.get("element_type"),"Unknown"); club=teams.get(p.get("team"),"Unknown")
            name=p.get("web_name","Unknown")
            if search and search.casefold() not in name.casefold(): continue
            if position and position.casefold()!=pos.casefold(): continue
            if team and team.casefold() not in club.casefold(): continue
            items.append({"id":p.get("id"),"name":name,"team":club,"position":pos,"price":(p.get("now_cost") or 0)/10,"season_points":p.get("total_points"),"form":p.get("form"),"minutes":p.get("minutes"),"status":p.get("status"),"availability_percent":p.get("chance_of_playing_next_round")})
        sort_map={"points":"season_points","price":"price","form":"form","name":"name"}
        if sort not in sort_map: raise HTTPException(400,"sort must be points, price, form, or name")
        key=sort_map[sort]
        items.sort(key=lambda p: str(p[key] or "") if key=="name" else float(p[key] or 0),reverse=key!="name")
        return {"scope":"Live official FPL player metadata and season-to-date statistics; no model projection", "total":len(items),"items":items[offset:offset+limit]}
    except HTTPException: raise
    except (RuntimeError,KeyError,ValueError) as exc: raise HTTPException(502,"Current official FPL player data was unavailable or had an unsupported response.") from exc

@app.get("/api/live/fixtures")
async def live_fixtures():
    try:
        fixtures,static=await asyncio.gather(fpl_get_json("fixtures/",ttl=300),fpl_get_json("bootstrap-static/",ttl=300))
        teams={t["id"]:t.get("name",str(t["id"])) for t in static["teams"]}
        for fixture in fixtures:
            fixture["team_h_name"]=teams.get(fixture.get("team_h"),"Unknown")
            fixture["team_a_name"]=teams.get(fixture.get("team_a"),"Unknown")
        return {"scope":"Live official FPL fixtures", "items":fixtures,"data_status":fpl_cache_state("fixtures/","bootstrap-static/")}
    except (RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(502, f"Official FPL fixtures could not be retrieved: {type(exc).__name__}") from exc

@app.get("/api/live/events/{gameweek}/")
async def live_event(gameweek: int):
    if not 1<=gameweek<=50: raise HTTPException(422,"Gameweek must be between 1 and 50.")
    try: return {"scope":"Live official FPL Gameweek points", "gameweek":gameweek,"data":await fpl_get_json(f"event/{gameweek}/live/",ttl=60),"data_status":fpl_cache_state(f"event/{gameweek}/live/")}
    except RuntimeError as exc: raise HTTPException(502,"Official FPL live Gameweek data could not be retrieved.") from exc

@app.get("/api/live/players/{player_id}/history")
async def live_player_history(player_id: int):
    if player_id<1: raise HTTPException(422,"Player id must be positive.")
    try: return {"scope":"Live official FPL player history", "player_id":player_id,"data":await fpl_get_json(f"element-summary/{player_id}/",ttl=300)}
    except RuntimeError as exc: raise HTTPException(502,"Official FPL player history could not be retrieved.") from exc

@app.get("/api/live/entry/{team_id}/event/{gameweek}/picks")
async def live_entry_picks(team_id: int, gameweek: int):
    """Attempt public picks retrieval; private/unavailable entries fall back to manual input."""
    if team_id<1 or not 1<=gameweek<=50: raise HTTPException(422,"Team id and Gameweek must be positive valid values.")
    try:
        result,static=await asyncio.gather(fpl_get_json(f"entry/{team_id}/event/{gameweek}/picks/",ttl=60),fpl_get_json("bootstrap-static/",ttl=300))
        if not isinstance(result,dict) or not isinstance(result.get("picks"),list): raise ValueError("Unexpected public picks response schema")
        player_map={p["id"]:p for p in static.get("elements",[])}
        teams={t["id"]:t.get("name","Unknown") for t in static.get("teams",[])}
        positions={p["id"]:("GK" if p.get("singular_name_short")=="GKP" else p.get("singular_name_short","Unknown")) for p in static.get("element_types",[])}
        picks=[]
        for pick in result["picks"]:
            player=player_map.get(pick.get("element"),{})
            picks.append({"id":player.get("id"),"name":player.get("web_name","Unknown"),"team":teams.get(player.get("team"),"Unknown"),"position":positions.get(player.get("element_type"),"Unknown"),"price":pick.get("selling_price",player.get("now_cost",0))/10,"slot":pick.get("position"),"is_captain":pick.get("is_captain",False),"is_vice_captain":pick.get("is_vice_captain",False)})
        entry_history=result.get("entry_history") or {}
        budget=((entry_history.get("value") or 0)+(entry_history.get("bank") or 0))/10
        return {"scope":"Official public FPL picks endpoint", "team_id":team_id,"gameweek":gameweek,"budget":budget,"items":picks,"data_status":fpl_cache_state(f"entry/{team_id}/event/{gameweek}/picks/","bootstrap-static/")}
    except (RuntimeError,KeyError,ValueError) as exc:
        raise HTTPException(502,"Public FPL picks were unavailable or had an unsupported response. The entry may not be public or this Gameweek's picks may not be released; enter the squad manually or import JSON.") from exc

@app.get("/api/live/predictions")
async def get_live_predictions():
    """Cached next-Gameweek inference assembled from past rows only."""
    global _live_predictions, _live_predictions_at
    now=asyncio.get_running_loop().time()
    if _live_predictions is not None and now-_live_predictions_at<300:
        return {**_live_predictions,"data_status":{"stale":False,"last_success_at":_live_predictions.get("generated_at")}}
    async with _live_predictions_lock:
        now=asyncio.get_running_loop().time()
        if _live_predictions is not None and now-_live_predictions_at<300:
            return {**_live_predictions,"data_status":{"stale":False,"last_success_at":_live_predictions.get("generated_at")}}
        try:
            bootstrap,fixtures=await asyncio.gather(fpl_get_json("bootstrap-static/",ttl=300),fpl_get_json("fixtures/",ttl=300))
            async def history_getter(player_id: int):
                return await fpl_get_json(f"element-summary/{player_id}/",ttl=300)
            result=await build_live_predictions(bootstrap,fixtures,history_getter)
            result["generated_at"]=pd.Timestamp.now(tz="UTC").isoformat()
            stale=any(fpl_cache_state(f"element-summary/{int(p['id'])}/")["stale"] for p in bootstrap.get("elements",[]))
            stale=stale or fpl_cache_state("bootstrap-static/","fixtures/")["stale"]
            result["data_status"]={"stale":stale,"last_success_at":result["generated_at"]}
            if result.get("items"):
                _live_predictions=result;_live_predictions_at=now
            elif _live_predictions is not None:
                return {**_live_predictions,"data_status":{"stale":True,"last_success_at":_live_predictions.get("generated_at"),"message":"Refresh failed; showing the last successful prediction table."}}
            return result
        except Exception as exc:
            if _live_predictions is not None:
                return {**_live_predictions,"data_status":{"stale":True,"last_success_at":_live_predictions.get("generated_at"),"message":"Refresh failed; showing the last successful prediction table."}}
            raise HTTPException(502,f"Live prediction table unavailable: {type(exc).__name__}") from exc


@app.post("/api/live/auto-pick")
async def live_auto_pick():
    predictions=await get_live_predictions()
    candidates=predictions.get("items",[])
    if len(candidates)<15:
        raise HTTPException(503,predictions.get("prediction_status","Not enough player estimates are available."))
    posmap={"GK":0,"DEF":1,"MID":2,"FWD":3}
    frame=pd.DataFrame(candidates)
    frame["pos_code"]=frame.position.map(posmap)
    if frame.pos_code.isna().any(): raise HTTPException(503,"Some live players have an unsupported position.")
    try:
        if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
        from optimize import pick_team
        chosen=pick_team(frame,"predicted_points",budget=100)
    except (ValueError,ImportError) as exc:
        raise HTTPException(422,"No legal auto-picked squad could be found under £100m.") from exc
    by_id={p["id"]:p for p in candidates}
    selected=[]
    for picked in chosen.to_dict("records"):
        player_id=int(picked["id"]);source=by_id[player_id]
        selected.append({**source,"starter":bool(picked["start"]),"captain":bool(picked["captain"]),"vice_captain":bool(picked["vice_captain"]),"bench_order":None if pd.isna(picked["bench_order"]) else int(picked["bench_order"])})
    starters=[p for p in selected if p["starter"]]
    captain=next(p for p in selected if p["captain"]);vice=next(p for p in selected if p["vice_captain"])
    cost=sum(p["price"] for p in selected)
    formation="-".join(str(sum(1 for p in starters if p["position"]==position)) for position in ("DEF","MID","FWD"))
    xi=sum(p["predicted_points"] for p in starters)+captain["predicted_points"]
    return {"selected":selected,"cost":round(cost,2),"remaining_budget":round(100-cost,2),"projected_xi_points":round(xi,3),"formation":formation,"captain":captain["name"],"vice_captain":vice["name"],"gameweek":predictions.get("gameweek"),"prediction_scope":predictions.get("model_scope"),"data_status":predictions.get("data_status")}


@app.post("/api/optimize-lineup")
def optimize_lineup(req: OptimizeRequest):
    """Optimize user-supplied estimates; never imply they came from the model."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc: raise HTTPException(503,"SciPy optimizer is unavailable.") from exc
    d=pd.DataFrame([p.model_dump() for p in req.players])
    if d.id.duplicated().any(): raise HTTPException(422,"Player ids must be unique.")
    if not all(math.isfinite(float(x)) for x in d.projected_points) or not all(math.isfinite(float(x)) for x in d.price): raise HTTPException(422,"Prices and supplied point estimates must be finite numbers.")
    posmap={"GK":0,"GKP":0,"DEF":1,"MID":2,"FWD":3}
    d["pos"]=d.position.str.upper().map(posmap)
    if d.pos.isna().any(): raise HTTPException(422,"Positions must be GK, DEF, MID, or FWD.")
    n=len(d)
    if n<15: raise HTTPException(422,"At least 15 candidates are required for a full squad.")
    pts=d.projected_points.to_numpy(); price=d.price.to_numpy(); pos=d.pos.to_numpy(); clubs=d.team.to_numpy()
    # Decision variables: 15 squad, 11 starters, captain, vice-captain.
    size=4*n; objective=np.r_[np.zeros(n),-pts,-pts, -pts*1e-5]
    rows=[]; lower=[]; upper=[]
    def add(v,l,h): rows.append(v);lower.append(l);upper.append(h)
    z=np.zeros(n); one=np.ones(n)
    add(np.r_[one,z,z,z],15,15);add(np.r_[price,z,z,z],-np.inf,req.budget)
    for k,count in {0:2,1:5,2:5,3:3}.items(): add(np.r_[(pos==k).astype(float),z,z,z],count,count)
    for club in set(clubs): add(np.r_[(clubs==club).astype(float),z,z,z],-np.inf,3)
    add(np.r_[z,one,z,z],11,11)
    for k,(lo,hi) in {0:(1,1),1:(3,5),2:(2,5),3:(1,3)}.items():add(np.r_[z,(pos==k).astype(float),z,z],lo,hi)
    add(np.r_[z,z,one,z],1,1);add(np.r_[z,z,z,one],1,1)
    eye=np.eye(n); add(np.r_[-eye,eye,np.zeros((n,2*n))].reshape(-1),-np.inf,0) if False else None
    for i in range(n):
        row=np.zeros(size);row[i]=-1;row[n+i]=1;add(row,-np.inf,0)
        row=np.zeros(size);row[2*n+i]=1;row[n+i]=-1;add(row,-np.inf,0)
        row=np.zeros(size);row[3*n+i]=1;row[2*n+i]=1;add(row,-np.inf,1)
        row=np.zeros(size);row[3*n+i]=1;row[n+i]=-1;add(row,-np.inf,0)
    result=milp(objective,integrality=np.ones(size),bounds=Bounds(0,1),constraints=LinearConstraint(np.asarray(rows),lower,upper),options={"time_limit":20})
    if not result.success or result.x is None: raise HTTPException(422,"No feasible squad found for these candidates and budget.")
    v=np.rint(result.x).astype(int);d["selected"]=v[:n].astype(bool);d["starter"]=v[n:2*n].astype(bool);d["captain"]=v[2*n:3*n].astype(bool);d["vice_captain"]=v[3*n:].astype(bool)
    chosen=d[d.selected].copy(); chosen["bench_order"]=None
    bench=chosen[~chosen.starter].copy()
    goalkeeper=bench[bench.position.str.upper()=="GK"]
    outfield=bench[bench.position.str.upper()!="GK"].sort_values("projected_points",ascending=False)
    chosen.loc[goalkeeper.index,"bench_order"]=0
    chosen.loc[outfield.index,"bench_order"]=range(1,len(outfield)+1)
    return {"estimate_source":"Caller-supplied projected_points; not generated by the saved model", "selected":json_records(chosen), "cost":float(chosen.price.sum()), "remaining_budget":round(req.budget-float(chosen.price.sum()),2),"projected_xi_points":float((chosen.loc[chosen.starter,"projected_points"].sum()+chosen.loc[chosen.captain,"projected_points"].sum())),"formation":"-".join(str(int((chosen.loc[chosen.starter,"pos"]==k).sum())) for k in (1,2,3)),"captain":chosen.loc[chosen.captain,"name"].iloc[0],"vice_captain":chosen.loc[chosen.vice_captain,"name"].iloc[0]}

@app.post("/api/transfer-analysis")
def transfer_analysis(req: TransferRequest):
    feasible=[p for p in req.candidates if p.position==req.outgoing.position and p.id!=req.outgoing.id and p.price-req.outgoing.price<=req.budget and p.team!=req.outgoing.team]
    return {"assumption":"Single like-for-like transfer; point values are caller supplied and reflect no future horizon beyond those estimates.","outgoing":req.outgoing.name,"options":[{"player":p.name,"team":p.team,"price":p.price,"projected_points":p.projected_points,"projected_gain":p.projected_points-req.outgoing.projected_points,"budget_change":req.outgoing.price-p.price} for p in sorted(feasible,key=lambda x:x.projected_points-req.outgoing.projected_points,reverse=True)]}

@app.post("/api/captain-recommendation")
def captain_recommendation(players: list[PlayerInput]):
    if len(players)<2: raise HTTPException(422,"Provide at least two players to choose captain and vice-captain.")
    ordered=sorted(players,key=lambda p:p.projected_points,reverse=True)
    return {"estimate_source":"Caller-supplied projected points", "captain":{"name":ordered[0].name,"projected_points":ordered[0].projected_points,"captaincy_total_if_estimate_realized":2*ordered[0].projected_points,"increment_from_captaincy":ordered[0].projected_points},"vice_captain":{"name":ordered[1].name,"projected_points":ordered[1].projected_points},"limitation":"Ranking uses only the supplied estimates; no minutes, fixture or uncertainty adjustment is available."}
