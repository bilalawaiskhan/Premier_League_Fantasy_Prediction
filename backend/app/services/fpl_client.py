"""Bounded, cached client for official public FPL endpoints."""
import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import httpx

BASE="https://fantasy.premierleague.com/api"
_cache: dict[str,tuple[float,Any,datetime]]={}
_state: dict[str,dict[str,Any]]={}
_locks: dict[str,asyncio.Lock]={}
_clients: dict[int,httpx.AsyncClient]={}
_cache_db=Path(os.getenv("FPL_API_CACHE_PATH",str(Path(__file__).resolve().parents[3]/"outputs"/"fpl_api_cache.sqlite")))
_disk_loaded=False

def _load_disk_cache() -> None:
    global _disk_loaded
    if _disk_loaded:return
    _disk_loaded=True
    if not _cache_db.is_file():return
    try:
        with sqlite3.connect(_cache_db,timeout=10) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS fpl_cache (path TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at REAL NOT NULL, ttl INTEGER NOT NULL)")
            rows=connection.execute("SELECT path,payload,fetched_at,ttl FROM fpl_cache").fetchall()
        now_wall=time.time();now_mono=time.monotonic()
        for path,payload,fetched_at,ttl in rows:
            data=json.loads(payload);stamp=datetime.fromtimestamp(fetched_at,timezone.utc)
            _cache[path]=(now_mono+max(0,int(ttl)-(now_wall-fetched_at)),data,stamp)
    except (sqlite3.Error,ValueError,OSError):
        return

def _save_disk_cache(path: str,data: Any,fetched_at: datetime,ttl: int) -> None:
    _cache_db.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(data,separators=(",",":"),ensure_ascii=False)
    with sqlite3.connect(_cache_db,timeout=20) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS fpl_cache (path TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at REAL NOT NULL, ttl INTEGER NOT NULL)")
        connection.execute("INSERT OR REPLACE INTO fpl_cache(path,payload,fetched_at,ttl) VALUES(?,?,?,?)",
                           (path,payload,fetched_at.timestamp(),int(ttl)))

def _client() -> httpx.AsyncClient:
    loop_id=id(asyncio.get_running_loop())
    client=_clients.get(loop_id)
    if client is None or client.is_closed:
        client=httpx.AsyncClient(timeout=12,headers={"User-Agent":"FPL-AI-Fantasy-Analytics/1.0"},
                                 limits=httpx.Limits(max_connections=12,max_keepalive_connections=8))
        _clients[loop_id]=client
    return client

async def get_json(path: str, ttl: int) -> Any:
    """GET and cache an endpoint with a short retry budget and finite timeout."""
    _load_disk_cache()
    now=time.monotonic(); cached=_cache.get(path)
    if cached and cached[0]>now:
        _state[path]={"stale":False,"last_success_at":cached[2].isoformat()}
        return cached[1]
    lock=_locks.setdefault(path,asyncio.Lock())
    async with lock:
        now=time.monotonic(); cached=_cache.get(path)
        if cached and cached[0]>now:
            _state[path]={"stale":False,"last_success_at":cached[2].isoformat()}
            return cached[1]
        last: Exception|None=None
        for attempt in range(3):
            try:
                response=await _client().get(f"{BASE}/{path.lstrip('/')}")
                response.raise_for_status(); data=response.json()
                fetched=datetime.now(timezone.utc)
                _cache[path]=(time.monotonic()+ttl,data,fetched)
                _state[path]={"stale":False,"last_success_at":fetched.isoformat()}
                try:
                    await asyncio.to_thread(_save_disk_cache,path,data,fetched,ttl)
                except (sqlite3.Error,OSError,TypeError):
                    pass
                return data
            except (httpx.HTTPError,ValueError) as exc:
                last=exc
                if attempt<2: await asyncio.sleep(.3*(2**attempt))
        if cached:
            _state[path]={"stale":True,"last_success_at":cached[2].isoformat(),"error":type(last).__name__ if last else "unknown"}
            return cached[1]
        _state[path]={"stale":True,"last_success_at":None,"error":type(last).__name__ if last else "unknown"}
        raise RuntimeError(f"FPL endpoint failed after bounded retries: {path}") from last


def cache_state(*paths: str) -> dict[str,Any]:
    """Report freshness for upstream resources used by a response."""
    states=[_state.get(path,{"stale":False,"last_success_at":None}) for path in paths]
    return {"stale":any(row.get("stale",False) for row in states),
            "last_success_at":max((row["last_success_at"] for row in states if row.get("last_success_at")),default=None)}

async def close_clients() -> None:
    loop_id=id(asyncio.get_running_loop())
    client=_clients.pop(loop_id,None)
    if client is not None:
        await client.aclose()
