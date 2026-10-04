"""Small SQLAlchemy persistence layer for manually managed squads."""
import os
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

PROJECT = Path(os.getenv("FPL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
default_db = f"sqlite:///{(PROJECT / 'outputs' / 'fpl_app.sqlite').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", default_db)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}, pool_pre_ping=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase): pass

class Squad(Base):
    __tablename__ = "user_squads"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), default="My squad")
    players_json: Mapped[str] = mapped_column(String, default="[]")
    budget: Mapped[float] = mapped_column(Float, default=100.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

def init_db():
    (PROJECT / "outputs").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)

def list_squads():
    import json
    with Session() as db:
        return [{"id":s.id,"name":s.name,"players":json.loads(s.players_json),"budget":s.budget,"created_at":s.created_at.isoformat()} for s in db.scalars(select(Squad).order_by(Squad.id.desc())).all()]

def save_squad(name: str, players: list[dict], budget: float, squad_id: int | None = None):
    import json
    with Session() as db:
        row=db.get(Squad,squad_id) if squad_id is not None else None
        if squad_id is not None and row is None: return None
        if row is None: row=Squad(); db.add(row)
        row.name=name; row.players_json=json.dumps(players,allow_nan=False); row.budget=budget
        db.commit(); db.refresh(row)
        return {"id":row.id,"name":row.name,"players":players,"budget":row.budget,"created_at":row.created_at.isoformat()}

def delete_squad(squad_id: int) -> bool:
    with Session() as db:
        row=db.get(Squad,squad_id)
        if row is None: return False
        db.delete(row); db.commit(); return True
