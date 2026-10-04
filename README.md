# FPL AI — Fantasy Premier League analytics

This application adds a local web interface and API around the existing FPL research project. It preserves and reuses the existing historical datasets, saved XGBoost artifact, evaluation reports, historical test predictions, feature engineering, and SciPy MILP work. The application does not retrain the model.

## What works

- Searchable historical 2025–26 held-out test player rows and recorded XGBoost estimates.
- Existing model comparison and feature-importance reports.
- Optional inference using the saved model on explicitly prepared, feature-compatible rows.
- Historical replay using the original feature engineering code, with actual points shown separately.
- Current official FPL bootstrap metadata and fixture retrieval (network required).
- Best-effort public Team ID picks retrieval; private/unreleased picks show an error and manual/JSON squad input remains available.
- A generated 73-feature compatibility inventory documenting candidate live sources, transformations, missing-data policy and leakage checks.
- SciPy 15-player squad / starting XI optimizer for caller-supplied projected points, plus single-transfer comparisons for caller-supplied estimates.
- Manual squad persistence using SQLite by default or PostgreSQL through `DATABASE_URL`.
- Responsive Next.js interface with player explorer, fixtures, historical model report, and manual optimizer input.

## Important scope and limitations

The supplied model expects 73 historical features. The repository does not provide a proven adapter from 2026–27 FPL data to all of those features. Consequently current player metadata and fixtures are live, while current model projections are explicitly unavailable. Historical estimates in the explorer refer to the last recorded held-out season (2025–26), not the current season. The optimizer only optimizes the projected points sent by the caller and never invents them. Transfer analysis is one-for-one and only covers the supplied points horizon.

No FPL password, account token, or private team endpoint is used. Squad persistence is local and has no authentication; keep it on a trusted local network unless authentication is added before public deployment. Database tables are created on startup; formal versioned migrations have not been added.

## Existing research and artifact provenance

- `data/`: season files 2022–23 through 2025–26; retained in place.
- `src/features.py`: original 73-feature, past-only feature engineering; retained in place.
- `src/`: original ingestion, EDA, training, leakage check, and optimizer code; retained in place.
- `outputs/xgb_model.joblib`: original trained XGBoost artifact; loaded as-is, never overwritten. XGBoost 3.1.2 was verified to deserialize it and reproduce the saved test predictions within floating-point tolerance; that runtime is pinned.
- `outputs/test_predictions.csv`, `outputs/tables/model_comparison.csv`, `outputs/tables/xgb_feature_importance.csv`: existing outputs surfaced without being regenerated.

Validation history in the existing README describes train seasons 2022–23/2023–24, validation season 2024–25, and test season 2025–26. Metrics in the UI are the reports already committed in `outputs/tables/model_comparison.csv`; the app does not recalculate them.

## Run locally

### Backend

From the repository root, create a virtual environment and install:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

On Linux/macOS use `python3 -m venv .venv`, activate `.venv/bin/activate`, then run the same `pip` and `uvicorn` commands. The API documentation is at `http://localhost:8000/docs`; health is `/api/health`.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` at build time if the backend is hosted somewhere else. Configure backend `CORS_ORIGINS` to the frontend origin.

### Docker Compose

From the repository root, run `docker compose up --build`. This starts PostgreSQL, FastAPI, and Next.js. The default database password is for local development only; set `POSTGRES_PASSWORD` in the environment for any shared deployment.

## API overview

- `GET /api/health`
- `GET /api/players` (historical test rows)
- `GET /api/players/{name}/history`
- `GET /api/model-performance`
- `GET /api/feature-importance`
- `GET /api/model-compatibility` (exact 73-feature compatibility inventory)
- `GET /api/gameweeks/current`, `GET /api/live/bootstrap`, `GET /api/live/players`, and `GET /api/live/fixtures` (live FPL public endpoints)
- `GET /api/live/entry/{team_id}/event/{gameweek}/picks` (public picks where available; no login/password)
- `POST /api/predict/prepared` (requires the exact model feature columns)
- `GET /api/predictions/historical-inference`
- `POST /api/optimize-lineup` and `POST /api/transfer-analysis`
- `GET/POST /api/squads`, `PUT/DELETE /api/squads/{id}`

### Live team picker

The frontend now starts at a guided team builder. Build a new squad or import a public FPL Team ID; picks are saved in this browser. Player estimates are served by `GET /api/live/predictions`, and `POST /api/live/auto-pick` uses the repository optimizer against that table. `GET /api/live/events/{gameweek}/` and `GET /api/live/fixtures` power the live scores view. All official FPL requests run through the backend, with a User-Agent, five-minute bootstrap/fixture cache, one-minute live cache, and a persistent SQLite cache for outage fallback.

For the live estimates, the original research artifact `outputs/xgb_model.joblib` remains untouched. The separately validated `outputs/xgb_live_model.joblib` uses 70 API-reproducible features. Official API team-strength values are zero and do not match the vaastav training scale, so the three strength columns are omitted. See `docs/live_model_validation.json` for the chronological validation and test metrics. Regular-starter test R² is about 0.11; estimates are guidance, not guarantees.

Run the backend from the repository root with `uvicorn app.main:app --app-dir backend --reload` after installing `backend/requirements.txt`. Run the UI in another terminal with `cd frontend; npm run dev`. No API key is needed. The browser-only localStorage squad is not synced between devices.

Full request schemas are available from FastAPI's OpenAPI page.

## ML methodology, in brief

Each row represents a player/Gameweek. `src/features.py` shifts player history before rolling calculations so the target Gameweek's results cannot enter its own inputs. The outcome is target Gameweek FPL points. Validation follows season chronology rather than a random split. MAE is typical absolute error in points; RMSE penalizes large misses more; R² compares with a mean baseline; Spearman measures player ranking; top-10 actual points summarizes outcomes among the model's selected top ten. Feature importance describes model usage patterns, not causal effects.

The XGBoost artifact contains a model plus its ordered feature list. `/api/predict/prepared` validates the columns and numeric values and passes them in saved order. It cannot make historical raw data ready by itself. `/api/predictions/historical-inference` rebuilds historical features with the original source code and returns newly generated inference beside actual points and the recorded test estimate. With XGBoost 3.1.2, a 500-row check reproduced recorded estimates with maximum absolute difference below 0.000001 points.

## Deployment and security

The Dockerfiles provide a starting point for container hosting. A hosted deployment must package the historical data and model artifact where the backend can read them. No deployment was performed or verified here. Add authentication, HTTPS, production CORS restrictions, secret rotation, and migration tooling before exposing squad storage publicly. Do not commit `.env`, database passwords, or FPL credentials.

## Further documentation

See `docs/architecture.md`, `docs/ml-methodology.md`, `docs/data-pipeline.md`, `docs/optimization.md`, `docs/api.md`, `docs/development.md`, `docs/deployment.md`, and `docs/testing.md`.

For hosting steps, see [docs/hosting.md](docs/hosting.md).
