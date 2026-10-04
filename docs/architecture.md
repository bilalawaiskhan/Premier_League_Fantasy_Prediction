# Architecture

The existing research source, historical CSV data, and artifacts remain where they are. The FastAPI backend reads those files directly, exposes historical research and calls the same original feature builder for a historical inference replay. Live FPL access is a separate asynchronous HTTP client. The frontend calls the backend through one configurable base URL. SQLite is the no-service local persistence default; SQLAlchemy can use PostgreSQL via `DATABASE_URL`.

Current static player metadata / fixture retrieval and model inference are separate paths. There is no current-season inference path because the current API has not been shown to supply all 73 compatible features. All such projections are marked unavailable rather than imputed silently.
