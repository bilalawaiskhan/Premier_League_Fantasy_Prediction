# API

Run `uvicorn app.main:app --app-dir backend --reload` from the repository root. Open `/docs` for generated OpenAPI schemas. `GET /api/players` and performance routes serve the recorded 2025–26 test output. `/api/live/*` and `/api/gameweeks/current` serve official current metadata / fixtures and do not generate predictions. The public entry picks route is best-effort and returns a clear fallback message for private or not-yet-released picks. Prepared inference rejects incompatible feature matrices. Squad endpoints persist user-entered JSON and estimates to the configured SQL database.

The API returns HTTP errors for missing reports, failed upstream requests, malformed input, infeasible optimization and model load/prediction errors. CORS origins are set by `CORS_ORIGINS`.
