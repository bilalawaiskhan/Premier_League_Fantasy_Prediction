# Development

Install backend and frontend dependencies as shown in the root README. The API uses `FPL_PROJECT_ROOT` to find the original `data`, `src`, and `outputs` folders; by default it derives the project root from its source path. SQLite is created under `outputs/fpl_app.sqlite` locally. Use `DATABASE_URL` for PostgreSQL.

Keep the 73-feature order stored in the joblib artifact. Do not edit or regenerate existing model outputs as part of application development. Add compatibility tests before allowing any live source to flow into inference. Keep historical test values, new historical inference, actual points, and live current data explicitly labeled.
