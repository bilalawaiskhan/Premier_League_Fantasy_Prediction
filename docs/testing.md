# Testing and verification report

Verification performed 4 October 2026 on the supplied project data and model.

## Passed

- Backend `unittest`: 11 tests passed. Covers health, saved historical player rows and metrics, exact live feature mapping/order, measured-value explanations, expired-cache outage fallback, saved-model inference, historical replay equality, source optimizer squad/formation/budget/club constraints, infeasible inputs, transfer comparison, and captain doubling.
- Original `src/leakage_test.py`: passed. Corrupting GW20 outcomes did not change GW20 features; GW21 features changed as the sensitivity check expects.
- Saved artifact inference: loaded `outputs/xgb_model.joblib` with XGBoost 3.1.2, the pinned runtime. A 500-row replay had maximum absolute difference of approximately 0.00000016 points compared with the recorded XGBoost predictions.
- FastAPI startup and local SQLite squad persistence: health returned 200; POST, GET, and DELETE squad operations succeeded in an isolated local database.
- Official FPL current-data smoke checks: live prediction route returned 667 players for GW6, `stale=false`, with numeric model-contribution explanations; request completed in about 26 seconds after the cache was populated. The auto-pick route returned 15 players, 2 GK / 5 DEF / 5 MID / 3 FWD, 11 starters, a 3-4-3 formation, one captain and vice-captain, maximum three per club, and £22.0m remaining.
- Frontend `npm run lint` (TypeScript check) and optimized production build passed on Next.js 16.3.8.
- `npm audit`: zero reported vulnerabilities after updating Next.js.

## Not verified / limitations

- Public Team ID picks vary by visibility and deadline. A direct check against Team ID 1 did not return public picks, so a successful import from a real public team remains unverified.
- Browser visual verification and the requested desktop/mobile screenshots could not be completed: the app browser denied permission to open localhost. Do not treat responsive visual acceptance as verified. The browser journey and successful public-Team-ID import also remain unverified.
- Live feature parity is established for the 70 retained columns with ordered-feature tests. The 3 unavailable opponent-strength columns are excluded from the separately trained reduced artifact; evaluation reports a 0.008-point increase in all-row MAE over the original 73-feature model.
- Docker was unavailable in the environment, so images / Compose could not be run. Hosted deployment, public authentication, database migrations, and a full browser-driven user journey were not verified. Squad persistence is intended for local use until authentication is added.
- XGBoost warns that the artifact uses older pickle serialization. With the pinned version, replay agreement was measured and the preserved artifact was not changed.
