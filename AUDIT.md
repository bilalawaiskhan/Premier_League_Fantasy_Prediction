# FPL team-picking app audit

Audit date: 4 October 2026. Scope: existing FastAPI/Next.js app, original training feature pipeline and saved artifact. No model was retrained during this audit.

## What exists

- FastAPI exposes historical prediction/report routes, current FPL bootstrap/player/fixture routes, public team picks, a caller-input MILP optimizer, and database-backed saved squads.
- Next.js presents a research dashboard, historic/live player tables, fixtures, and historical metrics. Its My Team and Transfers paths require JSON supplied by the user.
- `src/features.py` constructs 73 ordered features from past player Gameweek statistics and team/opponent rolling form. It shifts history before calculating rolling values to avoid target-Gameweek leakage. `src/optimize.py` implements legal 15-player squad selection using the SciPy MILP solver.
- `outputs/xgb_model.joblib` contains the fitted XGBoost estimator and its ordered feature list. Existing historical CSVs, test predictions, reports and feature importance are present.
- The backend FPL client already proxies official public endpoints with a User-Agent and in-memory TTL cache, but cache expiration discards old values; outages currently become errors instead of stale responses.

## What is broken or missing for an ordinary FPL user

- There is no guided 15-slot builder, live-prediction pool, validation UX, captain/vice-captain controls, or one-click auto-pick. The optimizer only accepts caller-entered JSON.
- No live score page or fixture-event refresh exists. The frontend currently identifies itself as a historical research edition.
- Selected-player rationales, evidence-linked drivers, transfer-hit comparison and browser-local persistence are absent.
- The current theme is green/white, not the requested dark purple/cyan visual direction; navigation is not a functional mobile-first picker.
- Live projections were deliberately disabled. A schema match alone cannot establish feature meaning: the model expects opponent-strength columns from season team tables and aggregates double fixtures using vaastav-specific rules. The FPL API has related but not necessarily equivalent strength and event fields. Those mappings need retrospective parity checks; absent proof, current XGBoost values must not be shown as validated forecasts.
- No E2E browser tests or desktop/mobile screenshots currently demonstrate the requested flows.

## Planned changes

1. Add a backend proxy/cache that retains successful responses after expiry, event live-score routes, and a live prediction builder. Test the API-to-model transformations against the original feature engineering and refuse model forecasts if parity is not established.
2. Replace JSON-first squad flow with a localStorage-backed guided pitch/picker, legal squad validation, auto-pick using the existing MILP, public Team ID import, and transfer suggestions.
3. Attach numeric XGBoost contribution explanations to forecasts/selections; only render a reason when the referenced feature value and contribution are both available.
4. Add current Gameweek fixtures, live event/player points, user-squad totals and polling with pause/stale-data behavior.
5. Apply a responsive purple/cyan design system, accessible status/controls, clear loading/error states, and verify desktop/mobile layouts and build/test results.

## Assumptions and limits

- “Next Gameweek” means the event marked `is_next`, or the current event while it is active if no next event is marked.
- Public entry picks can be unavailable for private teams or before picks are public; manual slot-by-slot building remains the fallback.
- A 2025–26 retrospective replay is required before calling live model features equivalent. If that cannot be achieved from the source fields, the live model forecast must remain disabled rather than be presented as reliable. The builder can still work from transparent, explicitly labeled fallback estimates until that decision is resolved.
- API responses are cached in process memory; multi-worker deployments would need shared caching to guarantee a single upstream request across workers.
