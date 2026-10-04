# Data pipeline

Historical source files are per-fixture player records. `src/data.py` maps opponents, derives opponent strengths, and groups double fixtures into player/Gameweek rows. `src/features.py` builds 73 model inputs, using shifted historical rows for form. Existing CSVs are read in place; the application does not rewrite them.

Live routes retrieve `/api/bootstrap-static/` and `/api/fixtures/` from the public FPL service with bounded timeouts. API records are returned as live metadata only. A feature compatibility audit is still required before producing current projections: player identities, prior season/team context, fixture/team features, price history and missing values need a supported mapping to the model's exact trained columns. No default-zero imputation is claimed to solve this.

`model_feature_compatibility.csv` was generated from the actual `feature_columns()` output over the supplied historical data. It lists the exact 73 feature names and a cautious candidate source / compatibility status for each. “Unverified” means no current live inference should be shown for that model feature set yet. Regenerate with `python scripts/generate_feature_compatibility.py` from the project root after changing the original feature builder.
