# ML methodology

The target is the points earned by one player in a target Gameweek. Prior player and team histories are shifted before rolling and exponential averages are calculated. That shift is the core leakage safeguard: a row for GW t is built from earlier weeks only. The original leakage script corrupts GW20 results and verifies its GW20 features remain unchanged.

The existing experiment trains on 2022–23 and 2023–24, validates/tunes on 2024–25, and holds 2025–26 out for testing. XGBoost is saved in `outputs/xgb_model.joblib` together with the ordered feature names. XGBoost 3.1.2 successfully deserialized it and matched the existing recorded estimates within floating-point tolerance on a 500-row replay; this version is pinned. This application loads but does not retrain or modify that artifact. The provided metrics and feature importance are surfaced from existing CSV reports.

MAE reports average absolute error in point units; RMSE weights large misses more; R² compares squared error with a constant-mean baseline; Spearman rank correlation assesses ordering; top-ten actual points averages realized points among the predicted top ten each Gameweek. Importance is a model-use statistic and cannot establish that a feature caused points.

At inference time the API rejects missing columns and nonnumeric/missing values. The artifact does not make arbitrary live API records compatible: the caller must supply the exact features. Historical replay runs the saved builder over past seasons and is a new inference result, distinct from the original held-out prediction file.
