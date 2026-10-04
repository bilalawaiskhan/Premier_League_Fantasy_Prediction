#!/usr/bin/env bash
# Reproduce everything (about 10-15 min on 1 CPU; Random Forest is the slow step)
set -e
cd "$(dirname "$0")/src"
python leakage_test.py   # must print LEAKAGE TEST PASSED
python eda.py
python train.py          # writes outputs/tables, figures, test_predictions.csv, xgb_model.joblib
python optimize.py       # needs test_predictions.csv from train.py
