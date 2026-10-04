"""Fit an API-compatible, chronologically evaluated reduced XGBoost model.

The original 73-feature artifact is never modified. Three unavailable team
strength inputs are omitted because official API values are not equivalent.
"""
import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from common import TRAIN_S, VAL_S, TEST_S, get_dataset, eligible
from features import feature_columns

OMIT={"opp_att_str","opp_def_str","opp_ovr_str"}
OUT=ROOT/"outputs"

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);residual=y-p
    return {"rows":int(len(y)),"MAE":float(np.abs(residual).mean()),"RMSE":float(np.mean(residual**2)**.5),"R2":float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),"Spearman":float(spearmanr(p,y).statistic)}

def main():
    ds=get_dataset(); old=joblib.load(OUT/"xgb_model.joblib")
    all_features=feature_columns(ds); features=[c for c in all_features if c not in OMIT]
    assert len(features)==70 and features==[c for c in old["features"] if c not in OMIT]
    ds=eligible(ds); tr=ds[ds.season.isin(TRAIN_S)]; va=ds[ds.season.isin(VAL_S)]; te=ds[ds.season.isin(TEST_S)]
    cfg=json.loads((OUT/"run_config.json").read_text()); params={**cfg["xgb_params"],"n_estimators":cfg["n_estimators"],"tree_method":"hist","n_jobs":-1,"random_state":0}
    validation=xgb.XGBRegressor(**params).fit(tr[features],tr.target); va_pred=validation.predict(va[features])
    trva=pd.concat([tr,va],ignore_index=True); model=xgb.XGBRegressor(**params).fit(trva[features],trva.target); test_pred=model.predict(te[features])
    regular=te.minutes_last_5>=30
    recorded=pd.read_csv(OUT/"test_predictions.csv"); old_rows=te[["name","season","GW"]].merge(recorded[["name","season","GW","m_XGBoost"]],on=["name","season","GW"],how="left")
    report={"feature_count":len(features),"omitted_features":sorted(OMIT),"feature_order":features,
      "validation_2024_25":metrics(va.target.to_numpy(),va_pred),"test_2025_26_all":metrics(te.target.to_numpy(),test_pred),
      "test_2025_26_regular_30min":metrics(te.loc[regular,"target"].to_numpy(),test_pred[regular.to_numpy()]),
      "original_73_feature_model_test_2025_26_all":metrics(te.target.to_numpy(),old_rows.m_XGBoost.to_numpy()),
      "live_test_minus_original_mae":float(np.abs(te.target.to_numpy()-test_pred).mean()-np.abs(te.target.to_numpy()-old_rows.m_XGBoost.to_numpy()).mean()),
      "method":"Same past-only feature engineering and chronological train/validation/test split. Existing XGBoost settings and estimator count; no target-Gameweek outcomes in inputs."}
    joblib.dump({"model":model,"features":features,"omitted_features":sorted(OMIT),"training":"2022-23 + 2023-24","validation":"2024-25","test":"2025-26"},OUT/"xgb_live_model.joblib")
    (ROOT/"docs"/"live_model_validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if isinstance(v,dict) or k=="live_test_minus_original_mae"},indent=2))

if __name__=="__main__": main()
