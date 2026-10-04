import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np, json, time
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb, lightgbm as lgb
from common import *
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 130

ds = get_dataset(); FC = feature_columns(ds); ds = eligible(ds)
assert ds.pos_code.notna().all()
tr, va, te = ds[ds.season.isin(TRAIN_S)], ds[ds.season.isin(VAL_S)], ds[ds.season.isin(TEST_S)]
print("rows  train/val/test:", len(tr), len(va), len(te), "| features:", len(FC))

# ---------- baselines (no learning) ----------
def add_baselines(d):
    d = d.copy(); P = d[[f"p_lag{k}" for k in range(1, 6)]]
    d["b_mean"] = tr.target.mean()
    d["b_prev_gw"] = d.p_lag1
    d["b_roll5"] = P.mean(axis=1)
    w = np.array([.40, .25, .15, .10, .10]); m = P.notna().values
    d["b_weighted5"] = (P.fillna(0).values * w * m).sum(1) / (w * m).sum(1)
    d["b_fpl_xP"] = d.xP                                 # FPL's own official expectation (external benchmark)
    return d
tr, va, te = add_baselines(tr), add_baselines(va), add_baselines(te)

# ---------- ML models ----------
def models(n_est=None):
    return {
        "Ridge (linear)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(150, min_samples_leaf=30, max_features=.4, n_jobs=-1, random_state=0)),
        "MLP (neural net)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=200, random_state=0)),
    }
Xtr, ytr, Xva, yva, Xte = tr[FC], tr.target, va[FC], va.target, te[FC]

# XGBoost: small grid tuned on the VALIDATION season only
grid = [dict(max_depth=d, learning_rate=lr, min_child_weight=mcw, subsample=.8, colsample_bytree=.7, reg_lambda=5)
        for d in (3, 4, 6) for lr in (.03,) for mcw in (10, 30)]
best = None
for p in grid:
    m = xgb.XGBRegressor(n_estimators=1500, early_stopping_rounds=60, tree_method="hist", n_jobs=-1, random_state=0, **p).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    mae = mean_absolute_error(yva, m.predict(Xva))
    if best is None or m.best_score < best[0]: best = (m.best_score, p, m.best_iteration + 1)
_, XP, XN = best; print("XGB best", XP, "n_est", XN)
lp = dict(num_leaves=15, learning_rate=.03, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_lambda=5, n_estimators=XN, verbose=-1, random_state=0)
def mk_xgb(): return xgb.XGBRegressor(n_estimators=XN, tree_method="hist", n_jobs=-1, random_state=0, **XP)
def mk_lgb(): return lgb.LGBMRegressor(**lp)

allm = {**models(), "XGBoost": mk_xgb(), "LightGBM": mk_lgb()}
# Stage A: fit on TRAIN only -> score on VAL (model selection).  Stage B: refit on TRAIN+VAL -> score on untouched TEST.
trva = pd.concat([tr, va])
for name, m in allm.items():
    t0 = time.time(); m.fit(Xtr, ytr); va["m_" + name] = m.predict(Xva)
    m2 = {**models(), "XGBoost": mk_xgb(), "LightGBM": mk_lgb()}[name]; m2.fit(trva[FC], trva.target); te["m_" + name] = m2.predict(Xte)
    allm[name] = m2; print(f"{name:18s} done {time.time()-t0:.0f}s")

# ---------- walk-forward (expanding window) XGBoost on the test season, retrain every 4 GWs ----------
te["m_XGBoost (walk-forward)"] = np.nan
for g0 in range(1, 39, 4):
    chunk = te.GW.between(g0, g0 + 3)
    hist = pd.concat([trva, te[te.GW < g0]])
    m = mk_xgb().fit(hist[FC], hist.target); te.loc[chunk, "m_XGBoost (walk-forward)"] = m.predict(te.loc[chunk, FC])

# ---------- evaluation ----------
def metrics(d, col):
    y, p = d.target, d[col]; ok = p.notna(); y, p = y[ok], p[ok]
    g = d[ok].assign(_p=p).groupby("GW")
    pg = g.apply(lambda x: spearmanr(x._p, x.target)[0] if x._p.nunique() > 1 else np.nan).mean()
    top10 = g.apply(lambda x: x.nlargest(10, "_p").target.mean()).mean()
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** .5, R2=r2_score(y, p),
                Spearman=spearmanr(p, y)[0], Spearman_per_GW=pg, Top10_actual_pts=top10)
def table(d, label):
    cols = [c for c in d.columns if c.startswith(("b_", "m_"))]
    rows = {c[2:]: metrics(d, c) for c in cols}
    t = pd.DataFrame(rows).T.round(3); t.index.name = label; return t
res = {}
for lab, d in [("VAL 2024-25 | all eligible", va), ("TEST 2025-26 | all eligible", te), ("TEST 2025-26 | regular players (>=30 avg min)", regulars(te))]:
    res[lab] = table(d, lab); print("\n" + lab + "\n", res[lab].sort_values("MAE"))
pd.concat(res).to_csv(TAB / "model_comparison.csv")
for i, (k, v) in enumerate(res.items()): v.to_csv(TAB / f"model_comparison_{i}.csv")

# ---------- figures ----------
t = res["TEST 2025-26 | regular players (>=30 avg min)"].sort_values("MAE")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
colors = ["#bbb" if n.startswith(("mean", "prev", "roll", "weighted", "fpl")) else "#2a7" for n in t.index]
ax[0].barh(t.index[::-1], t.MAE[::-1], color=colors[::-1]); ax[0].set_title("Test MAE, regular players (lower = better)\ngrey = baselines, green = ML"); ax[0].set_xlabel("MAE")
t2 = t.sort_values("Spearman_per_GW"); ax[1].barh(t2.index, t2.Spearman_per_GW, color=["#bbb" if n.startswith(("mean", "prev", "roll", "weighted", "fpl")) else "#2a7" for n in t2.index])
ax[1].set_title("Mean per-GW Spearman rank corr. (higher = better ranking)"); plt.tight_layout(); plt.savefig(FIG / "05_model_comparison.png"); plt.close()

imp = pd.Series(allm["XGBoost"].feature_importances_, FC).sort_values().tail(20)
fig, ax = plt.subplots(figsize=(8, 6)); imp.plot.barh(ax=ax, color="#2a7"); ax.set_title("XGBoost feature importance (top 20, trained on train+val)"); plt.tight_layout(); plt.savefig(FIG / "06_feature_importance.png"); plt.close()
imp.sort_values(ascending=False).to_csv(TAB / "xgb_feature_importance.csv", header=["importance"])

r = regulars(te); fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(r["m_XGBoost"], r.target, s=3, alpha=.25); ax[0].plot([0, 9], [0, 9], "r--"); ax[0].set_xlabel("predicted"); ax[0].set_ylabel("actual"); ax[0].set_title("XGBoost predicted vs actual (test, regulars)")
r.assign(b=pd.qcut(r["m_XGBoost"], 10, duplicates="drop")).groupby("b", observed=True).agg(p=("m_XGBoost", "mean"), a=("target", "mean")).plot(x="p", y="a", marker="o", ax=ax[1], legend=False)
ax[1].plot([0, 7], [0, 7], "r--"); ax[1].set_title("Calibration by prediction decile"); ax[1].set_xlabel("mean predicted"); ax[1].set_ylabel("mean actual"); plt.tight_layout(); plt.savefig(FIG / "07_pred_vs_actual.png"); plt.close()

# by position
rows = []
for pos, nm in enumerate(["GK", "DEF", "MID", "FWD"]):
    s = r[r.pos_code == pos]; rows.append(dict(position=nm, n=len(s), MAE_xgb=mean_absolute_error(s.target, s.m_XGBoost), MAE_roll5=mean_absolute_error(s.target, s.b_roll5.fillna(0)), Spearman_xgb=spearmanr(s.m_XGBoost, s.target)[0]))
pd.DataFrame(rows).round(3).to_csv(TAB / "by_position.csv", index=False); print(pd.DataFrame(rows).round(3))

keep = ["name", "season", "GW", "team", "position", "pos_code", "price", "target", "minutes_last_5", "b_prev_gw", "b_roll5", "b_weighted5", "b_fpl_xP", "m_XGBoost", "m_XGBoost (walk-forward)", "m_LightGBM", "m_Random Forest", "m_Ridge (linear)"]
te[keep].to_csv(OUT / "test_predictions.csv", index=False)
import joblib; joblib.dump(dict(model=allm["XGBoost"], features=FC), OUT / "xgb_model.joblib")
json.dump(dict(xgb_params=XP, n_estimators=XN, n_features=len(FC), rows=dict(train=len(tr), val=len(va), test=len(te))), open(OUT / "run_config.json", "w"), indent=1)
