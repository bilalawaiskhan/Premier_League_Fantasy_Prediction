import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from common import *
sns.set_theme(style="whitegrid", palette="deep"); plt.rcParams["figure.dpi"] = 130

ds = get_dataset(); raw = to_player_gw(load_raw())
play = raw[raw.minutes > 0]
summary = {
    "player-gameweek rows": len(raw), "unique players": raw.name.nunique(), "seasons": raw.season.nunique(),
    "GWs per season": 38, "missing values (all raw cols)": int(raw.isna().sum().sum()),
    "mean points (all rows)": raw.total_points.mean(), "median (all rows)": raw.total_points.median(),
    "mean points (played >0 min)": play.total_points.mean(), "median (played)": play.total_points.median(),
    "max points": raw.total_points.max(), "skewness (played)": play.total_points.skew(),
    "share of rows with 0 minutes": (raw.minutes == 0).mean(), "share of DGW rows": (raw.n_fixtures > 1).mean(),
}
pd.Series(summary).to_csv(TAB / "eda_summary.csv", header=["value"])
print(pd.Series(summary).round(3))
print(raw.position.value_counts())

# 1 distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(raw.total_points.clip(-3, 20), bins=24, ax=ax[0]); ax[0].set_title("FPL points, all player-GWs (skewed, mostly 0-2)")
sns.histplot(play.total_points.clip(-3, 20), bins=24, ax=ax[1], color="C1"); ax[1].set_title("Only players who played (>0 min)")
for a in ax: a.set_xlabel("points")
plt.tight_layout(); plt.savefig(FIG / "01_points_distribution.png"); plt.close()

# 2-5 relationship plots (binned means to expose the shape)
def binned(ax, x, title, xl, q=20, data=play):
    d = data.assign(b=pd.qcut(data[x], q, duplicates="drop")); m = d.groupby("b", observed=True).agg(x=(x, "mean"), y=("total_points", "mean"))
    ax.plot(m.x, m.y, "o-"); ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("mean points")
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5)); ax = ax.ravel()
binned(ax[0], "expected_goals", "xG vs points (same GW)", "xG")
binned(ax[1], "expected_assists", "xA vs points (same GW)", "xA")
binned(ax[2], "minutes", "Minutes vs points", "minutes", q=15)
binned(ax[3], "value", "Price (x0.1m) vs points", "price")
sns.boxplot(data=play, x="position", y="total_points", order=["GK", "DEF", "MID", "FWD"], ax=ax[4], showfliers=False); ax[4].set_title("Position vs points")
r = ds[ds.n_prev_apps >= 5]; r = r[r.minutes_last_5 >= 30]
binned(ax[5], "expected_goal_involvements_last_5", "PAST 5-GW xGI vs NEXT-GW points", "rolling xGI (before GW)", data=r)
plt.tight_layout(); plt.savefig(FIG / "02_relationships.png"); plt.close()

# 3 rolling form example
star = play[play.season == "2025-26"].groupby("name").total_points.sum().idxmax()
e = raw[(raw.name == star) & (raw.season == "2025-26")].sort_values("GW")
e["roll5"] = e.total_points.rolling(5).mean()
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.bar(e.GW, e.total_points, alpha=.5, label="points"); ax.plot(e.GW, e.roll5, "r-o", label="5-GW rolling avg")
ax.set_title(f"Rolling form example: {star} (2025-26)"); ax.set_xlabel("GW"); ax.legend(); plt.tight_layout(); plt.savefig(FIG / "03_rolling_form.png"); plt.close()

# correlation of past features with next-GW target (regulars only)
fc = feature_columns(ds); c = r[fc + ["target"]].corr(method="spearman")["target"].drop("target").sort_values(ascending=False)
c.head(15).to_csv(TAB / "top_feature_correlations.csv", header=["spearman"])
fig, ax = plt.subplots(figsize=(8, 5)); c.head(15)[::-1].plot.barh(ax=ax); ax.set_title("Top-15 features by Spearman corr. with next-GW points")
plt.tight_layout(); plt.savefig(FIG / "04_feature_correlations.png"); plt.close()
print(c.head(10).round(3))
# same-GW xG correlation vs past xG correlation (shows why leakage looks 'amazing')
print("same-GW xG corr:", round(play[["expected_goals", "total_points"]].corr(method="spearman").iloc[0, 1], 3),
      "| past-5 xG corr with next GW:", round(r[["expected_goals_last_5", "target"]].corr(method="spearman").iloc[0, 1], 3))
