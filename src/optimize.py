"""Phase 7: ML predictions -> valid FPL squad via Integer Linear Programming (SciPy MILP / HiGHS)."""
import pandas as pd, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from common import *

BUDGET = 100.0
POS = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}


def pick_team(d, col, bench_w=0.1, budget=BUDGET):
    """d: candidate players for ONE gameweek with prediction column `col`. Returns the optimal 15-man squad.
    Variables per player: x (in squad), s (in starting XI), c (captain, points count double)."""
    d = d.reset_index(drop=True); d["p"] = d[col].fillna(0.0); n = len(d)
    p, price, pos = d.p.values, d.price.values, d.pos_code.values
    cost = np.concatenate([-bench_w * p, -p * (1 - bench_w), -p, -p * 1e-5]) # squad, XI, captain, vice
    rows, lo, hi = [], [], []
    def add(vec, l, h): rows.append(vec); lo.append(l); hi.append(h)
    z = np.zeros(n); one = np.ones(n)
    add(np.concatenate([one, z, z, z]), 15, 15)                             # 15-man squad
    add(np.concatenate([price, z, z, z]), -np.inf, budget)                  # budget
    for k, cnt in {0: 2, 1: 5, 2: 5, 3: 3}.items():                         # 2 GK / 5 DEF / 5 MID / 3 FWD
        add(np.concatenate([(pos == k) * 1.0, z, z, z]), cnt, cnt)
    for t in d.team.unique():                                               # max 3 per club
        add(np.concatenate([(d.team.values == t) * 1.0, z, z, z]), -np.inf, 3)
    add(np.concatenate([z, one, z, z]), 11, 11)                             # 11 starters
    for k, (l, h) in {0: (1, 1), 1: (3, 5), 2: (2, 5), 3: (1, 3)}.items():  # legal formations
        add(np.concatenate([z, (pos == k) * 1.0, z, z]), l, h)
    add(np.concatenate([z, z, one, z]), 1, 1)                              # one captain
    add(np.concatenate([z, z, z, one]), 1, 1)                              # one vice
    I = np.eye(n)
    A_sx = np.hstack([-I, I, np.zeros((n, 2*n))])
    A_cs = np.hstack([np.zeros((n,n)), -I, I, np.zeros((n,n))])
    A_vs = np.hstack([np.zeros((n,n)), -I, np.zeros((n,n)), I])
    A_cv = np.hstack([np.zeros((n, 2*n)), I, I])
    A = np.vstack([np.array(rows), A_sx, A_cs, A_vs, A_cv])
    lb = np.concatenate([lo, -np.inf * np.ones(4 * n)]); ub = np.concatenate([hi, np.zeros(3 * n), np.ones(n)])
    res = milp(cost, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(4 * n), bounds=Bounds(0, 1),
               options=dict(time_limit=30))
    if not res.success or res.x is None:
        raise ValueError("No legal 15-player squad could be selected under the supplied budget.")
    v = np.round(res.x).astype(int)
    d["squad"], d["start"], d["captain"], d["vice_captain"] = v[:n], v[n:2*n], v[2*n:3*n], v[3*n:]
    d["bench_order"] = None
    bench = d[(d.squad == 1) & (d.start == 0)]
    keeper = bench[bench.pos_code == 0]
    outfield = bench[bench.pos_code != 0].sort_values("p", ascending=False)
    d.loc[keeper.index, "bench_order"] = 0
    d.loc[outfield.index, "bench_order"] = range(1, len(outfield)+1)
    return d[d.squad == 1]


def realized(sq):  # actual FPL points of the XI (captain doubled; ignores auto-subs)
    return (sq.target * (sq.start + sq.captain)).sum()


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    te = pd.read_csv(OUT / "test_predictions.csv")
    te["pos_code"] = te.pos_code.astype(int)
    strategies = {"Oracle (perfect hindsight)": "target", "XGBoost": "m_XGBoost", "XGBoost (walk-forward)": "m_XGBoost (walk-forward)",
                  "LightGBM": "m_LightGBM", "Rolling-5 avg": "b_roll5", "Previous GW": "b_prev_gw"}
    rows, ex = [], {}
    gws = sorted(te.GW.unique())
    for gw in gws:
        d = te[te.GW == gw]
        for name, col in strategies.items():
            if d[col].notna().sum() < 100: continue
            sq = pick_team(d.copy(), col)
            rows.append(dict(GW=gw, strategy=name, points=realized(sq), pred_total=(sq[col].fillna(0) * (sq.start + sq.captain)).sum()))
            if gw == 20 and name == "XGBoost": ex = sq
        print("GW", gw, "done", flush=True)
    r = pd.DataFrame(rows); r.to_csv(TAB / "optimization_by_gw.csv", index=False)
    summ = r.groupby("strategy").points.agg(["sum", "mean"]).round(1).sort_values("sum", ascending=False); summ.to_csv(TAB / "optimization_summary.csv"); print(summ)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in summ.index:
        s = r[r.strategy == name].sort_values("GW"); ax.plot(s.GW, s.points.cumsum(), label=name, lw=2.5 if name == "XGBoost" else 1.3, ls="--" if "Oracle" in name else "-")
    ax.set_xlabel("Gameweek (2025-26)"); ax.set_ylabel("cumulative points"); ax.set_title("Optimised weekly squads: realised points by prediction source"); ax.legend(); plt.tight_layout(); plt.savefig(FIG / "08_optimization_cumulative.png"); plt.close()
    ex = ex.assign(pos=ex.pos_code.map(POS)).sort_values(["start", "pos_code"], ascending=[False, True])
    ex[["name", "team", "pos", "price", "m_XGBoost", "target", "start", "captain"]].round(2).to_csv(TAB / "example_squad_GW20.csv", index=False)
    print(ex[["name", "team", "pos", "price", "m_XGBoost", "target", "start", "captain"]].round(2).to_string())
