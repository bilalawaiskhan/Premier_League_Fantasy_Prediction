"""Proves features for GW t do not change if GW t's own results are corrupted -> no leakage."""
import numpy as np, pandas as pd
from data import load_raw, to_player_gw, SUM_COLS
from features import build_dataset, feature_columns
pgw = to_player_gw(load_raw()); base = build_dataset(pgw); FC = feature_columns(base)
S, G = "2025-26", 20
bad = pgw.copy(); m = (bad.season == S) & (bad.GW == G)
for c in SUM_COLS: bad.loc[m, c] = 999.0           # destroy everything that "happens" in GW20
new = build_dataset(bad)
key = ["name", "season", "GW"]
a = base.set_index(key).loc[lambda x: (x.index.get_level_values("season") == S) & (x.index.get_level_values("GW") == G), FC].sort_index()
b = new.set_index(key).loc[lambda x: (x.index.get_level_values("season") == S) & (x.index.get_level_values("GW") == G), FC].sort_index()
same = np.allclose(a.fillna(-1).values, b.fillna(-1).values)
a2 = base.set_index(key).loc[lambda x: (x.index.get_level_values("season") == S) & (x.index.get_level_values("GW") == G + 1), FC].sort_index()
b2 = new.set_index(key).loc[lambda x: (x.index.get_level_values("season") == S) & (x.index.get_level_values("GW") == G + 1), FC].sort_index()
diff = not np.allclose(a2.fillna(-1).values, b2.fillna(-1).values)
print(f"GW{G} features unchanged after corrupting GW{G} results: {same}")
print(f"GW{G+1} features DO change (sanity check that test is sensitive): {diff}")
assert same and diff, "LEAKAGE DETECTED"
print("LEAKAGE TEST PASSED")
