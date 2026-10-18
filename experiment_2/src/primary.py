import sys, io, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from gee_lib import *
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "E2_analysis_data.csv")
dp = d[d.primary].copy()
F = "lc + C(subset, Treatment('MC')) + C(model, Treatment('luna')) + C(prompt, Treatment('B')) + lc:C(subset, Treatment('MC'))"
INTER = "lc:C(subset, Treatment('MC'))[T.MA]"
out, errs = [], []
d1 = prep(dp)
r1, w1, e1 = fit(F, d1, "step_success")
d2 = prep(dp[dp.late.notna()]); r2, w2, e2 = fit(F, d2, "late")
for name, r, w, e in (("GEE1 step_success", r1, w1, e1), ("GEE2 late", r2, w2, e2)):
    print(name, "converged:", None if r is None else r.converged, "| warnings:", w, "| error:", e)
    if r is not None: print(r.summary().tables[1])
    if e or w or (r is not None and not r.converged): errs.append((name, w, e, None if r is None else r.converged))
if errs: print("ERRORS", errs)
