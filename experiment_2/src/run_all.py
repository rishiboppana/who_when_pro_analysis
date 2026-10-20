import sys, warnings, hashlib
from pathlib import Path
import numpy as np, pandas as pd, patsy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(Path(__file__).parent))
from gee_lib import *
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT
d = pd.read_csv(ROOT / "E2_analysis_data.csv"); dp = d[d.primary].copy()
S, M, PR = "C(subset, Treatment('MC'))", "C(model, Treatment('luna'))", "C(prompt, Treatment('B'))"
F = f"lc + {S} + {M} + {PR} + lc:{S}"; INTER = f"lc:{S}[T.MA]"
txt, effects, coefs, errors = [], [], [], []


def record(tag, res, w, e, outcome, kind, inter=INTER, rows=None, extra=None):
    txt.append(f"\n===== {tag} | outcome={outcome} | converged={None if res is None else res.converged} | warnings={w} | error={e}")
    if res is None or not res.converged or w:
        errors.append(f"{tag} | {outcome} | converged={None if res is None else res.converged} | warnings={w} | error={e}"); return None
    txt.append(str(res.summary()))
    c = res.summary().tables[1]; cc = pd.DataFrame({"term": res.params.index, "coef": res.params.values, "robust_se": res.bse.values, "p": res.pvalues.values}); cc.insert(0, "model_tag", tag); cc.insert(1, "outcome", outcome); coefs.append(cc)
    for r in (rows if rows is not None else slopes(res, inter)):
        effects.append({"Outcome": outcome, "Subset": r["Effect"].split(", ")[-1] if "doubling" in r["Effect"] else "difference", "Effect": r["Effect"], "OR": r["OR"], "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"],
                        "p_raw": r["p_raw"], "p_holm": np.nan, "Kind": kind, "Model": tag, "n_traces": res.model.data.frame.trajectory_id.nunique(), "n_rows": int(res.nobs)})
    return res


def run(tag, data, outcome, formula=F, kind="sensitivity", cov="robust", length_col="log2_length", inter=INTER, rows=None):
    dd = prep(data, length_col)
    res, w, e = fit(formula, dd, outcome, cov_type=cov)
    return record(tag, res, w, e, outcome, kind, inter=inter, rows=rows), dd

mis = lambda x: x[x.late.notna()]
# ---- primary
r1, d1 = run("PRIMARY GEE1", dp, "step_success", kind="primary")
r2, d2 = run("PRIMARY GEE2", mis(dp), "late", kind="primary")
if r1 is None or r2 is None:
    (ROOT / "E2_GEE_errors.txt").write_text("\n".join(errors)); print("PRIMARY FAILED, stopped; see E2_GEE_errors.txt"); sys.exit(1)
# Holm over the 4 primary slope p-values
prim = [i for i, r in enumerate(effects) if r["Kind"] == "primary" and "doubling" in r["Effect"]]
adj = multipletests([effects[i]["p_raw"] for i in prim], method="holm")[1]
for i, a in zip(prim, adj): effects[i]["p_holm"] = a
# ---- sensitivity
run("S1 exclude parse failures", dp[dp.parse_ok == 1], "step_success")
run("S2 bias-reduced cov", dp, "step_success", cov="bias_reduced"); run("S2 bias-reduced cov", mis(dp), "late", cov="bias_reduced")
Fp = F + " + true_pos_rel"
run("S3 + true_pos_rel", dp, "step_success", formula=Fp); run("S3 + true_pos_rel", mis(dp), "late", formula=Fp)
run("S4 token length (log2 tokens)", dp, "step_success", length_col="log2_token_length"); run("S4 token length (log2 tokens)", mis(dp), "late", length_col="log2_token_length")
CELLS = set(zip(dp.model, dp.prompt))   # every model-prompt cell that exists in the data (luna, mimo and, once run, qwen)
full4 = dp.groupby("trajectory_id").filter(lambda g: set(zip(g.model, g.prompt)) == CELLS)
run("S5 balanced: only traces with all model-prompt cells", full4, "step_success"); run("S5 balanced: only traces with all model-prompt cells", mis(full4), "late")
Fall = f"lc + C(framework) + {M} + {PR}"
for outc, dd0 in (("step_success", d), ("late", mis(d))):
    rr, ddd = run("S6 all six frameworks, framework fixed effects (common slope)", dd0, outc, formula=Fall, rows=None, inter="lc")
# S6 uses slopes() with inter='lc' -> only the common slope is meaningful: keep that row
effects = [r for r in effects if not (r["Model"].startswith("S6") and r["Effect"] != "doubling of length, MC (reference)")]
for r in effects:
    if r["Model"].startswith("S6") and r["Effect"].startswith("doubling"): r["Effect"] = "doubling of length (common slope, all 6 frameworks)"; r["Subset"] = "all"
# ---- exploratory 6c interaction tests
def wald(tag, extra_terms, data, outcome):
    dd = prep(data); fm = F + " + " + " + ".join(extra_terms)
    res, w, e = fit(fm, dd, outcome)
    if res is None or not res.converged or w:
        errors.append(f"{tag} {outcome}: conv={None if res is None else res.converged} w={w} e={e}"); return None
    names = [n for n in res.params.index if any(t.split(":")[1].split("[")[0] in n for t in extra_terms) and "lc:" in n and "subset" not in n]
    idx = list(res.params.index); R = np.zeros((len(names), len(idx)))
    for i, n in enumerate(names): R[i, idx.index(n)] = 1
    wt = res.wald_test(R, scalar=True)
    return {"Exploratory test": tag, "outcome": outcome, "terms": "; ".join(names), "chi2": float(wt.statistic), "df": len(names), "p_raw": float(wt.pvalue), "note": "exploratory, uncorrected"}
expl = []
for outc, dd0 in (("step_success", dp), ("late", mis(dp))):
    for tag, terms in (("length x prompt", [f"lc:{PR}"]), ("length x model", [f"lc:{M}"])):
        r = wald(tag, terms, dd0, outc); expl.append(r) if r else None
# ---- exploratory 6b stratified fits
def strat_fit(label, data, outcome):
    ok, ev, ne = stratum_ok(data, outcome)
    base = {"stratum": label, "outcome": outcome, "n_traces": data.trajectory_id.nunique(), "n_rows": len(data), "events": ev, "non_events": ne, "note": "exploratory, uncorrected"}
    if not ok or data.length.nunique() < 3: return base | {"status": "not estimable (needs >=30 traces, >=10 events and non-events, >=3 distinct lengths)"}
    terms = ["lc"] + [t for t, col in ((S, "subset"), (M, "model"), (PR, "prompt")) if data[col].nunique() > 1]
    if data.subset.nunique() > 1: terms.append(f"lc:{S}")
    dd = prep(data); res, w, e = fit(" + ".join(terms), dd, outcome)
    if res is None or not res.converged or w: return base | {"status": f"not estimable: {w or e or 'did not converge'}"}
    c = contrast(res, {"lc": 1}, "doubling OR")
    if not np.isfinite(c["CI_hi"]) or c["OR"] > 1e6 or c["OR"] < 1e-6:
        return base | {"status": f"not estimable: unstable estimate (OR={c['OR']:.3g}, CI upper={c['CI_hi']:.3g}), likely quasi-separation; not altered"}
    if data.subset.nunique() > 1:
        c2 = contrast(res, {"lc": 1, INTER: 1}, "MA")
        return base | {"status": "ok", "OR_doubling_MC": c["OR"], "CI_lo_MC": c["CI_lo"], "CI_hi_MC": c["CI_hi"], "p_raw_MC": c["p_raw"],
                       "OR_doubling_MA": c2["OR"], "CI_lo_MA": c2["CI_lo"], "CI_hi_MA": c2["CI_hi"], "p_raw_MA": c2["p_raw"]}
    return base | {"status": "ok", "OR_doubling": c["OR"], "CI_lo": c["CI_lo"], "CI_hi": c["CI_hi"], "p_raw": c["p_raw"]}
strata = {}
for m in dp.model.unique(): strata[f"model={m}"] = dp[dp.model == m]
for p in dp.prompt.unique(): strata[f"prompt={p}"] = dp[dp.prompt == p]
for m in dp.model.unique():
    for p in dp.prompt.unique(): strata[f"model={m} x prompt={p}"] = dp[(dp.model == m) & (dp.prompt == p)]
for s in ["MC", "MA"]: strata[f"subset/task_type={s} ({dp[dp.subset == s].task_type.iloc[0]})"] = dp[dp.subset == s]
for b in sorted(dp.benchmark.unique()): strata[f"benchmark={b}"] = dp[dp.benchmark == b]
for f_ in sorted(dp.error_type_family.unique()): strata[f"error_family={f_}"] = dp[dp.error_type_family == f_]
for k, v in strata.items():
    expl.append(strat_fit(k, v, "step_success"))
    expl.append(strat_fit(k, mis(v), "late"))
pd.DataFrame(expl).to_csv(ROOT / "E2_exploratory_fits.csv", index=False)

# ---- outputs: GEE text/csv, effects table
(ROOT / "E2_GEE_output.txt").write_text("\n".join(txt)); pd.concat(coefs).to_csv(ROOT / "E2_GEE_output.csv", index=False)
(ROOT / "E2_GEE_errors.txt").write_text("\n".join(errors) if errors else "No non-converged fits and no fit warnings in any model run.\n")
E_ = pd.DataFrame(effects)
def interp(r):
    if "interaction" in r.Effect: return f"Slopes differ: OR {r.OR:.2f} (MA vs MC)"
    pct = (r.OR - 1) * 100
    what = "exact-step hit" if r.Outcome == "step_success" else "a miss landing late (downstream) rather than early"
    return f"A doubling of trajectory length is associated with {abs(pct):.0f}% {'higher' if pct > 0 else 'lower'} odds of {what}"
E_["Interpretation"] = E_.apply(interp, axis=1); E_["95% CI"] = E_.apply(lambda r: f"[{r.CI_lo:.2f}, {r.CI_hi:.2f}]", axis=1)
E_ = E_[["Outcome", "Kind", "Model", "Subset", "Effect", "OR", "95% CI", "p_raw", "p_holm", "n_traces", "n_rows", "Interpretation"]]
E_.rename(columns={"OR": "Odds ratio", "p_raw": "p-value (raw)", "p_holm": "p-value (Holm, primary only)"}).to_csv(ROOT / "E2_effects_table.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70)
print(E_.drop(columns=["Interpretation"]).round(4).to_string(index=False)); print("\nexploratory interaction tests:"); print(pd.DataFrame([e for e in expl if "Exploratory test" in e]).round(4).to_string(index=False)); print("\nerrors:", errors)
