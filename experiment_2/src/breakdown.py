import sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "E2_analysis_data.csv")
wil = lambda k, n: ("", "") if n == 0 else tuple(round(100 * x, 1) for x in proportion_confint(k, n, method="wilson"))


def table(x, by, label):
    rows = []
    for key, g in x.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        if n < 5: continue                                              # suppressed (<5 rows)
        mi = g[g.late.notna()]; k = int(g.step_success.sum()); kl = int(mi.late.sum()); nm = len(mi)
        lo, hi = wil(k, n); llo, lhi = wil(kl, nm)
        rows.append({"stratum": label, **dict(zip(by, key)), "n_traces": g.trajectory_id.nunique(), "n_rows": n,
                     "step_acc_%": round(100 * k / n, 1), "acc_CI_lo": lo, "acc_CI_hi": hi, "parsed_misses": nm,
                     "late_%_of_misses": round(100 * kl / nm, 1) if nm else np.nan, "late_CI_lo": llo, "late_CI_hi": lhi,
                     "median_signed_error": mi.signed_error.median() if nm else np.nan, "median_signed_error_norm": round(mi.signed_error_norm.median(), 3) if nm else np.nan,
                     "median_length": g.length.median(), "flag": "n < 10, descriptive only" if n < 10 else ""})
    return pd.DataFrame(rows)


def build(x, tag):
    x = x.copy()
    x["length_bin_display_only"] = pd.cut(x.length, [0, 4, 8, 1000], labels=["short (<=4)", "middle (5-8)", "long (>8)"])
    out = {"model": [], "prompt": [], "task_type": [], "error_type": []}
    out["model"] = [table(x, ["model", "prompt", "subset"], "model x prompt x subset"), table(x, ["model", "prompt"], "model x prompt"), table(x, ["model", "subset"], "model x subset"), table(x, ["model"], "model")]
    out["prompt"] = [table(x, ["prompt", "subset"], "prompt x subset"), table(x, ["prompt"], "prompt"), table(x, ["subset"], "subset"), table(x, ["prompt", "length_bin_display_only"], "length bin (display only) x prompt"),
                     table(x, ["subset", "length_bin_display_only"], "length bin (display only) x subset")]
    out["task_type"] = [table(x, ["task_type"], "task_type"), table(x, ["task_type", "prompt"], "task_type x prompt"), table(x, ["benchmark"], "benchmark"), table(x, ["benchmark", "prompt"], "benchmark x prompt"), table(x, ["subset"], "subset (level not available)")]
    modes = x.error_type_mode.value_counts(); keep = modes[x.drop_duplicates("trajectory_id").error_type_mode.value_counts().reindex(modes.index).fillna(0) >= 15].index
    out["error_type"] = [table(x, ["error_type_family"], "error family"), table(x, ["error_type_family", "prompt"], "family x prompt"), table(x, ["error_type_family", "subset"], "family x subset"),
                         table(x[x.error_type_mode.isin(keep)], ["error_type_mode"], "mode (traces >= 15)"), table(x[x.error_type_mode.isin(keep)], ["error_type_mode", "prompt"], "mode x prompt")]
    for k, v in out.items():
        df = pd.concat(v, ignore_index=True); df.insert(0, "set", tag); out[k] = df
    return out

res = {}
for tag, x in (("primary (MC+MA)", d[d.primary]), ("all six frameworks", d)):
    for k, v in build(x, tag).items(): res.setdefault(k, []).append(v)
for k, v in res.items():
    pd.concat(v, ignore_index=True).to_csv(ROOT / f"E2_breakdown_by_{k}.csv", index=False)
    print(k, sum(len(t) for t in v), "rows")
x = pd.read_csv(ROOT / "E2_breakdown_by_model.csv"); pd.set_option("display.width", 250)
print(x[(x.set == "primary (MC+MA)") & (x.stratum == "model x prompt x subset")][["model", "prompt", "subset", "n_traces", "n_rows", "step_acc_%", "parsed_misses", "late_%_of_misses", "median_signed_error", "median_length"]].to_string(index=False))
