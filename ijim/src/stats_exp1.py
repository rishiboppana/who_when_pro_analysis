"""Experiment 1 statistics (plan 7.3). Primary endpoint: exact joint accuracy. Within each dataset and judge, never pooled.
Primary: O1 vs O0. Secondary: O1-A vs O1, O2 vs O1. Paired cluster bootstrap (B=10,000, percentile, seed 20261022),
exact McNemar (two-sided binomial on discordant pairs), Holm over the comparisons in the dataset-judge family.
python src/stats_exp1.py"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
import load_data as ld

SEED, B = 20261022, 10_000
COMPARISONS = [("O1", "O0"), ("O1-A", "O1"), ("O2", "O1")]      # (treatment, reference); the first is the primary one
ORDER = ["O0", "O1", "O1-A", "O2"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(df, trt, ref, col):
    """Aligned boolean/float arrays over trajectories present in both conditions. -> (merged frame, a=ref, b=trt)."""
    x = df[df.condition == ref].set_index("traj_id"); y = df[df.condition == trt].set_index("traj_id")
    ids = x.index.intersection(y.index)
    return x.loc[ids], y.loc[ids]


def boot_delta(ref, trt, col, seed=SEED, b=B):
    d = trt[col].astype(float).values - ref[col].astype(float).values
    cl = ref["cluster_id"].values
    codes, uniq = pd.factorize(cl)
    s = np.bincount(codes, weights=d, minlength=len(uniq)); n = np.bincount(codes, minlength=len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), (b, len(uniq)))
    return np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])


def boot_median_err(ref, trt, seed=SEED, b=B):
    ok = ref.norm_err.notna() & trt.norm_err.notna()
    if ok.sum() == 0:
        return float("nan"), (float("nan"),) * 2
    r, t = ref[ok], trt[ok]
    d = t.norm_err.values - r.norm_err.values
    codes, uniq = pd.factorize(r["cluster_id"].values)
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = [np.median(d[np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])]) for _ in range(b)]
    return float(np.median(d)), tuple(np.percentile(vals, [2.5, 97.5]))


def results_table(df):
    rows = []
    for (d, j, c), g in df.groupby(["dataset", "judge", "condition"]):
        n = len(g); pf = int(g.parse_fail.sum()); par = g[~g.parse_fail]
        r = {"Dataset": d, "Judge": j, "Trace condition": c, "n": n, "n_parseable": len(par)}
        for name, col in (("Agent", "agent_ok"), ("Step", "step_ok"), ("Joint", "joint_ok")):
            k = int(g[col].sum()); lo, hi = wilson(k, n)
            r |= {f"{name} %": 100 * k / n, f"{name} Wilson lo": 100 * lo, f"{name} Wilson hi": 100 * hi}
        r |= {"Median normalized error": par.norm_err.median() if len(par) else float("nan"), "parse_fail %": 100 * pf / n}
        rows.append(r)
    t = pd.DataFrame(rows)
    t["_o"] = t["Trace condition"].map(ORDER.index)
    return t.sort_values(["Dataset", "Judge", "_o"]).drop(columns="_o")


def effects(df):
    prim, sec = [], []
    for (d, j), g in df.groupby(["dataset", "judge"]):
        fam = []
        for trt, ref in COMPARISONS:
            if trt not in set(g.condition) or ref not in set(g.condition):
                continue
            R, T = paired(g, trt, ref, "joint_ok")
            bb = int(((~R.joint_ok.astype(bool)) & T.joint_ok.astype(bool)).sum())     # treatment right, reference wrong
            cc = int((R.joint_ok.astype(bool) & (~T.joint_ok.astype(bool))).sum())
            p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if bb + cc else 1.0
            lo, hi = boot_delta(R, T, "joint_ok")
            fam.append({"Dataset": d, "Judge": j, "Comparison": f"{trt} vs {ref}", "n": len(R), "b": bb, "c": cc,
                        "Delta joint (pp)": 100 * (T.joint_ok.astype(float).mean() - R.joint_ok.astype(float).mean()),
                        "CI lo (pp)": 100 * lo, "CI hi (pp)": 100 * hi, "raw p": p})
            for col, name in (("agent_ok", "agent"), ("step_ok", "step"), ("parse_fail", "parse_fail")):
                lo2, hi2 = boot_delta(R, T, col)
                sec.append({"Dataset": d, "Judge": j, "Comparison": f"{trt} vs {ref}", "Metric": name, "n": len(R),
                            "Delta (pp)": 100 * (T[col].astype(float).mean() - R[col].astype(float).mean()), "CI lo (pp)": 100 * lo2, "CI hi (pp)": 100 * hi2})
            med, (l3, h3) = boot_median_err(R, T)
            sec.append({"Dataset": d, "Judge": j, "Comparison": f"{trt} vs {ref}", "Metric": "median paired change in normalized error (parseable in both)",
                        "n": int((R.norm_err.notna() & T.norm_err.notna()).sum()), "Delta (pp)": med, "CI lo (pp)": l3, "CI hi (pp)": h3})
        for row, h in zip(fam, holm([r["raw p"] for r in fam])):
            row["Holm p"] = h
        prim += fam
    return pd.DataFrame(prim), pd.DataFrame(sec)


def forest(eff, path):
    """Plot-only styling (dataviz palette: blue slot 1, recessive grid, ink tokens). Does not touch any statistic."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    e = eff[eff.Comparison == "O1 vs O0"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.8 * len(e)), facecolor=SURF)
    ax.set_facecolor(SURF)
    for i, r in e.iterrows():
        ax.hlines(i, r["CI lo (pp)"], r["CI hi (pp)"], color=BLUE, lw=2)
        ax.plot(r["Delta joint (pp)"], i, "o", color=BLUE, ms=9, mec=SURF, mew=2)
        ax.text(1.02, i, f"{r['Delta joint (pp)']:+.1f} pp   n={r['n']}   Holm p={r['Holm p']:.3g}", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(e))); ax.set_yticklabels([f"{r.Dataset}  ({r.Judge})" for r in e.itertuples()], color=INK)
    ax.set_xlabel("O1 minus O0: change in exact joint accuracy (percentage points; 95% paired cluster-bootstrap CI)", color=INK2, fontsize=9)
    ax.set_title("Adding explicit step labels: effect on exact joint WHO+WHEN accuracy", color=INK, fontsize=11, loc="left")
    ax.invert_yaxis(); ax.grid(axis="x", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def main():
    out = ld.ROOT / "outputs" / "exp1"
    df = pd.read_csv(out / "X1_predictions.csv")
    for c in ("agent_ok", "step_ok", "joint_ok", "parse_fail"):
        df[c] = df[c].astype(bool)
    results_table(df).to_csv(out / "X1_results.csv", index=False)
    prim, sec = effects(df)
    prim[prim.Comparison == "O1 vs O0"].to_csv(out / "X1_effects.csv", index=False)
    prim[prim.Comparison != "O1 vs O0"].to_csv(out / "X1_effects_secondary_joint.csv", index=False)
    sec.to_csv(out / "X1_effects_secondary.csv", index=False)
    forest(prim, out / "X1_forest.png")
    print("done")


if __name__ == "__main__":
    main()
