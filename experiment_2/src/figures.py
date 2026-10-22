import sys
from pathlib import Path
import numpy as np, pandas as pd, patsy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0, str(Path(__file__).parent))
from gee_lib import prep, fit
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "E2_analysis_data.csv"); dp = d[d.primary].copy()
S, M, PR = "C(subset, Treatment('MC'))", "C(model, Treatment('luna'))", "C(prompt, Treatment('B'))"
F = f"lc + {S} + {M} + {PR} + lc:{S}"
d1 = prep(dp); res, w, e = fit(F, d1, "step_success"); assert res is not None and res.converged and not w
mean_l2 = dp.log2_length.mean(); cov = res.cov_params().values; DI = patsy.dmatrix(F, d1, return_type='dataframe').design_info
assert list(DI.column_names) == list(res.params.index), (DI.column_names, list(res.params.index))
rhs = F; colors = {"MC": "#4c78a8", "MA": "#e45756"}; ls = {"luna": "-", "mimo": "--", "qwen": ":", "qwen122": "-.", "qwen9b": (0, (3, 1, 1, 1)), "qwen9b_nt": (0, (1, 1))}
MODELS = [m for m in ["luna", "mimo", "qwen", "qwen122", "qwen9b", "qwen9b_nt"] if m in set(dp.model)]
MCOL = {"luna": "#4c78a8", "mimo": "#f58518", "qwen": "#54a24b", "qwen122": "#b279a2", "qwen9b": "#e8a957", "qwen9b_nt": "#9c755f"}
sname = {"MC": "mathchat (reference)", "MA": "magentic-one"}
# ---- Fig 1
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, p in zip(axes, ["B", "C"]):
    for s in ["MC", "MA"]:
        sub = dp[dp.subset == s]; lo, hi = sub.log2_length.min(), sub.log2_length.max()
        xs = np.linspace(lo, hi, 60)
        for m in MODELS:
            nd = pd.DataFrame({"lc": xs - mean_l2, "subset": s, "model": m, "prompt": p, "step_success": 0})
            X = np.asarray(patsy.build_design_matrices([DI], nd, return_type="dataframe")[0])
            eta = X @ res.params.values; se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
            ax.plot(xs, expit(eta), ls[m], color=colors[s], label=f"{sname[s]}, {m}" )
            if m == "luna": ax.fill_between(xs, expit(eta - 1.96 * se), expit(eta + 1.96 * se), color=colors[s], alpha=.15)
        # empirical, quantile-binned (display only), pooled over models
        q = sub[sub.prompt == p].copy(); q["bin"] = pd.qcut(q.log2_length, 6, duplicates="drop")
        g = q.groupby("bin", observed=True).agg(x=("log2_length", "mean"), y=("step_success", "mean"), n=("step_success", "size"))
        ax.scatter(g.x, g.y, s=np.clip(g.n / 8, 12, 80), facecolors="none", edgecolors=colors[s], linewidths=1.2)
    ticks = [1, 2, 4, 8, 16, 32]; ax.set_xticks(np.log2(ticks)); ax.set_xticklabels(ticks); ax.set_xlabel("trajectory length (entries, log2 scale)")
    ax.set_title(f"Prompt {p}"); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("P(exact-step hit)"); axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Primary GEE 1 predicted probability (lines, 95% band shown for luna; delta method, robust cov). Open circles = quantile-binned empirical, display only.\nAssociation, not causal; subsets are different agent systems.", fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "E2_pred_prob_vs_length.png", dpi=140); plt.close(fig)
# ---- Fig 2 signed error (raw and normalized)
mis = dp[dp.late.notna()]
for col, name, lab in (("signed_error", "E2_signed_error_centered.png", "signed error (pred - true step)"), ("signed_error_norm", "E2_signed_error_norm_centered.png", "normalized signed error (/ length)")):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    rng = np.random.default_rng(0)
    for i, s in enumerate(["MC", "MA"]):
        for j, p in enumerate(["B", "C"]):
            ax = axes[i, j]
            for m, c in [(m_, MCOL[m_]) for m_ in MODELS]:
                g = mis[(mis.subset == s) & (mis.prompt == p) & (mis.model == m)]
                if g.empty: continue
                ax.scatter(g.log2_length + rng.normal(0, .02, len(g)), g[col], s=10, alpha=.35, color=c, label=f"{m} (n={len(g)})")
                if len(g) > 25 and g.log2_length.nunique() > 3:
                    lw = lowess(g[col], g.log2_length, frac=.8); ax.plot(lw[:, 0], lw[:, 1], color=c, lw=2)
            ax.axhline(0, color="k", lw=1); m_ = np.nanmax(np.abs(mis[col])); ax.set_ylim(-m_ * 1.05, m_ * 1.05)
            ax.set_title(f"{sname[s]}, prompt {p}"); ax.legend(fontsize=7)
            ticks = [2, 4, 8, 16, 32]; ax.set_xticks(np.log2(ticks)); ax.set_xticklabels(ticks)
            if j == 0: ax.set_ylabel(lab + "\n(negative = early, positive = late)")
    axes[1, 0].set_xlabel("trajectory length (log2 scale)"); axes[1, 1].set_xlabel("trajectory length (log2 scale)")
    fig.suptitle("Parsed misses only; lines = LOWESS per model; conditional on a miss, and limited by where the true step sits", fontsize=9); fig.tight_layout(); fig.savefig(ROOT / name, dpi=140); plt.close(fig)
# ---- Fig 3 breakdown bars
def bars(x, by, fname, title):
    keys = [k for k, g in x.groupby(by) if len(g) >= 20]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, (metric, lab) in zip(ax, (("step_success", "exact-step accuracy %"), ("late", "% late among parsed misses"))):
        for q, p in enumerate(["B", "C"]):
            for i, k in enumerate(keys):
                g = x[(x[by] == k) & (x.prompt == p)]
                g = g if metric == "step_success" else g[g.late.notna()]
                v = g[metric].dropna(); n = len(v)
                if n == 0: continue
                kk = v.sum(); lo, hi = proportion_confint(kk, n, method="wilson")
                a.bar(i + (q - .5) * .38, 100 * kk / n, .36, yerr=[[max(0, 100 * (kk / n - lo))], [max(0, 100 * (hi - kk / n))]], color=["#4c78a8", "#f58518"][q], capsize=2, label=f"Prompt {p}" if i == 0 else None)
                a.text(i + (q - .5) * .38, 100 * hi + 1, f"n={n}", ha="center", fontsize=6, rotation=90)
        a.set_xticks(range(len(keys))); a.set_xticklabels(keys, rotation=30, ha="right", fontsize=8); a.set_ylabel(lab); a.set_ylim(0, 118)
    ax[0].legend(); fig.suptitle(title + " (primary frameworks; Wilson 95% CI; luna and MiMo pooled; keys with <20 rows omitted)", fontsize=9); fig.tight_layout(); fig.savefig(ROOT / fname, dpi=140); plt.close(fig)
bars(dp, "benchmark", "E2_breakdown_task_type.png", "By benchmark (task type nested in framework)")
bars(dp, "error_type_family", "E2_breakdown_error_family.png", "By error-type family")
bars(dp, "error_type_mode", "E2_breakdown_error_mode.png", "By error-type mode")
print("figures done")
