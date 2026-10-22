"""Score raw responses and build the deliverables. Models with both prompts complete (2,670 each) go into the summary;
partial models are listed in the predictions file only. Run:  python src/analyze.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data import load_records
from parser import parse_response
from score import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
N_BOOT, SEED = 10000, 0
MODELS, PROMPTS = ["luna", "mimo", "qwen", "qwen122", "qwen9b", "qwen9b_nt"], ["B", "C"]

recs = {r["id"]: r for r in load_records()}
rows = []
for m in MODELS:
    for p in PROMPTS:
        f = OUT / "raw" / m / f"{p}.jsonl"
        if not f.exists():
            continue
        last = {}
        for l in f.open():
            d = json.loads(l)
            if not d.get("error"):
                last[d["trajectory_id"]] = d          # successful responses only; one per trajectory (exactly-once rule)
        for tid, d in last.items():
            r = recs[tid]; pr = parse_response(d["text"], d["finish_reason"]); sc = score(pr, r)
            rows.append({"model": m, "prompt": p, "trajectory_id": tid, "framework": r["framework"], "benchmark": r["benchmark"],
                         "mode": r["mode"], "n_entries": r["n_entries"], "gt_agents": "|".join(r["gt_agents"]),
                         "gt_steps": "|".join(map(str, r["gt_steps"])), "raw_response_path": str(f.relative_to(ROOT)),
                         "parse_ok": pr["ok"], "failure": pr["failure"], "pred_agent": pr["agent"], "pred_step": pr["step"],
                         **sc, "output_tokens": d.get("output_tokens"), "finish_reason": d["finish_reason"],
                         "truncated": pr["failure"] == "truncated"})
P = pd.DataFrame(rows)
P.to_csv(ROOT / "E1_predictions.csv", index=False)
counts = P.groupby(["model", "prompt"]).size().unstack()
print("rows per (model, prompt):\n", counts)

full = [m for m in MODELS if m in counts.index and all(counts.loc[m].get(p, 0) == len(recs) for p in PROMPTS)]
wil = lambda k, n: tuple(round(100 * x, 1) for x in proportion_confint(k, n, method="wilson"))
fmt = lambda k, n: f"{100*k/n:.1f} [{wil(k, n)[0]}, {wil(k, n)[1]}]"
srows = []
for m in full:
    for p in PROMPTS:
        d = P[(P.model == m) & (P.prompt == p)]; n = len(d)
        srows.append({"Model": m, "Prompt": p, "n": n, "Parse fail %": round(100 * (~d.parse_ok).mean(), 1),
                      "Truncated": int(d.truncated.sum()),
                      "Agent acc %": fmt(d.agent_correct.sum(), n), "Exact-step acc %": fmt(d.step_correct.sum(), n),
                      "Joint acc % (primary)": fmt(d.joint_correct.sum(), n),
                      "Median norm. error": round(d.norm_error.dropna().median(), 3)})
S = pd.DataFrame(srows)

# paired comparison C vs B per model (prompt A is not run, so the primary contrast is C - B); joint is primary
rng = np.random.default_rng(SEED); pair_rows = []
for m in full:
    b = P[(P.model == m) & (P.prompt == "B")].set_index("trajectory_id"); c = P[(P.model == m) & (P.prompt == "C")].set_index("trajectory_id")
    ids = sorted(recs); 
    for metric in ["joint_correct", "step_correct", "agent_correct"]:
        x, y = b.loc[ids, metric].values.astype(int), c.loc[ids, metric].values.astype(int)
        diff = y - x; idx = rng.integers(0, len(ids), (N_BOOT, len(ids)))
        boots = diff[idx].mean(1) * 100
        b01, b10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
        pv = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        pair_rows.append({"Model": m, "Comparison": "C - B", "Metric": metric, "change_pp": round(diff.mean() * 100, 2),
                          "boot95_lo": round(np.percentile(boots, 2.5), 2), "boot95_hi": round(np.percentile(boots, 97.5), 2),
                          "C_only_correct": b01, "B_only_correct": b10, "mcnemar_exact_p": pv,
                          "primary": metric == "joint_correct"})
Q = pd.DataFrame(pair_rows)
prim = Q.primary; pv = Q.loc[prim, "mcnemar_exact_p"].values; order = np.argsort(pv); k = len(pv); holm = np.empty(k); run = 0
for rank, i in enumerate(order):
    run = max(run, (k - rank) * pv[i]); holm[i] = min(1.0, run)
Q["holm_p"] = np.nan; Q.loc[prim, "holm_p"] = holm
Q.to_csv(OUT / "E1_paired.csv", index=False)
S.to_csv(ROOT / "E1_summary.csv", index=False)
print(S.to_string(index=False)); print(Q.to_string(index=False))

# by framework (joint acc), descriptive
fw = P[P.model.isin(full)].groupby(["model", "prompt", "framework"]).agg(n=("joint_correct", "size"), agent=("agent_correct", "mean"),
      step=("step_correct", "mean"), joint=("joint_correct", "mean"), parse_fail=("parse_ok", lambda s: 1 - s.mean())).round(3).reset_index()
fw.to_csv(OUT / "E1_by_framework.csv", index=False); print(fw.to_string(index=False))

# figure: Agent / Step / Joint with 95% Wilson CIs
if full:
    fig, ax = plt.subplots(1, len(full), figsize=(5.2 * len(full), 4), squeeze=False, sharey=True)
    for a, m in zip(ax[0], full):
        for j, (metric, lab) in enumerate([("agent_correct", "Agent"), ("step_correct", "Exact step"), ("joint_correct", "Joint")]):
            for q, p in enumerate(PROMPTS):
                d = P[(P.model == m) & (P.prompt == p)]; k_, n_ = d[metric].sum(), len(d); lo, hi = proportion_confint(k_, n_, method="wilson")
                a.bar(j + (q - .5) * .38, 100 * k_ / n_, .36, yerr=[[100 * (k_ / n_ - lo)], [100 * (hi - k_ / n_)]], color=["#4c78a8", "#f58518"][q], label=f"Prompt {p}" if j == 0 else None, capsize=3)
        a.set_xticks(range(3)); a.set_xticklabels(["Agent", "Exact step", "Joint"]); a.set_title(m); a.set_ylim(0, 100)
    ax[0][0].set_ylabel("accuracy % (95% Wilson CI)"); ax[0][0].legend()
    fig.suptitle("E1 on Who&When Pro multi-agent text traces (n=2,670); parse failures count as incorrect", fontsize=9); fig.tight_layout()
    fig.savefig(OUT / "E1_accuracy.png", dpi=130)
