"""Token and latency analysis from the raw logs (successful rows only, same rows as the scoring). Run: python src/tokens_latency.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
pricing = yaml.safe_load((ROOT / "config/pricing.yaml").read_text())
S = pd.read_csv(ROOT / "E1_predictions.csv")[["model", "prompt", "trajectory_id", "framework", "n_entries", "agent_correct", "step_correct", "joint_correct", "parse_ok"]]
rows = []
for m in ["luna", "mimo", "gemma"]:
    for p in ["B", "C"]:
        f = OUT / "raw" / m / f"{p}.jsonl"
        if not f.exists(): continue
        last = {}
        for l in f.open():
            d = json.loads(l)
            if not d.get("error"): last[d["trajectory_id"]] = d
        for tid, d in last.items():
            rows.append({"model": m, "prompt": p, "trajectory_id": tid, "input_tokens": d.get("input_tokens"), "output_tokens": d.get("output_tokens"),
                         "reasoning_tokens": d.get("reasoning_tokens"), "latency_s": d["latency_s"], "transport_retries": d.get("transport_retries", 0)})
T = pd.DataFrame(rows).merge(S, on=["model", "prompt", "trajectory_id"])
T["answer_tokens"] = T.output_tokens - T.reasoning_tokens.fillna(0)
T.to_csv(OUT / "E1_tokens_latency_rows.csv", index=False)
q = lambda s, x: s.quantile(x)
agg = T.groupby(["model", "prompt"]).agg(n=("latency_s", "size"), in_mean=("input_tokens", "mean"), in_median=("input_tokens", "median"), in_max=("input_tokens", "max"),
    out_mean=("output_tokens", "mean"), out_median=("output_tokens", "median"), out_p95=("output_tokens", lambda s: q(s, .95)), out_max=("output_tokens", "max"),
    reason_mean=("reasoning_tokens", "mean"), lat_mean=("latency_s", "mean"), lat_median=("latency_s", "median"), lat_p95=("latency_s", lambda s: q(s, .95)), lat_max=("latency_s", "max"),
    in_total=("input_tokens", "sum"), out_total=("output_tokens", "sum")).round(1)
agg["cost_usd"] = [round((r.in_total * pricing[m]["input"] + r.out_total * pricing[m]["output"]) / 1e6, 2) if pricing[m]["input"] is not None else np.nan for (m, p), r in agg.iterrows()]
agg["out_per_in"] = (agg.out_total / agg.in_total).round(2)
agg.to_csv(OUT / "E1_tokens_latency_summary.csv"); print(agg.T.to_string())
fw = T.groupby(["model", "prompt", "framework"]).agg(n=("latency_s", "size"), in_mean=("input_tokens", "mean"), out_mean=("output_tokens", "mean"), lat_median=("latency_s", "median")).round(0)
fw.to_csv(OUT / "E1_tokens_latency_by_framework.csv"); print(fw.to_string())
# output tokens / latency for correct vs incorrect (joint), and by input-length bin
T["outcome"] = np.where(T.joint_correct, "joint correct", "joint wrong")
oc = T.groupby(["model", "prompt", "outcome"]).agg(n=("latency_s", "size"), out_mean=("output_tokens", "mean"), lat_median=("latency_s", "median")).round(1); print(oc.to_string())
oc.to_csv(OUT / "E1_tokens_latency_by_outcome.csv")
T["in_bin"] = pd.cut(T.input_tokens, [0, 2000, 5000, 10000, 1e9], labels=["<2k", "2-5k", "5-10k", ">10k"])
ib = T.groupby(["model", "prompt", "in_bin"], observed=True).agg(n=("latency_s", "size"), out_mean=("output_tokens", "mean"), lat_median=("latency_s", "median"), joint_acc=("joint_correct", "mean")).round(2)
ib.to_csv(OUT / "E1_tokens_latency_by_input_length.csv"); print(ib.to_string())
print("rho(input tokens, latency):", {f"{m}/{p}": round(g.input_tokens.corr(g.latency_s, method="spearman"), 2) for (m, p), g in T.groupby(["model", "prompt"])})
print("rho(output tokens, latency):", {f"{m}/{p}": round(g.output_tokens.corr(g.latency_s, method="spearman"), 2) for (m, p), g in T.groupby(["model", "prompt"])})
print("rows with transport retries>0:", T.groupby(["model", "prompt"]).transport_retries.apply(lambda s: int((s > 0).sum())).to_dict())
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for (m, p), g in T.groupby(["model", "prompt"]):
    ax[0].hist(np.log10(g.output_tokens.clip(lower=1)), bins=40, histtype="step", label=f"{m}/{p}", density=True)
    ax[1].hist(np.log10(g.latency_s.clip(lower=0.1)), bins=40, histtype="step", label=f"{m}/{p}", density=True)
    ax[2].scatter(g.input_tokens, g.output_tokens, s=2, alpha=.3, label=f"{m}/{p}")
ax[0].set_xlabel("log10 output tokens (incl. reasoning)"); ax[1].set_xlabel("log10 latency (s)"); ax[2].set_xlabel("input tokens"); ax[2].set_ylabel("output tokens"); ax[2].set_yscale("log")
ax[0].legend(fontsize=8); fig.suptitle("E1 token and latency distributions (latency includes provider queueing and concurrency effects)", fontsize=9); fig.tight_layout()
fig.savefig(OUT / "E1_tokens_latency.png", dpi=130)
