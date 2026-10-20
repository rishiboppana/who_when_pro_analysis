import sys; sys.path.insert(0, f"{E1}/src")
import glob, statistics as st, pandas as pd
from parser import parse_response
def summarize(model="qwen", stage="smoke"):
    """Latest SUCCESSFUL response per trajectory (same rule as the runner and the scoring). Old error rows are counted separately."""
    rows, errs = [], {}
    for f in sorted(glob.glob(f"{E1}/outputs/{stage}/{model}/*.jsonl")):
        last = {}
        for l in open(f):
            d = json.loads(l)
            if d.get("error"):
                errs[d["prompt_id"]] = errs.get(d["prompt_id"], 0) + 1; continue
            last[d["trajectory_id"]] = d
        for d in last.values():
            pr = parse_response(d.get("text"), d.get("finish_reason"))
            rows.append({"prompt": d["prompt_id"], "ok": pr["ok"], "failure": pr["failure"], "finish": d["finish_reason"], "in": d.get("input_tokens"), "out": d.get("output_tokens"), "lat": d["latency_s"]})
    return pd.DataFrame(rows), errs
S, ERRS = summarize()
print("old error rows (NOT counted):", ERRS)
print(S.groupby("prompt").agg(n=("ok", "size"), parse_failures=("ok", lambda s: int((~s).sum())), truncated=("finish", lambda s: int((s == "length").sum())),
                              out_mean=("out", "mean"), out_max=("out", "max"), lat_mean=("lat", "mean"), lat_max=("lat", "max")).round(1))
print("failure reasons:", S[~S.ok].groupby(["prompt", "failure"]).size().to_dict())
bad = S.groupby("prompt").ok.apply(lambda s: int((~s).sum()))
print("SMOKE", "PASS" if bad.max() <= 1 else "FAIL: stop and review", "| parse failures per prompt (truncation counts as a failure):", bad.to_dict(), "| rule: at most 1 in 12")
calls, conc = 2 * 2670, PRESET["seqs"]
print(f"Rough projection for 5,340 calls at concurrency {conc} (latency-based; smoke is a small batch, the real rate is usually better): ~{S.lat.mean() * calls / conc / 3600:.1f} h")
