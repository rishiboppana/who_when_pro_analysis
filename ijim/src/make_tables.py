"""raw/calls.jsonl -> outputs/exp1/X1_predictions.csv and outputs/exp2/X2_runs.csv (parse + score; no repair, no retries).
Only completed (non-error) calls are scored. python src/make_tables.py"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import load_data as ld
from parse import parse_response
from score import score_run

ROOT = ld.ROOT
COLS = ["dataset", "judge", "condition", "replicate", "traj_id", "cluster_id", "raw_response", "parsed_agent", "parsed_step", "true_agent", "true_step",
        "agent_ok", "step_ok", "joint_ok", "parse_fail", "parse_failure_reason", "norm_err", "L", "tokens_in", "tokens_out", "tokens_reasoning",
        "latency_s", "finish_reason", "response_id", "model_snapshot", "timestamp"]


def build(calls_path=ROOT / "raw" / "calls.jsonl"):
    recs = {}
    for n in ("WW_AG", "WW_HC", "WWPRO_TXT"):
        for r in ld.load_normalized(n):
            recs[(n, r["traj_id"])] = r
    rows = []
    for d in map(json.loads, open(calls_path)):
        if d.get("error"):
            continue
        r = recs[(d["dataset"], d["traj_id"])]
        pr = parse_response(d.get("text"), d.get("finish_reason"))
        sc = score_run(pr, r, d.get("aliases"))
        rows.append({"dataset": d["dataset"], "judge": d["judge"], "condition": d["condition"], "replicate": d["replicate"], "traj_id": d["traj_id"],
                     "cluster_id": d["cluster_id"], "raw_response": d.get("text"), "parsed_agent": pr["agent"], "parsed_step": pr["step"],
                     "true_agent": "|".join(r["true_agents"]), "true_step": r["true_step_native"], **sc, "parse_failure_reason": pr["failure"], "L": r["L"],
                     **{k: d.get(k) for k in ("tokens_in", "tokens_out", "tokens_reasoning", "latency_s", "finish_reason", "response_id", "model_snapshot", "timestamp")}})
    return pd.DataFrame(rows, columns=COLS)


def main():
    df = build()
    e1 = df[df.replicate == 0].drop(columns=["replicate"])
    e2 = df[(df.replicate >= 1) & (df.condition == "O1")]
    (ROOT / "outputs/exp1").mkdir(parents=True, exist_ok=True); (ROOT / "outputs/exp2").mkdir(parents=True, exist_ok=True)
    e1.to_csv(ROOT / "outputs/exp1/X1_predictions.csv", index=False)
    e2.to_csv(ROOT / "outputs/exp2/X2_runs.csv", index=False)
    print(len(e1), "exp1 rows;", len(e2), "exp2 rows")


if __name__ == "__main__":
    main()
