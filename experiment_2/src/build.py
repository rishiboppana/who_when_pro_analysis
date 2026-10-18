"""Section 2: build E2_analysis_data.csv from E1_predictions.csv (Who&When Pro multi-agent text, prompts B and C, luna + mimo).
Adaptations vs the plan are listed in E2_notes.md."""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd, tiktoken

ROOT = Path(__file__).resolve().parents[1]; E1 = ROOT.parent / "experiment_1"
sys.path.insert(0, str(E1 / "src"))
from data import load_records
from render import render_A

PRIMARY = {"mathchat": "MC", "magentic-one": "MA"}          # frameworks with within-framework length variation
CODE = {"mathchat": "MC", "magentic-one": "MA", "debate": "DB", "dylan": "DY", "macnet": "MN", "metagpt": "MG"}
FAMILY = {"P": "Perception", "R": "Reasoning", "PL": "Planning", "A": "Action", "V": "Verification", "C": "Coordination"}
TASK_CAT = {"gpqa": "STEM", "mmlu_pro": "STEM", "math": "STEM", "scibench": "STEM", "bigcodebench": "Coding", "humaneval": "Coding",
            "livecodebench_pro": "Coding", "gaia": "Deep Search", "simpleqa_verified": "Deep Search"}


def build():
    recs = {r["id"]: r for r in load_records()}
    enc = tiktoken.get_encoding("o200k_base")
    meta = {}
    for tid, r in recs.items():
        meta[tid] = {"seed_group": hashlib.md5((r["framework"] + "||" + r["problem"]).encode()).hexdigest()[:12],
                     "token_length": len(enc.encode(render_A(r["entries"]), disallowed_special=())), "steps": r["gt_steps"]}
    P = pd.read_csv(E1 / "E1_predictions.csv")
    d = pd.DataFrame({"trajectory_id": P.trajectory_id, "model": P.model, "prompt": P.prompt, "framework": P.framework,
                      "subset": P.framework.map(CODE), "primary": P.framework.isin(PRIMARY), "benchmark": P.benchmark,
                      "task_type": P.benchmark.map(TASK_CAT), "error_type_mode": P["mode"],
                      "error_type_family": P["mode"].str.extract(r"^([A-Z]+)")[0].map(FAMILY), "length": P.n_entries,
                      "parse_ok": P.parse_ok.astype(int), "pred_step": P.pred_step, "step_success": P.step_correct.astype(int)})
    d["seed_group"] = d.trajectory_id.map(lambda t: meta[t]["seed_group"]); d["token_length"] = d.trajectory_id.map(lambda t: meta[t]["token_length"])
    steps = d.trajectory_id.map(lambda t: meta[t]["steps"])
    d["true_step"] = steps.map(min); d["true_step_max"] = steps.map(max)
    d["log2_length"] = np.log2(d.length); d["log2_token_length"] = np.log2(d.token_length)
    d["true_pos_rel"] = np.where(d.length > 1, d.true_step / (d.length - 1).clip(lower=1), 0.0)
    miss = (d.parse_ok == 1) & (d.step_success == 0)
    d["signed_error"] = np.where(miss, np.where(d.pred_step < d.true_step, d.pred_step - d.true_step, d.pred_step - d.true_step_max), np.nan)
    d["signed_error_norm"] = d.signed_error / d.length
    d["miss_direction"] = np.where(d.signed_error < 0, "early", np.where(d.signed_error > 0, "late", ""))
    d["late"] = np.where(miss, (d.signed_error > 0).astype(float), np.nan)
    d["out_of_range"] = ((d.pred_step >= d.length) | (d.pred_step < 0)).astype(int)
    d["level"] = ""; d["label_flag"] = 0
    return d


if __name__ == "__main__":
    d = build(); d.to_csv(ROOT / "E2_analysis_data.csv", index=False)
    print(len(d), "rows;", d.trajectory_id.nunique(), "traces;", d.seed_group.nunique(), "seed groups")
