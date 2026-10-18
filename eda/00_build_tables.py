"""Step 0: normalize data/text.jsonl into traces.parquet and steps.parquet.

Label fields are kept exactly as released (gt_agent, gt_agents, gt_step, gt_round,
gt_position, gt_stage). Only derived columns are added; see eda/outputs/schema_notes.md.

Run from repo root:  .venv/bin/python eda/00_build_tables.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)

# Task category per benchmark: inner ring of Figure 1 in the paper / ag2ai/whowhen_pro README.
# 'dsqa' is DeepSearchQA (paper App. H.1 lists 16 text datasets that match the 16 benchmark keys).
TASK_CATEGORY = {
    "alfworld": "Embodied",
    "gpqa": "STEM", "mmlu_pro": "STEM", "math": "STEM", "scibench": "STEM",
    "dabench": "Data Science", "dacode": "Data Science", "databench": "Data Science",
    "kramabench": "Data Science", "tablebench": "Data Science",
    "bigcodebench": "Coding", "humaneval": "Coding", "livecodebench_pro": "Coding",
    "gaia": "Deep Search", "simpleqa_verified": "Deep Search", "dsqa": "Deep Search",
}
# Topology per framework (paper App. G / README): smolagents and alfagent are single-agent.
SINGLE_AGENT = {"smolagents", "alfagent"}
FAMILY = {"P": "Perception", "R": "Reasoning", "PL": "Planning", "A": "Action",
          "V": "Verification", "C": "Coordination"}

tax = yaml.safe_load(open(ROOT / "data" / "taxonomy.yaml"))
MODE_NAME = {k: v["name"] for k, v in tax.items()}
pd.DataFrame([{"mode": k, "name": v["name"], "family": FAMILY[k.split(".")[0]]}
              for k, v in tax.items()]).to_csv(OUT / "tables" / "taxonomy_dataset.csv", index=False)


def L(x):
    return json.loads(x) if isinstance(x, str) else x


def words(s):
    return len(s.split()) if isinstance(s, str) else 0


def step_text(s):
    """Text of one trajectory element, whatever its kind."""
    parts = []
    for k in ("content", "reasoning", "action", "code", "observation", "output", "plan"):
        v = s.get(k)
        if isinstance(v, str):
            parts.append(v)
    for t in s.get("turns") or []:
        parts.append(t.get("output") or "")
    return "\n".join(parts)


def units(fw, traj):
    """Indices into traj that count as 'steps' for position purposes (exclude the task prompt
    and the final_answer record)."""
    if fw == "smolagents":  # label step 0 = the planning element, 1.. = action steps
        return [i for i, s in enumerate(traj) if s.get("kind") in ("planning", "action")]
    if fw == "alfagent":
        return [i for i, s in enumerate(traj) if s.get("kind") == "action"]
    if fw in ("debate", "dylan"):
        return [i for i, s in enumerate(traj) if s.get("kind") == "round"]
    return [i for i, s in enumerate(traj) if s.get("kind") not in ("user", "final_answer")]


def locate(fw, traj, gt, u):
    """Index (into traj) of the injected element, or None. Mirrors whowhen_eval/score.py."""
    if fw in ("mathchat", "macnet", "metagpt"):
        hit = [i for i in u if traj[i].get("is_injected")]
        return hit[0] if hit else None
    if fw in ("smolagents", "alfagent"):
        if gt.get("step") == 0:
            hit = [i for i in u if traj[i].get("kind") == "planning"]
        else:
            hit = [i for i in u if traj[i].get("kind") == "action" and traj[i].get("step_number") == gt.get("step")]
        return hit[0] if hit else None
    if fw in ("debate", "dylan"):
        hit = [i for i in u if traj[i].get("round") == gt.get("round")]
        return hit[0] if hit else None
    if fw == "magentic-one":
        try:
            r, p = str(gt["step"]).split(".")
            hit = [i for i in u if traj[i].get("round") == int(r) and traj[i].get("position") == int(p)]
        except (KeyError, ValueError):
            hit = []
        return hit[0] if hit else None
    return None


trace_rows, step_rows = [], []
for line in open(ROOT / "data" / "text.jsonl"):
    r = json.loads(line)
    fw, task, traj, gt, ex = r["framework"], L(r["task"]), L(r["trajectory"]), L(r["ground_truth"]), L(r["extras"])
    u = units(fw, traj)
    inj = locate(fw, traj, gt, u)
    k = u.index(inj) if inj is not None else None
    n = len(u)
    agent = gt.get("agent")
    trace_rows.append({
        "id": r["id"], "framework": fw, "benchmark": r["benchmark"],
        "task_category": TASK_CATEGORY[r["benchmark"]], "modality": "text",
        "topology": "single-agent" if fw in SINGLE_AGENT else "multi-agent",
        "backbone": "NA",
        "seed_group": hashlib.md5((fw + "||" + task["query"]).encode()).hexdigest()[:12],
        "task_query": task["query"], "gold_answer": task.get("answer"),
        "gt_agent": agent, "gt_agents": json.dumps(gt["agents"]) if "agents" in gt else None,
        "gt_step": None if gt.get("step") is None else str(gt["step"]),
        "gt_round": gt.get("round"), "gt_position": gt.get("position"), "gt_stage": gt.get("stage"),
        "gt_mode": gt["mode"], "mode_name": MODE_NAME[gt["mode"]],
        "family": FAMILY[gt["mode"].split(".")[0]],
        "n_accepted_alternates": len(gt.get("accepted_predictions") or []),
        "n_traj_elements": len(traj), "n_steps": n,
        "n_words": sum(words(step_text(s)) for s in traj),
        "inj_unit_idx": k,  # 0-indexed position among 'steps' (see units())
        "injection_pos_rel": None if k is None else (k + 1) / n,
        "steps_after_injection": None if k is None else n - (k + 1),
        "step_position_unit": {"smolagents": "planning/action step", "alfagent": "action step",
                               "debate": "round", "dylan": "round"}.get(fw, "agent turn"),
    })
    for i, s in enumerate(traj):
        step_rows.append({
            "trace_id": r["id"], "framework": fw, "idx": i, "kind": s.get("kind"),
            "step_number": s.get("step_number"),
            "agent": s.get("agent") or s.get("agent_id"), "role": s.get("role"),
            "round": s.get("round"), "position": s.get("position"), "stage": None if s.get("stage") is None else str(s.get("stage")),
            "is_injected_flag": s.get("is_injected"), "is_replayed_flag": s.get("is_replayed"),
            "is_decisive_step": i == inj,
            "n_words": words(step_text(s)), "text": step_text(s),
        })

traces = pd.DataFrame(trace_rows)
steps = pd.DataFrame(step_rows)
traces["seed_group_size"] = traces.groupby("seed_group")["id"].transform("size")
traces.to_parquet(OUT / "tables" / "traces.parquet", index=False)
steps.to_parquet(OUT / "tables" / "steps.parquet", index=False)
print(len(traces), "traces;", len(steps), "steps")
print("unlocated injection:", int(traces["inj_unit_idx"].isna().sum()))
print(traces.groupby("framework")[["n_steps", "n_words", "injection_pos_rel", "seed_group_size"]].mean().round(2))
