"""Load the Who&When Pro multi-agent text traces (2,670), flatten each into an ordered entry list and map the
released label to the flat 0-based index. Entry text comes from the official whowhen_eval per-framework renderers
(turn text with their `Step R.P | Agent: x` header removed), so no turn content is re-written here."""
import json, sys
from pathlib import Path

import os
REPO = os.environ.get("WHOWHEN_EVAL_REPO", "/private/tmp/claude-501/-Users-user-Desktop-thesis-who-and-when-pro-analysis/7923cc4d-c32a-4b06-a28c-1710d742019d/scratchpad/repo")  # clone of ag2ai/whowhen_pro @ 14369dc
sys.path.insert(0, REPO)
from whowhen_eval.render import get_renderer, TASK_ANCHOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MULTI = {"macnet", "metagpt", "magentic-one", "debate", "dylan", "mathchat"}
J = lambda x: json.loads(x) if isinstance(x, str) else x


def flatten(rel):
    """-> list of (agent, body, coord) in order. Task/user-input anchor excluded, except mathchat's first
    user_proxy turn, which is a real turn (flat index 0 = 2*round+position convention)."""
    rr = get_renderer(rel["framework"])(rel)
    out = []
    for b in rr.blocks:
        if b.coord == TASK_ANCHOR:
            continue
        head, _, body = b.text.partition("\n")
        if " | Agent: " in head:
            out.append((head.split(" | Agent: ", 1)[1].strip(), body, b.coord))
        elif head.strip() == "User Input":
            out.append(("user_proxy", body, None))
        else:
            raise ValueError(f"unrecognised block header {head!r} in {rel['id']}")
    return out


def gt_flat(fw, gt, entries):
    """Return (set of correct flat step indices, set of correct agents). Mapping follows whowhen_eval/score.py."""
    coords = [e[2] for e in entries]
    agents = set()
    if gt.get("agent"):
        agents.add(gt["agent"])
    agents |= set(gt.get("agents") or [])
    if fw == "mathchat":
        steps = {2 * int(gt["round"]) + int(gt["position"])}
    elif fw == "metagpt":
        steps = {int(gt["stage"])}
    elif fw == "macnet":
        steps = {coords.index(f"{gt['round']}.{gt['position']}")}
    elif fw == "magentic-one":
        steps = {coords.index(str(gt["step"]))}
    elif fw in ("debate", "dylan"):  # round-level label: any turn inside the round counts (official scorer)
        steps = {i for i, c in enumerate(coords) if c and c.split(".")[0] == str(gt["round"])}
    else:
        raise ValueError(fw)
    return steps, agents


def load_records():
    recs = []
    for line in open(ROOT / "data" / "text.jsonl"):
        r = json.loads(line)
        if r["framework"] not in MULTI:
            continue
        rel = {k: (J(v) if k in ("task", "trajectory", "ground_truth", "extras") else v) for k, v in r.items()}
        ents = flatten(rel)
        steps, agents = gt_flat(r["framework"], rel["ground_truth"], ents)
        recs.append({"id": r["id"], "framework": r["framework"], "benchmark": r["benchmark"],
                     "problem": rel["task"]["query"], "ground_truth": rel["task"].get("answer"),
                     "entries": [(a, b) for a, b, _ in ents], "gt_steps": sorted(steps), "gt_agents": sorted(agents),
                     "mode": rel["ground_truth"]["mode"], "n_entries": len(ents),
                     "round_level": r["framework"] in ("debate", "dylan")})
    return sorted(recs, key=lambda x: x["id"])
