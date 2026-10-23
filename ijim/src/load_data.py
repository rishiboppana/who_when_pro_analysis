"""Load W&W (AG, HC) and Who&When Pro (text, filtered + sampled) into normalized records (plan section 3).
Record: dataset, traj_id, cluster_id, question, final_answer_or_null, events[{i, agent_raw, role_raw, content,
native_fields}], L, true_agent, true_agents, true_step_native, step_base, labels (never rendered).
No label ever enters a prompt: render.py reads only question, final_answer_or_null and events."""
import hashlib, json, math, random, re, sys
from collections import defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
WW_DIR = ROOT / "vendor" / "Agents_Failure_Attribution" / "Who&When"
PRO_JSONL = REPO / "data" / "text.jsonl"
PRO_VENDOR = REPO / "vendor" / "whowhen_pro"
NORM_DIR = ROOT / "data" / "normalized"
CFG = yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text())
SEED = CFG["analysis_seed"]
ALWAYS_EXCLUDED = {"debate", "dylan"}    # round-level labels: a step label is a round, not a single stored event


def cluster_id(question):
    norm = re.sub(r"\s+", " ", str(question)).strip().lower()
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def _rec(dataset, traj_id, question, answer, events, true_agents, true_step, labels):
    for i, e in enumerate(events):
        e["i"] = i
    return {"dataset": dataset, "traj_id": traj_id, "cluster_id": cluster_id(question), "question": question,
            "final_answer_or_null": answer, "events": events, "L": len(events), "true_agent": true_agents[0],
            "true_agents": true_agents, "true_step_native": true_step, "step_base": 0, "labels": labels}


def load_ww(subset):
    """subset in {Algorithm-Generated, Hand-Crafted}. Speaker key is `name` (AG) / `role` (HC), as in the original
    all_at_once. mistake_step is a string in the raw files; converted to int (0-based, indexes history directly)."""
    ds = "WW_AG" if subset == "Algorithm-Generated" else "WW_HC"
    hc = subset == "Hand-Crafted"
    out = []
    for p in sorted((WW_DIR / subset).glob("*.json"), key=lambda p: int(p.stem)):
        d = json.loads(p.read_text())
        events = []
        for h in d["history"]:
            speaker = h.get("role" if hc else "name", "Unknown Agent")
            known = {"name", "role", "content"}
            events.append({"agent_raw": speaker, "role_raw": None if hc else h.get("role"),
                           "content": h.get("content", ""),
                           "native_fields": {k: v for k, v in h.items() if k not in known or (k == "role" and not hc)}})
        answer = d.get("ground_truth", d.get("groundtruth"))
        labels = {k: d.get(k) for k in ("mistake_agent", "mistake_step", "mistake_reason", "is_correct", "level",
                                        "mistake_type") if k in d}
        out.append(_rec(ds, f"{subset}/{p.stem}", d["question"], answer, events, [str(d["mistake_agent"])],
                        int(d["mistake_step"]), labels))
    return out


# ---------------------------------------------------------------- Who&When Pro
def _pro_imports():
    sys.path.insert(0, str(PRO_VENDOR))
    from whowhen_eval.render import get_renderer, TASK_ANCHOR
    return get_renderer, TASK_ANCHOR


_J = lambda x: json.loads(x) if isinstance(x, str) else x


def pro_flatten(rel):
    """Ordered (agent, body, native_coord) from the official per-framework renderer, task anchor excluded
    (mathchat's first user_proxy turn is a real turn). Turn text is not rewritten, only its header is removed."""
    get_renderer, ANCHOR = _pro_imports()
    out = []
    for b in get_renderer(rel["framework"])(rel).blocks:
        if b.coord == ANCHOR:
            continue
        head, _, body = b.text.partition("\n")
        if " | Agent: " in head:
            out.append((head.split(" | Agent: ", 1)[1].strip(), body, b.coord))
        elif head.strip() == "User Input":
            out.append(("user_proxy", body, None))
        else:
            raise ValueError(f"unrecognised block header {head!r} in {rel['id']}")
    return out


def pro_step(fw, gt, coords):
    """Single flat 0-based event index for the label, or None. macnet / magentic-one: exact stored coordinate lookup.
    mathchat: 2*round+position and metagpt: stage are the official scorer's flat-index conventions (whowhen_eval/score.py);
    outputs/index_check.md reports how often the labelled agent speaks at that event."""
    try:
        if fw == "macnet":
            return coords.index(f"{gt['round']}.{gt['position']}")
        if fw == "magentic-one":
            return coords.index(str(gt["step"]))
        if fw == "mathchat":
            return 2 * int(gt["round"]) + int(gt["position"])
        if fw == "metagpt":
            return int(gt["stage"])
    except (KeyError, ValueError, TypeError):
        return None
    return None


def pro_scan(frameworks=None):
    """-> (kept records, exclusion counter). Filtering order is recorded for dataset_inclusion_report.md."""
    keep_fw = set(frameworks or CFG["datasets"]["WWPRO_TXT"]["frameworks"])
    kept, ex = [], defaultdict(int)
    for line in open(PRO_JSONL):
        r = json.loads(line)
        fw = r["framework"]
        ex["rows_total"] += 1
        gt = _J(r["ground_truth"])
        if fw in ALWAYS_EXCLUDED:
            ex[f"excluded_round_level_label:{fw}"] += 1; continue
        if fw not in keep_fw:
            ex[f"excluded_framework_not_in_keep_list:{fw}"] += 1; continue
        if not gt.get("agent"):
            ex[f"excluded_null_agent:{fw}"] += 1; continue
        rel = {k: (_J(v) if k in ("task", "trajectory", "ground_truth", "extras") else v) for k, v in r.items()}
        ents = pro_flatten(rel)
        step = pro_step(fw, gt, [e[2] for e in ents])
        if step is None or not (0 <= step < len(ents)):
            ex[f"excluded_step_not_indexing_event_list:{fw}"] += 1; continue
        events = [{"agent_raw": a, "role_raw": None, "content": b, "native_fields": {"coord": c}} for a, b, c in ents]
        task = rel["task"]
        rec = _rec("WWPRO_TXT", r["id"], task["query"], task.get("answer"), events, [gt["agent"]] + list(gt.get("agents") or []),
                   step, {"ground_truth_json": r["ground_truth"], "mode": gt.get("mode"), "framework": fw})
        rec["true_agent"] = gt["agent"]
        rec["framework"] = fw
        kept.append(rec)
        ex[f"kept:{fw}"] += 1
    return kept, dict(ex)


def pro_sample(kept, target=None, min_per=None, seed=SEED):
    """Stratified by framework: min(min_per, available) each, the rest allocated proportionally to the remaining
    capacity (largest remainder). Deterministic given the seed."""
    cfg = CFG["datasets"]["WWPRO_TXT"]
    target, min_per = target or cfg["target_n"], min_per or cfg["min_per_framework"]
    by = defaultdict(list)
    for r in kept:
        by[r["framework"]].append(r["traj_id"])
    if len(kept) <= target:
        return sorted(r["traj_id"] for r in kept)
    base = {f: min(min_per, len(v)) for f, v in by.items()}
    spare = target - sum(base.values())
    cap = {f: len(by[f]) - base[f] for f in by}
    tot = sum(cap.values())
    raw = {f: spare * cap[f] / tot for f in by}
    alloc = {f: base[f] + int(math.floor(raw[f])) for f in by}
    for f in sorted(by, key=lambda f: (-(raw[f] - math.floor(raw[f])), f))[: target - sum(alloc.values())]:
        alloc[f] += 1
    rng, ids = random.Random(seed), []
    for f in sorted(by):
        ids += rng.sample(sorted(by[f]), min(alloc[f], len(by[f])))
    return sorted(ids)


def load_pro(sample_path=ROOT / "config" / "pro_sample_ids.json", create=True):
    kept, _ = pro_scan()
    if Path(sample_path).exists():
        ids = set(json.loads(Path(sample_path).read_text())["ids"])
    elif create:
        sel = pro_sample(kept)
        Path(sample_path).write_text(json.dumps({"seed": SEED, "n": len(sel), "ids": sel}, indent=1))
        ids = set(sel)
    else:
        ids = {r["traj_id"] for r in kept}
    return [r for r in kept if r["traj_id"] in ids]


def load_dataset(name):
    if name == "WW_AG":
        return load_ww("Algorithm-Generated")
    if name == "WW_HC":
        return load_ww("Hand-Crafted")
    if name == "WWPRO_TXT":
        return load_pro()
    raise ValueError(name)


def load_normalized(name):
    p = NORM_DIR / f"{name}.jsonl"
    return [json.loads(l) for l in open(p)]


def write_normalized(names=("WW_AG", "WW_HC", "WWPRO_TXT")):
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        recs = load_dataset(n)
        with open(NORM_DIR / f"{n}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(n, len(recs))


if __name__ == "__main__":
    write_normalized()
