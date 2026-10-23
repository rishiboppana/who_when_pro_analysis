"""Phase 0/1 recon (no model calls). Writes: config/schema_notes.md, config/index_convention.md, config/provenance_inventory.md,
outputs/dataset_inclusion_report.md, outputs/index_check.md, outputs/prompt_token_report.md, outputs/examples/*.txt and the
normalized data. Run: python src/recon.py"""
import json, re, statistics, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
import tiktoken
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import load_data as ld
from render import build_prompt, CONDITIONS
from score import match_agent

ROOT = ld.ROOT
OUT = ROOT / "outputs"
ENC = tiktoken.get_encoding("o200k_base")     # approximation of the judge tokenizer
CTX, MAXOUT = 400_000, 32_768                  # judge context window assumed; CHECK against provider docs before the freeze
tok = lambda s: len(ENC.encode(s, disallowed_special=()))
cfg = yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text())["datasets"]


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def agent_check(recs, n=None):
    hit = tot = 0
    for r in recs[:n]:
        tot += 1
        hit += match_agent(r["true_agents"], r["events"][r["true_step_native"]]["agent_raw"]) or any(
            t.lower() in r["events"][r["true_step_native"]]["agent_raw"].lower() for t in r["true_agents"])
    return hit, tot


def main():
    OUT.mkdir(exist_ok=True); (OUT / "examples").mkdir(exist_ok=True)
    ww_sha = sh(f"git -C '{ld.ROOT / 'vendor/Agents_Failure_Attribution'}' rev-parse HEAD")
    ag, hc = ld.load_ww("Algorithm-Generated"), ld.load_ww("Hand-Crafted")
    kept_all, ex = ld.pro_scan(frameworks=["magentic-one", "macnet", "mathchat", "metagpt"])   # candidate frameworks for the index check
    pro = ld.load_pro()
    data = {"WW_AG": ag, "WW_HC": hc, "WWPRO_TXT": pro}
    ld.write_normalized()

    # ---- schema notes
    rows = [sum(1 for _ in open(ld.PRO_JSONL))]
    fw_counts = Counter(json.loads(l)["framework"] for l in open(ld.PRO_JSONL))
    notes = [f"# Schema notes (Phase 0)\n",
             f"W&W source: git clone of https://github.com/ag2ai/Agents_Failure_Attribution @ {ww_sha} (under ijim/vendor, gitignored).",
             f"Counts: Algorithm-Generated {len(ag)} (expected 126), Hand-Crafted {len(hc)} (expected 58), total {len(ag)+len(hc)} (expected 184).",
             "Record keys and history entry keys are read from the files (see load_data.load_ww); `mistake_step` is a STRING in the raw files and is converted to int.",
             f"Pro source: local copy data/text.jsonl of HF Leoxx/whowhen_pro text modality, {rows[0]} rows (card says text ~6.3k). Frameworks: {dict(fw_counts)}.",
             "Pro columns: id, framework, benchmark, task (JSON: query, answer), trajectory (JSON), ground_truth (JSON, = the error label), extras (JSON).",
             "Pro final answer: `task.answer`; null for some benchmarks (LiveCodeBench-Pro). `extras` was not used as an answer source.",
             "AgentRx (HF microsoft/AgentRx): gated; the data files returned HTTP 403 with the available HF token. Not loaded."]
    # W&W entry keys
    for name, recs in (("WW_AG", ag), ("WW_HC", hc)):
        ks = Counter(k for r in recs for e in r["events"] for k in ["agent_raw"] + list(e["native_fields"]))
        notes.append(f"{name} event keys beyond content: {dict(ks)}")
    (ROOT / "config" / "schema_notes.md").write_text("\n".join(notes) + "\n")

    # ---- index convention (+ per-framework Pro check, >=10 rows each)
    ic = ["# Step-index convention\n", "Label coordinate: W&W `mistake_step` is 0-based and indexes `history` directly (re-verified below, rule: labelled agent speaks at that event, case-insensitive substring).",
          "Pro: the label is mapped to a single flat 0-based event index of the official renderer output (task anchor excluded)."]
    chk = ["# Index check: does the labelled agent speak at the labelled event?\n", "| dataset/framework | n | agent at step | agent at step-1 | agent at step+1 |", "|---|---|---|---|---|"]
    groups = [("WW_AG", ag), ("WW_HC", hc)]
    byfw = defaultdict(list)
    for r in kept_all:
        byfw[r["framework"]].append(r)
    groups += [(f"WWPRO/{f}", v) for f, v in sorted(byfw.items())]
    for name, recs in groups:
        def at(off):
            h = t = 0
            for r in recs:
                j = r["true_step_native"] + off
                if 0 <= j < r["L"]:
                    t += 1
                    h += any(x.lower() in r["events"][j]["agent_raw"].lower() for x in r["true_agents"])
            return f"{h}/{t}"
        chk.append(f"| {name} | {len(recs)} | {at(0)} | {at(-1)} | {at(1)} |")
    (OUT / "index_check.md").write_text("\n".join(chk) + "\n\nA framework is kept only if 'agent at step' is clearly the best offset (decided by the user at STOP 1).\n")
    (ROOT / "config" / "index_convention.md").write_text("\n".join(ic) + "\n\nSee outputs/index_check.md for the per-framework agent-at-step table.\n")

    # ---- provenance inventory
    pi = ["# Provenance inventory (fields beyond what render_O0 shows)\n"]
    for name, recs in data.items():
        n = sum(r["L"] for r in recs)
        cov = Counter(k for r in recs for e in r["events"] for k, v in e["native_fields"].items() if v is not None)
        pi.append(f"## {name}  ({n} events)")
        for k, c in sorted(cov.items()):
            shown = (name == "WW_HC" and k == "role") or k == "coord"
            why = "already in O0 (speaker label)" if name == "WW_HC" and k == "role" else (
                "step coordinate of the source framework, not provenance: excluded to avoid a second step numbering" if k == "coord" else "")
            pi.append(f"- `{k}`: {100*c/n:.1f}% of events {('-> NOT usable for O2: ' + why) if why else ''}")
        pi.append(f"- O2 fields configured: {cfg[name]['o2_fields']} -> " + ("O2 applicable" if cfg[name]["o2_fields"] else "O2 not applicable (O2 would equal O1)") + "\n")
    (ROOT / "config" / "provenance_inventory.md").write_text("\n".join(pi) + "\n")

    # ---- variants + prompt tokens + examples
    answered = sum(r["final_answer_or_null"] not in (None, "") for r in pro)
    inc = ["# Dataset inclusion report\n",
           f"- WW_AG: {len(ag)} kept (expected 126). WW_HC: {len(hc)} kept (expected 58).",
           f"- Pro text rows scanned: {ex.get('rows_total')}. Exclusions and kept counts (candidate frameworks magentic-one, macnet, mathchat, metagpt):"]
    inc += [f"  - {k}: {v}" for k, v in sorted(ex.items())]
    inc += [f"- Pro sample: n={len(pro)} (target 300, seed {ld.SEED}); per framework {dict(Counter(r['framework'] for r in pro))}",
            f"- Pro stored final answer in {answered}/{len(pro)} sampled rows ({100*answered/max(len(pro),1):.0f}%) -> answer_variant should be {'with_answer' if answered/max(len(pro),1)>=0.95 else 'no_answer'} (set in config/datasets.yaml)",
            "- AgentRx: microsoft/AgentRx is gated; data files returned HTTP 403 with the available HF token. Its card documents root-cause failures with step_number and failed_agent, so it could qualify; the user must accept the access terms. Excluded for now.",
            "- Aegis-Bench (no step labels) and TRAIL (span-level, no agent label) are excluded as ineligible.",
            f"- Eligible datasets: 3 (WW_AG, WW_HC, WWPRO_TXT) -> leave-one-dataset-out works with 3, but each training fold has only 2 datasets."]
    (OUT / "dataset_inclusion_report.md").write_text("\n".join(inc) + "\n")

    tr = ["# Rendered prompt token report (o200k_base approximation of the judge tokenizer)\n", f"Assumed judge context {CTX}, max output {MAXOUT}: limit {CTX-MAXOUT}. VERIFY the context window in the provider docs.\n",
          "| dataset | condition | n | max | median | 5 longest (tokens) | over limit |", "|---|---|---|---|---|---|---|"]
    for name, recs in data.items():
        variant = cfg[name]["answer_variant"] if cfg[name]["answer_variant"] != "auto" else ("with_answer" if answered / max(len(pro), 1) >= .95 else "no_answer")
        for c in CONDITIONS:
            if c == "O2" and not cfg[name]["o2_fields"]:
                continue
            ts = sorted((tok(build_prompt(r, c, variant, cfg[name]["o2_fields"])[0]) for r in recs), reverse=True)
            tr.append(f"| {name} | {c} | {len(ts)} | {ts[0]} | {int(statistics.median(ts))} | {ts[:5]} | {sum(t > CTX-MAXOUT for t in ts)} |")
        r = recs[0] if name != "WWPRO_TXT" else sorted(recs, key=lambda r: r["L"])[len(recs)//2]
        for c in CONDITIONS:
            if c == "O2" and not cfg[name]["o2_fields"]:
                continue
            p, al, n = build_prompt(r, c, variant, cfg[name]["o2_fields"])
            (OUT / "examples" / f"{name}_{c}.txt").write_text(f"# traj_id={r['traj_id']} condition={c} variant={variant} alias_replacements_in_content={n}\n\n{p}")
    (OUT / "prompt_token_report.md").write_text("\n".join(tr) + "\n")

    # original-rule scorability of the labelled agent
    sc = ["# Agent-rule scorability (original evaluate.py token rule)\n"]
    for name, recs in data.items():
        ok = sum(match_agent(r["true_agents"], r["true_agents"][0]) for r in recs)
        sc.append(f"- {name}: labelled agent matches itself under the `[\\w_]+` token rule in {ok}/{len(recs)} records (a lower number means the rule can never score those as correct).")
    (OUT / "agent_rule_scorability.md").write_text("\n".join(sc) + "\n")
    print("done")


if __name__ == "__main__":
    main()
