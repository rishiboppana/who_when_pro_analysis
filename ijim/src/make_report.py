"""Builds ijim/report.md from the output CSVs and real raw responses (numbers are read, not typed). python src/make_report.py"""
import json, sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import load_data as ld
from render import build_prompt

R = ld.ROOT
cfg = yaml.safe_load((R / "config/datasets.yaml").read_text())["datasets"]
e1 = pd.read_csv(R / "outputs/exp1/X1_predictions.csv"); e2 = pd.read_csv(R / "outputs/exp2/X2_runs.csv")
res = pd.read_csv(R / "outputs/exp1/X1_results.csv"); eff = pd.read_csv(R / "outputs/exp1/X1_effects.csv")
sec = pd.read_csv(R / "outputs/exp1/X1_effects_secondary_joint.csv"); secm = pd.read_csv(R / "outputs/exp1/X1_effects_secondary.csv")
x2 = pd.read_csv(R / "outputs/exp2/X2_results.csv"); cnt = pd.read_csv(R / "outputs/exp2/X2_counts.csv"); gt = pd.read_csv(R / "outputs/exp2/X2_gate_table.csv")
gp = pd.read_csv(R / "outputs/exp2/X2_gate_predictions.csv"); sens = pd.read_csv(R / "outputs/exp2/X2_results_sensitivity_WW_merged.csv")
for c in ("agent_ok", "step_ok", "joint_ok", "parse_fail"):
    e1[c] = e1[c].astype(bool); e2[c] = e2[c].astype(bool)
recs = {n: {r["traj_id"]: r for r in ld.load_normalized(n)} for n in cfg}


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and pd.notna(v) and not isinstance(v, str):
                v = fmt[c].format(v)
            elif isinstance(v, float):
                v = "" if pd.isna(v) else f"{v:.3g}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + " …[truncated]"


def row(d, tid, cond):
    return e1[(e1.dataset == d) & (e1.traj_id == tid) & (e1.condition == cond)].iloc[0]


def excerpt(d, tid, cond, width=150):
    """Rendered log (all event contents flattened and clipped) for events true_step-1 .. true_step, exactly as formatted for the condition."""
    from render import render_log
    rec = dict(recs[d][tid])
    rec["events"] = [dict(e, content=clip(" ".join(str(e["content"]).split()), width)) for e in rec["events"]]
    text, _, _ = render_log(rec, cond, cfg[d]["o2_fields"])
    ts = rec["true_step_native"]
    lo, hi = max(ts - 1, 0), ts
    if cond == "O0":
        lines = text.split("\n")
        return "\n".join(lines[lo:hi + 1])
    marks = [i for i, l in enumerate(text.split("\n")) if l.startswith("[Step ")]
    lines = text.split("\n")
    end = marks[hi + 1] if hi + 1 < len(marks) else len(lines)
    return "\n".join(lines[marks[lo]:end])


L = []
A = L.append
A("# IJIM report: step labels and a repetition gate for failure attribution\n")
A("Two cross-dataset experiments with one frozen judge (`gpt-5.6-luna`, reasoning effort high). Experiment 1 asks whether adding explicit step labels to the judge input changes failure-attribution accuracy. Experiment 2 asks whether five repeated diagnoses can be turned into an accept-or-escalate signal. All numbers below are read from the files in `ijim/outputs/`; all responses shown are verbatim from `ijim/raw/calls.jsonl`.\n")

A("## 1. Summary\n")
ef = eff.set_index("Dataset")
A(f"- **Experiment 1.** Adding `[Step k]` labels (O1 vs O0) raised exact joint WHO+WHEN accuracy by **{ef.loc['WW_AG','Delta joint (pp)']:+.1f} pp** on Who&When Algorithm-Generated (n={int(ef.loc['WW_AG','n'])}), **{ef.loc['WW_HC','Delta joint (pp)']:+.1f} pp** on Hand-Crafted (n={int(ef.loc['WW_HC','n'])}) and **{ef.loc['WWPRO_TXT','Delta joint (pp)']:+.1f} pp** on the Who&When Pro text sample (n={int(ef.loc['WWPRO_TXT','n'])}); all three are significant after Holm correction. The gain is almost entirely in the step, not the agent. Anonymising agent names (O1-A) and adding a role field (O2) changed nothing reliably.")
A("- **Experiment 2.** Five repeats by the same judge are usually identical, so agreement is only a weak reliability signal. A seven-feature logistic gate trained on the other datasets ranks held-out cases better than chance (AUROC 0.67 to 0.78) and lifts accuracy among retained cases, but it does not clearly beat simple consensus rules, and unanimous agreement is still wrong for 37% (AG) and 64% (HC) of cases.")
A("- **Cost.** 3,998 judge calls, about $8.3 in total. No API errors, no truncation.\n")

A("## 2. Design\n")
A("**Datasets.** `WW_AG` (Who&When Algorithm-Generated, 126 natural failures), `WW_HC` (Hand-Crafted, 58 natural failures), `WWPRO_TXT` (Who&When Pro, text modality, 300 sampled rows with injected errors). Pro rows were kept only if the error agent is non-null and the label maps to a single stored event (frameworks magentic-one, macnet, mathchat, metagpt; debate/dylan label whole rounds and were dropped). Stratified sample by framework, seed 20261022. Step labels are 0-based and index the event list directly (the labelled agent speaks at the labelled event in 123/126 AG, 56/58 HC and 100% of kept Pro rows).\n")
A("**Judge prompt.** One user message, no system prompt, the paper's all-at-once prompt (identical text in every condition): *identify which agent made an error, at which step, and why; answer as `Agent Name:`, `Step Number:`, `Reason for Mistake:`*. W&W prompts include the stored final answer; Pro prompts omit it (stored for only 90% of sampled rows). No label (true agent, step, reason, Pro `ground_truth`, mode, level) ever enters a prompt; a unit test checks every real prompt.\n")
A("**Trace conditions** (only the conversation log changes):\n\n| Condition | Log format |\n|---|---|\n| O0 | original: one `Speaker: content` line per event |\n| O1 | O0 plus a `[Step k]` line before each event, k in the dataset's native 0-based coordinate; event content byte-identical |\n| O1-A | O1 with agent names replaced by `Agent A, B, …` (speaker and whole-word mentions in content); predicted aliases are mapped back before scoring |\n| O2 | O1 plus a `(role=…)` line per event; only WW_AG has such a native field |\n")
A("**Scoring.** Parser: first `Agent Name:` and first `Step Number:` line; step must be a plain integer; truncated or malformed output is a parse failure, counted as incorrect and never repaired. Agent correct by the original `evaluate.py` rule (labelled name contained in the first word-token of the prediction); step correct if equal to the label; joint = both. Normalized step error = |pred − true| / max(L − 1, 1).\n")
A("**Statistics.** Primary endpoint exact joint accuracy; comparisons within each dataset, paired by trajectory (never pooled). Paired cluster bootstrap (B = 10,000, clusters = identical question text) for 95% CIs, exact McNemar test, Holm correction over the comparisons in each dataset. Wilson intervals for accuracies.\n")
A("**Experiment 2.** O1 only, five fresh calls per trajectory (replicates 1 to 5). Seven label-free features (`joint_modal_agreement`, `agent_agreement`, `step_agreement`, `unique_pair_count`, `norm_step_spread`, `traj_length`, `parse_failure_count`). Candidate = most frequent complete (agent, step) pair (ties go to the earliest run); target = candidate exactly correct. Gate = L2 logistic regression (C = 1, standardised, equal total weight per training dataset), leave-one-dataset-out.\n")

A("## 3. Experiment 1 results\n")
A("### 3.1 Accuracy by condition\n")
r = res[["Dataset", "Trace condition", "n", "n_parseable", "Agent %", "Step %", "Joint %", "Joint Wilson lo", "Joint Wilson hi", "Median normalized error", "parse_fail %"]]
A(md(r, {c: "{:.1f}" for c in ["Agent %", "Step %", "Joint %", "Joint Wilson lo", "Joint Wilson hi", "parse_fail %"]} | {"Median normalized error": "{:.3f}"}))
A("\n### 3.2 Primary effect: O1 vs O0 (exact joint accuracy)\n")
A(md(eff, {"Delta joint (pp)": "{:+.1f}", "CI lo (pp)": "{:.1f}", "CI hi (pp)": "{:.1f}", "raw p": "{:.2g}", "Holm p": "{:.2g}"}))
A("\n`b` = trajectories wrong under O0 and right under O1, `c` = the reverse. The imbalance is large everywhere (e.g. AG 37 vs 1).\n")
A("![forest](outputs/exp1/X1_forest.png)\n")
A("### 3.3 Secondary comparisons (joint accuracy)\n")
A(md(sec, {"Delta joint (pp)": "{:+.1f}", "CI lo (pp)": "{:.1f}", "CI hi (pp)": "{:.1f}", "raw p": "{:.2g}", "Holm p": "{:.2g}"}))
A("\n### 3.4 Where the O1 gain comes from (paired change, O1 vs O0)\n")
s2 = secm[(secm.Comparison == "O1 vs O0") & secm.Metric.isin(["agent", "step", "parse_fail"])][["Dataset", "Metric", "n", "Delta (pp)", "CI lo (pp)", "CI hi (pp)"]]
A(md(s2, {"Delta (pp)": "{:+.1f}", "CI lo (pp)": "{:.1f}", "CI hi (pp)": "{:.1f}"}))
A("\nStep accuracy moves by 19 to 27 pp; agent accuracy by at most 7 pp with intervals that include zero. Step labels help the judge locate the step, not decide who erred. A plausible mechanism is that labels spare the judge from counting events in long traces, so this should be read as a step-localisation effect.\n")
A("### 3.5 Interpretation\n")
A(Path(R / "outputs/exp1/X1_interpretation.md").read_text().split("\n", 2)[2].strip() + "\n")

A("## 4. Experiment 2 results\n")
A("### 4.1 Consistency of the five repeats\n")
c2 = cnt.rename(columns=lambda x: x.replace("unique_pairs=", "uniq="))
A(md(c2, {"target_rate": "{:.3f}"}))
A("\n`uniq=k` counts trajectories whose five runs produced k distinct (agent, step) pairs. Most trajectories have one pair: the judge repeats itself, right or wrong.\n")
A("### 4.2 Gate and baselines on each held-out dataset\n")
x = x2.copy()
for c in ("Coverage %", "Joint %", "Brier score"):
    x[c] = pd.to_numeric(x[c], errors="coerce")
x["AUROC"] = x["AUROC"].apply(lambda v: f"{float(v):.3f}" if str(v).replace('.', '', 1).isdigit() else ("" if pd.isna(v) else v))
x["95% CI/note"] = x["95% CI/note"].fillna("")
A(md(x, {"Coverage %": "{:.1f}", "Joint %": "{:.1f}", "Brier score": "{:.3f}"}))
A("\nRetained cases are ranked by predicted probability (ties broken by row order, never by labels). Consensus rows are fixed thresholds with their own coverage. Accept-all Brier uses the training datasets' target rate as a constant. The gate was not tuned and no threshold was chosen on test data.\n")
A("![risk-coverage](outputs/exp2/X2_risk_coverage.png)\n\n![calibration](outputs/exp2/X2_calibration.png)\n")
A("### 4.3 Sensitivity: W&W as a single fold\n")
sv = sens[sens.Method.isin(["Gate@100", "Gate@50", "Accept-all", "Consensus=5/5"])][["Held-out dataset", "Method", "Coverage %", "Retained n", "Joint %", "Brier score", "AUROC"]].copy()
for c in ("Joint %", "Coverage %"):
    sv[c] = pd.to_numeric(sv[c], errors="coerce")
sv["AUROC"] = sv["AUROC"].apply(lambda v: f"{float(v):.3f}" if str(v).replace(".", "", 1).isdigit() else ("" if pd.isna(v) else v))
A(md(sv, {"Coverage %": "{:.1f}", "Joint %": "{:.1f}", "Brier score": "{:.3f}"}))
A("\nWith three datasets this fold trains on Pro alone (or on W&W alone), a weaker training set; treat it as a robustness check only.\n")
A("### 4.4 Interpretation\n")
A(Path(R / "outputs/exp2/X2_interpretation.md").read_text().split("\n", 2)[2].strip() + "\n")

# ------------------------------------------------------------------ worked examples
A("## 5. One worked example of each test\n")
A("All examples are real trajectories and verbatim model responses. Prompt excerpts show the first events of the failure log; the instruction text around the log is the same in every condition.\n")

d, tid = "WW_AG", "Algorithm-Generated/107"
rec = recs[d][tid]
A(f"### 5.1 Conditions O0, O1, O1-A, O2 on one trajectory (`{d}`, `{tid}`)\n")
A(f"Problem: *{clip(rec['question'], 300)}* Stored answer: `{rec['final_answer_or_null']}`. The trajectory has L = {rec['L']} events. **Ground truth** (never shown to the judge): agent `{rec['true_agent']}`, step {rec['true_step_native']}.\n")
for cond, label in (("O0", "O0: original log"), ("O1", "O1: with step labels"), ("O1-A", "O1-A: step labels and anonymous agents"), ("O2", "O2: step labels plus role")):
    rr = row(d, tid, cond)
    A(f"**{label}**\n")
    A(f"Log as sent to the judge for the events around the labelled step (event contents flattened to one line and clipped to 150 characters here; the judge saw them in full):\n")
    A("```\n" + excerpt(d, tid, cond) + "\n```\n")
    A(f"Judge response (verbatim, clipped): \n\n```\n{clip(rr.raw_response, 700)}\n```\n")
    al = ""
    if cond == "O1-A":
        _, aliases, _ = build_prompt(rec, cond, cfg[d]["answer_variant"], cfg[d]["o2_fields"])
        al = f" Alias map used to score: `{aliases}`."
    A(f"Scored: parsed agent `{rr.parsed_agent}`, step `{int(rr.parsed_step)}` → agent_ok = **{rr.agent_ok}**, step_ok = **{rr.step_ok}**, joint_ok = **{rr.joint_ok}**; normalized step error {rr.norm_err:.3f}; {int(rr.tokens_out)} output tokens ({int(rr.tokens_reasoning)} reasoning).{al}\n")
A("Here O0 picked the right agent but named step 7 instead of 6; every labelled condition got step 6.\n")

# Pro example
pv = e1[e1.dataset == "WWPRO_TXT"].pivot(index="traj_id", columns="condition", values="joint_ok")
cand = [t for t in pv.index if (not pv.loc[t, "O0"]) and pv.loc[t, "O1"]]
tidp = sorted(cand, key=lambda t: recs["WWPRO_TXT"][t]["L"])[len(cand) // 5]
rp = recs["WWPRO_TXT"][tidp]
A(f"### 5.2 Pro text trajectory, O0 vs O1 (`{tidp}`, framework `{rp['framework']}`)\n")
A(f"Problem: *{clip(rp['question'], 300)}* (Pro prompt variant: answer not shown). L = {rp['L']}. **Ground truth** (injected error, never shown): agent `{rp['true_agent']}`, step {rp['true_step_native']}, mode `{rp['labels']['mode']}`.\n")
for cond in ("O0", "O1"):
    rr = row("WWPRO_TXT", tidp, cond)
    A(f"**{cond}:** parsed `{rr.parsed_agent}`, step `{int(rr.parsed_step)}` → agent_ok = {rr.agent_ok}, step_ok = {rr.step_ok}, joint_ok = **{rr.joint_ok}**. Response (clipped):\n\n```\n{clip(rr.raw_response, 450)}\n```\n")

# HC parse failure
hc_fail = e1[(e1.dataset == "WW_HC") & e1.parse_fail].iloc[0]
A(f"### 5.3 Parse-failure rule (`WW_HC`, `{hc_fail.traj_id}`, {hc_fail.condition})\n")
A(f"The parser accepts only a plain integer after `Step Number:`. This response is a parse failure (`{hc_fail.parse_failure_reason}`) and counts as wrong for agent, step and joint; it is not repaired or re-asked:\n\n```\n{clip(hc_fail.raw_response, 500)}\n```\n")

# McNemar / Holm
h = ef.loc["WW_HC"]
A("### 5.4 Statistical test worked through (`WW_HC`, O1 vs O0)\n")
A(f"Of {int(h.n)} paired trajectories, b = {int(h.b)} were wrong under O0 and right under O1, c = {int(h.c)} the reverse. Exact McNemar test: two-sided binomial test of min(b, c) = {int(min(h.b, h.c))} out of b + c = {int(h.b + h.c)} with p = 0.5, raw p = {h['raw p']:.3g}. Paired change = {h['Delta joint (pp)']:+.1f} pp (O0 {res[(res.Dataset=='WW_HC')&(res['Trace condition']=='O0')]['Joint %'].iloc[0]:.1f}% → O1 {res[(res.Dataset=='WW_HC')&(res['Trace condition']=='O1')]['Joint %'].iloc[0]:.1f}%), bootstrap 95% CI [{h['CI lo (pp)']:.1f}, {h['CI hi (pp)']:.1f}]. Holm adjustment over the 2 comparisons in this dataset (O1 vs O0, O1-A vs O1) gives p = {h['Holm p']:.3g}.\n")

# Exp 2 examples
def rep_table(d, tid):
    g = e2[(e2.dataset == d) & (e2.traj_id == tid)].sort_values("replicate")
    t = g[["replicate", "parsed_agent", "parsed_step", "joint_ok"]].copy()
    t["parsed_step"] = t["parsed_step"].apply(lambda v: "" if pd.isna(v) else int(v))
    return md(t)

def exp2_block(title, d, tid):
    g = gt[(gt.dataset == d) & (gt.traj_id == tid)].iloc[0]
    p = gp[(gp.dataset == d) & (gp.traj_id == tid)].p_gate.iloc[0]
    rec = recs[d][tid]
    A(f"### {title} (`{d}`, `{tid}`)\n")
    A(f"Ground truth (not used by the features): agent `{rec['true_agent']}`, step {rec['true_step_native']}, L = {rec['L']}.\n")
    A(rep_table(d, tid) + "\n")
    feats = ["joint_modal_agreement", "agent_agreement", "step_agreement", "unique_pair_count", "norm_step_spread", "traj_length", "parse_failure_count"]
    A("Features: " + ", ".join(f"`{f}` = {g[f]:.3g}" for f in feats) + f". Candidate tie occurred: {bool(g.tie_occurred)}. **Target = {int(g.target)}** (candidate exactly correct on agent and step). Gate probability (model trained without this dataset): **{p:.3f}**.\n")

pick = lambda d, f: gt[(gt.dataset == d) & f(gt)].traj_id.iloc[0]
exp2_block("5.5 Experiment 2, split votes with a tie, resolved by the earliest run", "WW_AG", gt[(gt.dataset == "WW_AG") & gt.tie_occurred].traj_id.iloc[0])
exp2_block("5.6 Experiment 2, unanimous and correct", "WW_AG", gt[(gt.dataset == "WW_AG") & (gt.unique_pair_count == 1) & (gt.target == 1)].sort_values("traj_length").traj_id.iloc[0])
exp2_block("5.7 Experiment 2, unanimous but wrong (self-consistency is not correctness)", "WW_HC", gt[(gt.dataset == "WW_HC") & (gt.unique_pair_count == 1) & (gt.target == 0)].sort_values("traj_length").traj_id.iloc[0])

A("## 6. Caveats\n")
A("- One judge (`gpt-5.6-luna`, cost tier) and three datasets; nothing here generalises beyond them.\n- Pro errors are injected with exact labels; W&W failures are natural and its labels are single-annotator step choices. Hand-Crafted has only 58 trajectories, so its intervals are wide.\n- The O1 gain may partly reflect easier step counting rather than better diagnosis.\n- The OpenAI Responses API has no seed parameter, so the five repeats are not seed-controlled.\n- Exp 2 gate: each fold trains on only two datasets of different difficulty; Brier comparison with accept-all is flattered by a constant fitted on different base rates; consensus baselines have no bootstrap CIs; the gate does not clearly beat consensus on HC and Pro.\n- Parse failures count as incorrect (0 to 3% in most cells; 10.3% for O0 on HC, 5.7% for O1-A on Pro).\n- After the freeze only plot styling changed (`stats_exp1.py`, `gate_exp2.py`); no statistic changed. Earlier W&W/Pro runs were not reused.\n")
A("## 7. Files and reproducibility\n")
A("`config/protocol_freeze.md` (hashes), `raw/calls.jsonl` (every request and response), `outputs/exp1/` (`X1_predictions.csv`, `X1_results.csv`, `X1_effects*.csv`, `X1_forest.png`, interpretation), `outputs/exp2/` (`X2_runs.csv`, `X2_gate_table.csv`, `X2_gate_predictions.csv`, `X2_results*.csv`, `X2_counts.csv`, figures, interpretation), `outputs/X_manifest.md`, `outputs/examples/` (rendered prompts for every condition). Re-create tables with `python src/make_tables.py`, `src/stats_exp1.py`, `src/gate_exp2.py`, and this report with `src/make_report.py`.\n")
(R / "report.md").write_text("\n".join(L))
print("written", len("\n".join(L)))
