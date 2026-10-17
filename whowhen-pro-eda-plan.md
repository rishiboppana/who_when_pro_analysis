# Who&When Pro: Error Analysis (EDA) Plan

**Dataset:** Who&When Pro (Liu et al., arXiv 2607.09996). Project page: https://whowhenpro.github.io
**Goal:** Understand what kinds of errors the benchmark contains, where they sit in traces, which tasks/frameworks/modalities they come from, and which are easy vs hard to catch.

## Ground rules (read first)

- The traces are **synthetic**: a successful trajectory was replayed up to step t, an LLM-steered bad action was substituted at t, and the run was kept only if it then failed. Labels (agent, step, mode) are exact by construction.
- Mode frequencies reflect the **sampling design and the failure filter**, not how often real agents make each error. Every distribution plot should carry this caveat in its caption/notes.
- Do not guess the data schema. Inspect it first (Step 0) and write down the actual field names before analysis.
- Save every intermediate table as CSV/parquet in `outputs/tables/` and every figure as PNG in `outputs/figures/`. Keep one script or notebook per section so each is re-runnable.
- Report counts alongside percentages. Flag any cell with fewer than 30 traces as low-support.

## Step 0: Data acquisition and schema discovery

1. Find the released traces and labels via the project page / paper links (GitHub or Hugging Face). Download them.
2. Inspect the files. Write `outputs/schema_notes.md` documenting: file format, one example trace in full, and the actual field names for each of the following (mark "not present" if missing):
   - trace id, task/question, gold answer
   - source benchmark, task category (the paper has 9), modality (text/image/video)
   - agent framework, single- vs multi-agent topology, backbone model
   - ordered steps (step index, agent name/role, content, tool calls, observations, images)
   - labels: error agent, error step, error mode (18-mode taxonomy), any family field
   - injection metadata (injection position, seed trace id, replay info) if released
3. Build one normalized table `traces.parquet` (one row per trace) and `steps.parquet` (one row per step), with a `is_decisive_step` flag. Add derived columns: `n_steps`, `n_words`, `injection_pos_rel` (= error step / n_steps), `steps_after_injection`, `family` (Perception, Reasoning, Planning, Action, Verification, Coordination).
4. Sanity checks: total traces should be about 12,326; check the counts against the paper (26 benchmarks, 9 categories, 15 frameworks, 3 modalities, 18 modes, avg trace length 7.5 steps). Report any mismatch rather than silently fixing it.

### Taxonomy reference (18 modes)

P.1 Visual misidentification, P.2 Grounding error, R.1 Hallucination, R.2 Reasoning error, R.3 Calculation error, R.4 Task misunderstanding, PL.1 Ineffective planning, PL.2 Goal drift, A.1 Tool parameter error, A.2 Hallucinated tool/action, A.3 Output format error, A.4 Premature termination, A.5 Looping behavior, V.1 Context loss, V.2 Inadequate verification, C.1 Delegation/orchestration error, C.2 Communication failure, C.3 Over-reliance on other agents.

## Part A: Distribution analyses (no LLM needed)

**A1. Mode and family distribution.** Counts and share per mode and per family. Bar charts. Check how far from uniform it is.

**A2. Injection position.** Histogram of `injection_pos_rel` overall (paper reports 64% in the 40-80% range; verify). Then by mode and by family (box/violin plots). Test whether planning errors are early and verification errors late, as the paper says.

**A3. Absolute step index.** Distribution of the error step index and of `n_steps`. How often is the error at step 1, 2, 3? How many traces are very short, where "localization" is trivial?

**A4. Task category x mode.** Heatmap of row-normalized shares (mode given task category) and a count heatmap. Note empty cells and explain them by applicability rules (e.g. perception modes only on image/video).

**A5. Source benchmark x mode.** Same as A4 at the 26-benchmark level; show only the top N rows/columns if too wide.

**A6. Framework x mode.** Heatmap by agent framework (15). Add the framework's trace share so it's clear when a pattern is just a big framework.

**A7. Modality x mode.** Text/image/video by mode and by family. Also modality x single/multi-agent.

**A8. Topology (single vs multi-agent) x mode.** Confirm coordination modes (C.1-C.3) appear only in multi-agent traces. For multi-agent traces, show which agent roles receive the injected error (e.g. orchestrator/planner vs executor/coder/reviewer) and the mode mix per role.

**A9. Backbone model x mode.** If the metadata records the backbone (GPT-4.1 vs Gemini 3 Flash, etc.), show mode and trace-length differences between them.

**A10. Confounding check.** Many of the above are correlated (framework determines modality and topology). Fit a simple multinomial/logistic model or run stratified tables to see which factor still matters after controlling for the others. Report only what the data supports.

## Part B: Detectability analysis ("which errors get caught, which don't")

"Caught" needs definitions. Use all of these, reported separately:
- **Agent hit:** predicted agent equals the label (multi-agent traces only).
- **Step hit:** predicted step equals the label exactly (Step@0); also Step@1 and Step@2.
- **Mode hit:** predicted mode equals the label; per-mode precision/recall/F1.
- **Joint hit:** agent, step and mode all correct.

**B1. Run attribution on a stratified subset.**
- Take a stratified sample of about 1,500 traces (preserve modality and mode distribution; also keep a second sample forcing at least 30 traces for every mode).
- Use the paper's **all-at-once** protocol and prompt (Appendix J.1 of the paper: given the question and full transcript, output Agent Name, Step Number, Error Mode, Reason; closed-book, i.e. no gold answer). Recreate the prompt from the paper.
- Run at least two models: one API frontier model and one small local/open model I can run (e.g. Qwen3-8B). Estimate cost before launching and ask me to confirm the budget if over about $50.
- Log raw outputs, parse failures, token counts and cost. Use structured/regex parsing and report the parse-failure rate.
- Cache results per (trace, model, protocol) so reruns don't repeat calls.

**B2. Detectability breakdowns.** Hit rates (agent/step/mode/joint) by: mode, family, modality, task category, framework, topology, injection position bin (early/middle/late), trace length bin (by tokens), and error-agent role.

**B3. Confusion matrices.**
- Mode confusion, 18x18 and family-level 6x6, row-normalized. The paper claims planning/verification/coordination errors are often labeled as R.2 (Reasoning error); check whether that holds in my runs.
- Step prediction distance (predicted minus true step). Is it early-biased or late-biased? How much mass is at exactly +-1?

**B4. Cheap detectability proxies (no LLM).** For the decisive step, compute length (words), whether it has a tool call, whether it is the last step, number of steps after it, lexical overlap with the task, and similarity to the previous step. Check whether these predict hit/miss (logistic regression or a small tree). This tells us if misses are explained by structure rather than content.

**B5. Easy vs hard error profile.** For the top 10% easiest and bottom 10% hardest traces (by hit rate across the models), compare mode, modality, length, framework, and position. Read about 20 examples from each group by hand and write short notes on patterns.

**B6. Symptom-vs-cause analysis.** For misses where the predicted step is after the true step, measure the gap and check whether the predicted step is the step where the error first becomes visible (e.g. a wrong tool result or a wrong final answer). This tests the paper's claim that models flag symptoms rather than root causes.

## Part C: Further EDA questions worth examining

1. **Is the injection detectable by style?** Train a simple classifier (TF-IDF + logistic regression) to distinguish the injected step from other steps in the same trace, using only text features. High accuracy would suggest injection artifacts that a model could exploit without truly diagnosing the error.
2. **Cascade length.** How many steps follow the injection, and does the number of post-injection steps relate to hit rate? Does the agent ever self-correct partially (as in the paper's Gentle Monster example) and still fail?
3. **Multiple valid decisive steps.** Find traces where a different step could arguably be the decisive error (e.g. a later step repeats or compounds the mistake). Manually review about 30 high-miss traces and estimate how much label ambiguity exists.
4. **Duplicate seeds.** How many injected traces come from the same seed trajectory or the same task? If many, accuracy estimates are correlated; compute confidence intervals with a cluster bootstrap by seed/task.
5. **Mode applicability audit.** For each mode, list the (framework, benchmark) pairs where it appears. Are there modes that only ever occur in one framework (so "mode" and "framework" are not separable)?
6. **Trace length and tokens.** Hit rate against token length by model (the paper reports a drop from about 94% on traces under 3K tokens to about 50% above 12K). Check whether this reproduces, and whether it is explained by position and number of correct steps around the error.
7. **Gold answer effect.** On a smaller subset, rerun with the gold answer in the prompt and compare mode accuracy for perception vs reasoning errors (the paper reports opposite effects).
8. **Agent-level difficulty in multi-agent traces.** Is the error agent harder to identify when there are more agents, or when the injected agent is a non-orchestrator?
9. **Natural vs injected comparison.** If the original Who&When (184 natural traces, https://huggingface.co/datasets/Kevin355/Who_and_When) can be loaded, run the same method and compare the hit rates and the position/length distributions. This addresses whether results on injected errors transfer to real failures.
10. **Windowing sensitivity (for my own pipeline).** Evaluate how a windowed or per-step method (instead of full-trace all-at-once) changes step/mode hit rates, split by injection position and trace length. This connects the analysis to my Stage 1 setup.

## Deliverables

1. `outputs/schema_notes.md` with the actual schema and anything that differs from the paper.
2. `outputs/tables/` and `outputs/figures/` for each section (name files `A1_...`, `B3_...`, etc.).
3. A short `outputs/findings.md` summarizing, per section, 3-5 findings with the numbers behind them, the sample sizes, and any caveats. Separate "observed in data" from "interpretation". Do not claim a pattern that has fewer than 30 supporting traces.
4. A list of open questions or anything that could not be done (missing metadata, cost limits, parse failures).

## Order of work

1. Step 0 (data and schema). Stop and show me `schema_notes.md` before continuing if any field from the list above is missing.
2. Part A (all of it). Show me A1-A4 first.
3. Part B, starting with the subset selection and a 50-trace pilot run to check parsing and cost before the full run.
4. Part C items, prioritizing 1, 4, 9 and 10.
