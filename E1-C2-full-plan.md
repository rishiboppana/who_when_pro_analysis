# Plan: Experiment 1, rebuilt (prompts A, B, C2) on the 184 original Who&When trajectories

Models: **gpt-5.6-luna (high reasoning), MiMo-V2.6-Pro, Gemma 4.**
This plan **supersedes `E1-gemma4-swap-plan.md`** where they conflict: the prompt files and render functions from the Oct 12 run are lost, so they are rebuilt here, and prompt C is replaced by a new prompt **C2**.

## 0. Read first: what is frozen, what is new, and what must be disclosed

- **A** = the original Who&When all-at-once prompt (verbatim text in section 3).
- **B** = A with the log rendered with `step - N :` headers. Nothing else changes.
- **C2** = a NEW evidence-first prompt (section 5). It differs from the earlier C in three ways requested by the user:
  1. no cap on the number of flagged steps (the earlier C allowed at most 5)
  2. the 17-mode taxonomy from the Who&When Pro `taxonomy.yaml` is shown to the model as a checklist for flagging errors
  3. the final answer is the **earliest decisive** step
- **Disclosure (must appear in any write-up):** C2 was written AFTER the original C had been run and its results seen. C2 is therefore a post-hoc revision, not a pre-registered prompt. The original C results (luna, MiMo, Kimi-K3) are kept in the archive and reported descriptively as "C v1". Do not present C2 as confirmatory without this note.
- Models were fixed before any C2 result; do not swap them after seeing results.
- Evaluation dataset: **the 184 original Who&When trajectories** (126 Algorithm-Generated + 58 Hand-Crafted; verify). This is NOT Who&When Pro. Pro is used here only as the source of the taxonomy file.

## 1. Frozen rules (apply to every phase)

1. Each prompt text is saved once to `prompts/` and never edited after the first API call. Record the SHA-256 of every prompt file, the taxonomy file, the parser, the scorer and the render functions in `E1_manifest.md`.
2. Placeholders are filled by **plain string replacement**, not `str.format` or f-strings (the texts contain braces). Replace `{problem}`, `{ground truth}`, `{failure log}` (and `{taxonomy}` in C2).
3. Send each prompt as **one user message, no system prompt**, for every model. If a provider requires a system message, stop and ask.
4. One fixed primary parser (section 7). A response that violates the schema is a **PARSE FAILURE**: never hand-fix, re-prompt, or retry because of it. Parse failures count as incorrect in primary accuracy and are also reported separately.
5. Run each (model, prompt, trajectory) **exactly once**. Only transport errors (HTTP 5xx, timeout, connection reset, rate limit) may be retried, up to 3 times with identical parameters.
6. Never read correctness during the smoke test: parseability, truncation, latency, cost and context fit only.
7. Never include `system_prompt`, `mistake_agent`, `mistake_step`, `mistake_reason`, `level` or `is_correct` in any prompt.
8. Do not strip, clean or re-render entry content.

## 2. Phase 0: Recover the data and the original rendering

1. Locate the 184 trajectories (JSON files under `Who&When/Algorithm-Generated` and `Who&When/Hand-Crafted`, from https://github.com/ag2ai/Agents_Failure_Attribution or the Hugging Face dataset `Kevin355/Who_and_When`). Inspect one record of each subset and write the real field names to `config/schema_notes.md` (expected: `question`, `ground_truth`, `history` entries with `content`, `role`, `name`, possibly `step`; labels `mistake_agent`, `mistake_step` as a STRING, `mistake_reason`). Verify the counts (126 + 58 = 184) and report any mismatch.
2. **Derive the original rendering.** In the repo, read `Automated_FA/Lib/utils.py`, function `all_at_once`: how the file is loaded, how each history entry is turned into text (which fields, in what order, which separator), and the exact prompt string. (Automated web fetching of GitHub is blocked for me, so use `git clone` or your own copy.) Copy that logic exactly into `render_A(history)`. Also read `evaluate.py` and replicate how agent and step matches are decided.
3. **Prove render_A is faithful.** Monkeypatch the original `all_at_once` client to capture the message it would send for 5 trajectories; assert `build_prompt_A(record)` is byte-identical to the captured prompt (after substituting the same placeholders). If they differ, stop and report.
4. **Index convention.** Confirm `mistake_step` is a 0-based index into `history`. Spot-check at least 10 records by reading the entry at that index and confirming it plausibly belongs to `mistake_agent`. Write the verified convention to `config/index_convention.md`. If ambiguous, stop and ask.
5. **Label sanity.** Report any record where `int(mistake_step) >= len(history)`. Known issue: Hand-Crafted trajectories HC_025, HC_039, HC_050 had label discrepancies between the Hugging Face rows and an earlier experiment file (the earlier audit found the experiment file's values were 24, 4 and 15 for those three, and found no effect on Experiment 1). If the experiment file is not available, score with the repo's labels, flag these three, and report headline results both with and without them. Do not edit any label.
6. **Check what survived from the Oct 12 run:** look for old raw predictions, manifests or logged prompts for luna, MiMo and Kimi-K3, and report what exists per (model, prompt). Archive Kimi-K3 (see section 10).

## 3. Prompts A and B

### Prompt A (`prompts/A.txt`), verbatim
Copied from the Oct 12 chat, where it was pasted from the original. **Diff it against the string in `Lib/utils.py` (`all_at_once`) and use the repo's version if they differ in whitespace or quotes**, keeping its quirks (curly quotes, spacing in the example).

```
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. The problem is: {problem}. The Answer for the problem is: {ground truth}. Identify which agent made an error, at which step, and explain the reason for the error. Here’s the conversation: {failure log} Based on this conversation, please predict the following: 1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert. 2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: { ”agent a”: ”xx”, ”agent b”: ”xxxx”, ”agent c”: ”xxxxx”, ”agent a”: ”xxxxxxx” }, each entry represents a ’step’ where an agent provides input. The ’x’ symbolizes the speech of each agent. If the mistake is in agent c’s speech, the step number is 2. If the second speech by ’agent a’ contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred. 3. The reason for your prediction. Please answer in the format: Agent Name: (Your prediction) Step Number: (Your prediction) Reason for Mistake: (Your reason)
```

`{failure log}` = `render_A(history)`.

### Prompt B (`prompts/B.txt`)
Byte-for-byte the same text as A. Only `{failure log}` changes: `render_B(history)`:

```
step - 0 :
<entry 0 text exactly as render_A renders it>

step - 1 :
<entry 1 text exactly as render_A renders it>
```

- Header is exactly `step - N :` (lowercase, spaces around the dash, a space before the colon), N from 0 in list order, a newline, the entry text, then a blank line before the next header.
- Implementation: `render_B = "\n\n".join(f"step - {i} :\n{render_entry(e)}" for i, e in enumerate(history))`, where `render_entry` is the per-entry formatting taken from the original code. If A joins entries with a different separator than `\n\n`, B keeps A's separator between the entry text and the next header, and the equivalence test normalizes only that separator.

Tests before any API call: (a) removing every `step - N :` header line from the B log must reproduce the A log, modulo the separator; (b) the full A and B prompts differ only by the inserted headers; (c) exactly N headers numbered 0..N-1 with no gaps; (d) hash entry texts before and after rendering to prove none were altered. Save one rendered example per prompt to `outputs/example_rendered_prompt_{A,B,C2}.txt`.

## 4. Taxonomy block for C2

- Source: `taxonomy.yaml` in the Hugging Face dataset repo `Leoxx/whowhen_pro` (license cc-by-4.0, attribute it in the write-up). Download once to `prompts/taxonomy.yaml`, record its SHA-256, and never re-download during the experiment.
- The file defines **17 modes** in file order: P.1 Visual misidentification; P.2 Spatial/grounding error; R.1 Hallucination; R.2 Reasoning error; R.3 Numerical/calculation error; R.4 Task misunderstanding; PL.1 Ineffective Planning; PL.2 Goal drift; A.1 Tool parameter/invocation error; A.2 Output format/syntax error; A.3 Premature termination; A.4 Repetitive/looping behavior; V.1 Context/memory loss; V.2 Inadequate or incorrect verification; C.1 Delegation/orchestration error; C.2 Communication failure; C.3 Over-reliance on other agents.
- Note: the Pro paper's Table 5 lists 18 modes (it has an extra "hallucinated tool or action"), and its code numbering differs from the yaml in the Action family. **Use the yaml's 17 codes and wording only.** Do not mix the two.
- Rendering (`render_taxonomy`): one entry per mode, in file order (no shuffling), each as one line `"{code} - {name}: {description}"`, where the description's whitespace and newlines are collapsed to single spaces and its text is otherwise unchanged. Entries are joined by a single newline. Unit-test that all 17 appear and that the descriptions match the yaml after whitespace collapse.
- Caveat to keep in the write-up: P.1 and P.2 are image modes and the 184 trajectories are text, and the taxonomy was designed for injected errors in the Pro benchmark. The prompt says some modes may not apply. The modes the model assigns are NOT scored (the original 184 have no mode labels); they are used only for secondary analysis.

## 5. Prompt C2 (`prompts/C2.txt`)

Placeholders: `{problem}`, `{ground truth}`, `{failure log}`, `{taxonomy}`. `{failure log}` = `render_A(history)` **(unlabeled, same as A; see the design note below)**.

```
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. The problem is: {problem}. The Answer for the problem is: {ground truth}. Identify which agent made an error, at which step, and explain the reason for the error. Here’s the conversation: {failure log}

Each entry in the conversation is a step in which an agent provides input. Steps are numbered from 0 in the order they appear: the first entry is step 0, the second is step 1, and so on. Count carefully.

Below is a taxonomy of 17 error modes. Use it as a checklist to help you notice errors in the conversation. An error may fit one mode, several modes, or none of them, and some modes may not apply to this conversation.

{taxonomy}

Analyze the conversation in the following four stages, in order. Be brief: one or two sentences per item.

Stage 1 - Final failure. State what final result or answer the conversation produced, which step states it, and how it differs from the correct answer given above.

Stage 2 - Flag erroneous steps. Go through the conversation from the first step to the last and flag every step in which an agent made any error, however many steps that is. Use the taxonomy as a checklist for what to look for. For each flagged step give: the step number, the agent, the taxonomy code or codes that best fit (write "none" if no mode fits), a short quote or close paraphrase of the questionable content, and why it is wrong, unsupported, or inadequate in light of the task and the correct answer. An error can be a false statement, faulty reasoning or code, an unjustified assumption, a misread instruction, a wrong tool use, or a necessary check or action that was skipped. Do not flag a step merely because it comes late or restates the outcome. If you find no clearly erroneous step, say so and flag the single most suspicious step.

Stage 3 - Originating or propagated. For each flagged step, decide whether it is ORIGINATING or PROPAGATED. ORIGINATING: the error first enters the conversation at this step and does not depend on an earlier wrong step. PROPAGATED: the step is wrong only because it builds on, repeats, confirms, or reports the consequences of an earlier error; name the earlier step it depends on. The last steps of a conversation, and steps that verify or announce the final answer, are often propagated or merely symptoms; judge a step by what caused its error, not by where the failure becomes visible.

Stage 4 - Earliest decisive step. Among the ORIGINATING flagged steps, choose the earliest decisive error: the earliest step such that correcting it would have led the conversation to the correct final answer. Skip earlier flagged steps whose correction would not have changed the final result. If no step is clearly decisive, choose the single most likely originating step anyway; you must give one answer.

Finish with exactly these three lines and nothing after them. Do not begin any earlier line with "Agent Name:", "Step Number:", or "Reason for Mistake:".
Agent Name: (the agent from your decision, written exactly as it appears in the conversation)
Step Number: (the single integer N of the chosen step, counting from 0)
Reason for Mistake: (the taxonomy code in parentheses if one fits, then one or two sentences on why this step is the origin of the failure)
```

**Design note for the user to confirm before running:** C2 uses the **unlabeled** `render_A` log, so that C2 vs A isolates the evidence-first + taxonomy structure (the approved design: B vs A isolates labels, C vs A isolates structure). The risk is that with an uncapped list of flagged steps the models must count entries to give step numbers, and B showed labels matter a lot for step accuracy. The alternative is to give C2 B's labeled log and change the closing step line to say "the number printed in the step label"; then C2 vs B isolates structure and C2 vs A is confounded with labels. Default in this plan: unlabeled. If the user prefers labeled, change only the `{failure log}` source and the sentence after the conversation, and note that the primary comparison becomes C2 vs B.

Output length: with no cap on flagged steps, long trajectories can produce long answers. Set the **same generous max output tokens for all three prompts** (reasoning tokens included for reasoning models), record it, and make truncation count as a parse failure. Measure the output length distribution in the smoke test.

## 6. Models and settings

| Slot | Model | Setting |
|---|---|---|
| 1 | `gpt-5.6-luna` | reasoning effort = high; OpenAI API |
| 2 | MiMo-V2.6-Pro | same provider, quantization and thinking setting as before (read from the old manifest; if lost, ask me) |
| 3 | Gemma 4 | variant, runtime and thinking mode to be given by the user |

For every model, before any call, write to `E1_manifest.md`: exact model ID or checkpoint, provider or runtime and version, quantization, thinking mode and budget (one setting, held constant across all prompts), temperature (0 where possible), top_p, seed, max output tokens, context window. Closed APIs are not fully deterministic: note that as a limitation. For local runs, record the hardware.

Context check: compute token counts for the longest trajectory under A, B and C2 (C2 adds the taxonomy, about 2k tokens, and long outputs). Each model must fit the longest prompt plus the output allowance with no truncation. If not, stop and report.

## 7. Parser and scorer (written and unit-tested BEFORE any API call, then frozen)

**Primary parser (identical for A, B, C2):**
- Search the response for lines starting with `Agent Name:`, `Step Number:`, `Reason for Mistake:` (line start, exact label, case-sensitive). Use the **last** occurrence of each (C2's earlier stages may mention the labels).
- All three must be present. `Step Number` must be a single integer after trimming whitespace (no markdown, ranges or text). `Agent Name` must be non-empty. Otherwise PARSE FAILURE.
- No fixing of markdown (`**`, backticks) or of agent-name spelling.

**Scorer (replicate the original `evaluate.py` semantics; document them):**
- Agent correct: parsed agent equals `mistake_agent` (exact, case-sensitive, after trimming surrounding whitespace only).
- Step correct: parsed step equals `int(mistake_step)`.
- Joint: both correct. Parse failure = incorrect on all three.
- Normalized error: `|pred_step - true_step| / len(history)`, over parsed responses only.

**Secondary parser (C2 only, separate file, tolerant):** from Stage 2 extract the set of flagged step numbers and their taxonomy codes (regex on `step N` plus codes like `R.2`, `PL.1`). Used only for secondary analyses (section 9). Never used for primary accuracy.

Unit tests: valid responses, responses with extra text before the final lines, missing fields, a non-integer step, duplicate label lines (last one wins), and responses whose Stage 2 mentions the label strings.

## 8. Run matrix and order

| Model | A | B | C2 |
|---|---|---|---|
| luna | reuse if old predictions AND matching prompts are recoverable (byte-compare), else run | same | **run** |
| MiMo | same rule | same | **run** |
| Gemma 4 | **run** | **run** | **run** |

Reuse is allowed only if the old raw responses exist and the old prompts can be shown to equal the rebuilt A and B. Otherwise re-run. C2 is run for all three models because it is new.

Order:
1. Phase 0 (section 2). **Stop and show me `schema_notes.md`, `index_convention.md` and the render faithfulness result before continuing.**
2. Build prompts, render functions, parser, scorer; run all tests; freeze and hash.
3. Smoke test: A, B and C2, all three models, on 12 trajectories (the 6 longest plus 6 chosen by a fixed seed). Record parseability, truncation, output-token distribution, latency and cost. Pass rule: at most 1 parse failure in 12 per (model, prompt). If a prompt fails, stop and ask me; do not edit prompts, the parser, the model or its settings.
4. Cost estimate (input tokens × 184 per prompt plus expected output) per model. Ask me to confirm if it exceeds the spend cap I give you.
5. Full run (resumable, spend cap enforced, raw responses logged). Stop and report if a (model, prompt) has more than 25% parse failures after the first 40 trajectories.
6. Score, analyse, write deliverables.

## 9. Statistics

- Wilson 95% CIs for Agent, Exact-step and Joint accuracy, per (model, prompt).
- **Primary comparison: joint accuracy, B vs A and C2 vs A within each model (6 comparisons).** For each: paired change in percentage points with a paired-bootstrap 95% CI (resample trajectories, 10,000 resamples, fixed seed, recorded), McNemar's test (exact binomial on discordant pairs), Holm correction across the 6 p-values.
- Everything else (agent, step, parse rate, normalized error) is secondary.
- No significance tests between models (different families and scales); descriptive only.
- **Extra, labeled exploratory (not Holm-corrected):** C2 vs B within each model; C v1 vs A for luna and MiMo from the archived run (Kimi-K3 shown separately).
- **C2 secondary analyses:**
  - number of steps flagged per trajectory (distribution by model, by subset AG/HC)
  - recall of the true step within the flagged set vs exact-step accuracy of the final answer (does the model flag the right step but pick the wrong one?)
  - how often the final answer is the earliest flagged step vs a later one
  - distribution of taxonomy codes assigned at the chosen step (descriptive; no ground truth)
  - AG vs HC breakdown for all three prompts
  - output length vs trajectory length

## 10. Archive and deviations

- Move Kimi-K3 results to `E1_archive/kimi_k3/` with a README: "replaced by Gemma 4 after pre-registration, by user decision; excluded from the main table".
- Move the original C (v1) results to `E1_archive/C_v1/` with its prompt text if recoverable; if the prompt text is not recoverable, say so.
- `E1_manifest.md` must list every deviation: the Kimi-K3 to Gemma 4 swap, C replaced by C2 after seeing C results, rebuilt (not original) prompt files and render functions, and the label-source decision.

## 11. Deliverables

- `E1_predictions.csv`: one row per (model, prompt, trajectory): trajectory id, subset, raw response path, parse_ok, parsed agent, parsed step, agent_correct, step_correct, joint_correct, normalized_error, output tokens, truncated flag.
- `E1_prompts.txt`: the exact A, B, C2 texts, the rendered taxonomy block, and a rendered example of each, with SHA-256s.
- `E1_summary.csv`: Model, Prompt, n, Parse fail %, Agent acc %, Exact-step acc %, Joint acc % (each with the Wilson CI), median normalized error, plus the paired rows (change in pp, bootstrap CI, raw p, Holm p).
- `E1_c2_flagging.csv`: per trajectory and model, the flagged steps and codes from the secondary parser.
- One figure: Agent / Step / Joint accuracy with 95% CIs, 3 models × 3 prompts (A, B, C2).
- `E1_manifest.md` and `config/` notes as above.
