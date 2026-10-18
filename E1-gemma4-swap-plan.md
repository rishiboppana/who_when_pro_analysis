# Plan: Experiment 1 (3 models x 3 prompts) with Gemma 4 replacing Kimi-K3

## What changed and why
The slate frozen on Oct 12 was gpt-5.6-luna, MiMo-V2.6-Pro, Kimi-K3. The third model is now **Gemma 4**, chosen by the user. This is a deliberate deviation from the pre-registered slate, made before any Gemma 4 result was seen. State it in the write-up. Kimi-K3 results are archived, not deleted, and are excluded from the main E1 table.

Everything else is unchanged: 184 original Who&When trajectories (126 Algorithm-Generated + 58 Hand-Crafted; verify), frozen prompts A/B/C, frozen parser, frozen scorer, one run per (model, trajectory, prompt).

## Frozen rules (same as before; do not change)
1. Use the existing frozen files: `prompts/A.txt`, `prompts/B.txt`, `prompts/C.txt`, `render_A`, `render_B`, the parser and the scorer from the Oct 12 run. **Do not retype or regenerate any of them.** If any is missing from this workspace, stop and ask me for it. Do not reconstruct from memory.
2. Print and record the SHA-256 of each prompt file, the parser and the scorer. They must match the values logged in the original run's manifest. If they don't, stop and report.
3. A response that violates the required schema is a PARSE FAILURE. Never hand-fix, re-prompt, or retry because of a parse failure. Only transport errors (HTTP 5xx, timeout, connection reset, rate limit) may be retried, up to 3 times with identical parameters.
4. Parse failures count as INCORRECT in primary accuracy. Also report the parse-failure rate separately.
5. Same trajectory text inside each prompt condition (A uses `render_A`; B uses `render_B`; C uses `render_A` plus the C instruction, exactly as in the existing C plan).
6. Score against the same label source as the earlier run (the experiment file). Do not edit any label. (Context: HC_025, HC_039, HC_050 differ between the Hugging Face rows and the experiment file; the earlier audit showed no effect on E1.)

## Models
| Slot | Model | Status |
|---|---|---|
| 1 | gpt-5.6-luna, reasoning effort = high | **Reuse** existing predictions for A, B, C. Do not re-run. |
| 2 | MiMo-V2.6-Pro (same provider, quantization and thinking setting as before) | **Reuse** existing predictions for A, B, C. Do not re-run. |
| 3 | **Gemma 4** | **Run** B and C. Run A too if Gemma 4 A predictions do not already exist (check first). |

Before any Gemma 4 call, write these into `E1_manifest.md`. Ask me for any that I haven't given you:
- exact checkpoint/ID (size variant, instruct/IT version), quantization, runtime and version (vLLM / llama.cpp / MLX / hosted provider and its exact model string)
- thinking/reasoning mode: on or off, and the budget or effort level. Pin one setting and use it for all three prompts. If Gemma 4 has no reasoning mode, record that
- temperature (0 if possible), top_p, max output tokens, seed
- context window, and the measured token count of the longest trajectory under prompt C (with B's extra labels). If it doesn't fit without truncation, stop and tell me. Truncation is a hidden confound, not a parse failure.
- if run locally: hardware and memory used

Do not use different sampling parameters or a different max-output limit across prompts for Gemma 4. If C's staged answer needs more output tokens than A/B, set the same generous limit for all three and record it.

## Steps
1. **Inventory.** Find the existing E1 artifacts (predictions, manifest, parser, scorer, prompts). List what exists per (model, prompt). Confirm luna and MiMo have all three prompts on all 184 trajectories. Report any gaps before doing anything else.
2. **Hash check.** See rule 2.
3. **Archive Kimi-K3.** Move its predictions to `E1_archive/kimi_k3/` with a README saying: "pre-registered slate member, replaced by Gemma 4 on 2026-10-18 by user decision; not part of the main E1 table." Do not delete.
4. **Gemma 4 smoke test.** Run prompt A, B and C on 12 trajectories (the 6 longest plus 6 chosen by a fixed seed). Record only parseability, truncation, latency and cost. **Do not look at correctness.** Pass rule: at most 1 parse failure in 12 per prompt. If a prompt fails the rule, stop and report. Do not edit prompts or the parser, and do not switch the model or its settings. Ask me.
5. **Cost estimate.** Compute input tokens (trajectory count x tokens per prompt) plus expected output tokens for Gemma 4. Ask me to confirm before the full run if the estimate exceeds the spend cap I give you.
6. **Full Gemma 4 run.** All 184 trajectories for each prompt it still needs (B, C, and A if missing). Resumable, spend cap enforced, log every raw response. Stop and tell me if a prompt's parse-failure rate exceeds 25% after the first 40 trajectories.
7. **Parse and score** with the frozen parser and scorer. Produce one row per (model, prompt, trajectory).

## Metrics
- **Agent acc**, **Exact-step acc**, **Joint acc** (agent and step both correct). Parse failures count as incorrect.
- **Parse fail %.**
- **Median normalized error:** median of |predicted step - true step| / number of steps in the trajectory, over parsed responses only.

## Statistics
- Wilson 95% CIs for Agent, Exact-step and Joint accuracy for each of the 9 configurations.
- **Primary comparison is joint accuracy:** B vs A and C vs A within each model (6 comparisons).
  - paired change in percentage points with a paired-bootstrap 95% CI (resample trajectories, 10,000 resamples, fixed seed; record it)
  - McNemar's test on the paired joint-correct outcomes (use the exact binomial version on discordant pairs; state which one you used)
  - Holm-correct the 6 p-values
- All other metrics (agent, step, normalized error, parse rate) are secondary. Label them as such and do not Holm-correct them.
- Do not compare models to each other with a significance test. Models differ in family and scale, so model differences are descriptive here.

## Deliverables
- `E1_predictions.csv`: one row per (model, prompt, trajectory): trajectory id, subset (AG/HC), raw response path, parsed agent, parsed step, parse_ok, agent_correct, step_correct, joint_correct, normalized_error.
- `E1_prompts.txt`: the exact A, B, C texts with their SHA-256.
- `E1_summary.csv`: table with columns Model, Prompt, n, Parse fail %, Agent acc %, Exact-step acc %, Joint acc % (each with the Wilson CI), Median normalized error, plus the paired B-A and C-A rows with the change, bootstrap CI, raw p and Holm p.
- One figure: Agent / Step / Joint accuracy with 95% CIs, 3 models x 3 prompts.
- `E1_manifest.md`: model IDs and settings, hashes, seeds, dates, the Kimi to Gemma 4 deviation note, and anything unexpected.

## Reporting notes
- State the slate deviation (Kimi-K3 replaced by Gemma 4 after the pre-registration) in the first paragraph of any write-up.
- Luna and MiMo numbers are from the earlier run. Say so, and give the run date.
- Gemma 4 is likely a different scale from luna and MiMo, and "high reasoning" is not the same setting across providers. Say that these are not like-for-like model comparisons.
- Report the order in which things were done (smoke test, then full run) so it's clear nothing was tuned on results.
