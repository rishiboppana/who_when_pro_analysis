# E1 manifest (Experiment 1, Who&When Pro multi-agent text traces; prompts B and C)
Created 2026-10-18T19:40:30Z. Frozen BEFORE any model call.

## Scope / design decisions (user, 2026-10-18)
- Data: Who&When Pro **text** split restricted to the 2,670 multi-agent traces (have agent + step labels): mathchat 920, magentic-one 811, debate 445, macnet 261, dylan 152, metagpt 81. Single-agent traces (smolagents, alfagent) excluded (no agent label). Image/GUI/video splits not used.
- Prompts: **B** = plan's A text + flattened log with `step - N :` headers (0-indexed). **C** = plan's evidence-first prompt (the plan's "C2", renamed C), unlabeled log (plan default), taxonomy block from the dataset's taxonomy.yaml (17 modes). Prompt A is NOT run (user decision). Step labels appear only in B.
- Scoring: exact match (agent case-sensitive; step exact flat index). debate/dylan labels are round-level, so any turn inside the labelled round counts as correct (official scorer semantics). Parse failures = incorrect.
- Flattening: entry text taken from the official whowhen_eval per-framework renderers (turn text with their step header removed), one entry per turn; entry 0 = first turn. Label mapping: mathchat 2*round+position, metagpt stage, macnet/magentic by R.P coordinate, debate/dylan round (official score.py). 0 mismatches between the labelled agent and the agent at the labelled flat index.
- 129 traces (livecodebench_pro: macnet 73, metagpt 56) have no gold answer: the sentence "The Answer for the problem is: ..." is dropped for them in both prompts.
- Models: gpt-5.6-luna (reasoning high), XiaomiMiMo/MiMo-V2.6-Pro-RL:novita (same as Oct 12 config). Settings in config/models.yaml; single user message, no system prompt, no temperature/top_p/seed sent.
- Not reused: Oct 12 raw results are for the 184 original trajectories (different dataset); nothing reused.

## Hashes (SHA-256)
```
ba3cb5646f5f8efbcb1661eddbd050de1ca759bd6ebcf83c638893aeb7128791  prompts/B.txt
8cf25c968aab6a0de241e225fe1283b326b72b26a7c9bc00fdfba286f3bd4879  prompts/C.txt
f5d77edee84622e025519c31404a5b741ff4e008171392e35df0cceebd41d140  prompts/taxonomy.yaml
d9b345165ee1985b1cc667663bd6d542cf18145d3ee3186a83bb46d55c2a6caa  src/parser.py
e7cc7dee0189b37d79957a5a2412148b9af6191720ca4d7235944f99ebceffd6  src/score.py
c4c448e725b191e283b8e4acdffed10a3ece393b1c7bf15db2c61a64f9e4fa98  src/render.py
923dc31ddfc96f5ed96108b32a5a693fc06d989315eced0f60035ef02d8de9c4  src/data.py
d8755201caeac17b2121c24cc18d7a23b6eac573fcb40581ee83f27d8365969f  src/run.py
fbe8dca57ac8f7ad1001fe023db5e2c8e5970fbf354ad741f1bf877349e6c124  config/models.yaml
```
Source of prompts: extracted programmatically from E1-C2-full-plan.md (no retyping). Taxonomy: data/taxonomy.yaml from HF Leoxx/whowhen_pro (cc-by-4.0), f5d77ede....
Unit tests: 11 passed (tests/test_all.py).

## Deviations / disclosures
- Original-184 plan replaced by Pro multi-agent subset; the plan's Phase 0 faithfulness test against Lib/utils.py does not apply.
- C (plan's C2) was written after the earlier C results were seen: post-hoc, not pre-registered.
- "High reasoning" is not the same setting across providers; closed APIs are not fully deterministic.

## Update 2026-10-18T20:07:38Z: smoke test done for luna and mimo
- Smoke (12 ids: 6 longest + 6 seed-0 random; config/smoke_ids.txt), correctness NOT inspected. luna B 0/12 failures, C 0/12; mimo B 1/12 (missing_agent), C 0/12; no truncation, no API errors. Pass rule (<=1/12) met.
- run.py edit after smoke: a row with an API/transport error is no longer counted as done on resume (so it is retried). No prompt/parser/scorer change.
- User instruction: run the full luna and mimo calls now (all 2,670 traces, B and C).
Current hashes:
```
ba3cb5646f5f8efbcb1661eddbd050de1ca759bd6ebcf83c638893aeb7128791  prompts/B.txt
8cf25c968aab6a0de241e225fe1283b326b72b26a7c9bc00fdfba286f3bd4879  prompts/C.txt
f5d77edee84622e025519c31404a5b741ff4e008171392e35df0cceebd41d140  prompts/taxonomy.yaml
d9b345165ee1985b1cc667663bd6d542cf18145d3ee3186a83bb46d55c2a6caa  src/parser.py
e7cc7dee0189b37d79957a5a2412148b9af6191720ca4d7235944f99ebceffd6  src/score.py
c4c448e725b191e283b8e4acdffed10a3ece393b1c7bf15db2c61a64f9e4fa98  src/render.py
923dc31ddfc96f5ed96108b32a5a693fc06d989315eced0f60035ef02d8de9c4  src/data.py
6a1e4b89f13efbce8118004f9de7b1597f341b1ea3e9cff5d982ec1f8669b760  src/run.py
75e07fda6bdffcc4394db3bc79b1265646222d9197477fae0a3ebcadd9b069a5  config/models.yaml
e72408541396b03c1397e5d5e759d88f527a64ed125bf5bc4f786d358f24de56  config/pricing.yaml
```

## Update 2026-10-20: Gemma 4 removed
- Gemma 4 was attempted and dropped by user decision. History, for provenance only: OpenRouter free tier returned 429 and the paid tier 402 (no credits); a local vLLM server on Google Colab was tried with the 31B and then the 12B model; the 12B smoke test failed the pass rule on truncation at the 16,384-token output cap (prompt C). No Gemma result is part of any analysis in this repository.
- All Gemma code, config entries, smoke outputs and the Colab notebook were deleted. Retained, not Gemma-specific: src/data.py reads the renderer repo path from env WHOWHEN_EVAL_REPO (pinned clone of ag2ai/whowhen_pro @ 14369dc), and src/run.py no longer counts API/transport-error rows as done on resume.

## Update 2026-10-20T19:03:09Z: third model = Qwen3.5-9B, 4-bit, on Google Colab (planned; not yet run)
- User decision: Qwen3.5-9B, 4-bit quantized, thinking on, on a larger Colab GPU. Config slot `qwen` added to config/models.yaml (and pricing.yaml); new src/run_local.py sends the sampling fields to a local vLLM server; analysis scripts list the model.
- Checkpoint chosen: RedHatAI/Qwen3.5-9B-quantized.w4a16 (GPTQ INT4 weights, group size 128, BF16 activations; Apache-2.0; linear-attention layers, embeddings and vision encoder stay BF16). Alternatives in the notebook: QuantTrio/Qwen3.5-9B-AWQ, cyankiwi/Qwen3.5-9B-AWQ-4bit. This is the quantized model, not the BF16 original.
- Settings (from the Qwen3.5 model card, thinking mode, general tasks): temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, seed 0; thinking on (enable_thinking=true, reasoning parser qwen3); max output 32,768 tokens including thinking (same cap as luna); max_model_len 57,344. Defaults chosen by the assistant, to be confirmed by the user.
- Verified before building: vLLM 0.32.0 registers Qwen3_5ForConditionalGeneration; transformers 5.19.0 contains models/qwen3_5; the pinned versions resolve together for Linux x86_64 / Python 3.12.
- Notebook: colab/whowhen_qwen35_colab.ipynb (self-contained). Tested locally only (cell logic against a fake server); NOT yet run on a GPU.

## Update 2026-10-20T20:21:15Z: Qwen run speed; fp8 KV cache enabled (user decision)
- Observed on an A100 40 GB: the vLLM KV cache was full (667,184 tokens, 97-99.8% used), only about 45 of 64 requests ran, about 1,100 output tokens/s; Qwen3.5 thinking is long (about 8.8k tokens per call on prompt B, 11.6k on C in the first completed calls).
- Decision: enable `--kv-cache-dtype fp8` (vLLM 0.32.0; default, uncalibrated scales of 1.0; applies only to the 8 full-attention layers of this hybrid model) to roughly double KV capacity. This is a second approximation on top of the 4-bit weights and must be disclosed. The notebook falls back to the default KV dtype if the server cannot start with fp8, and records the dtype actually used. Results produced before and after this switch (if the run was resumed) used different KV dtypes and should be flagged by call timestamp.
- First completed Qwen calls (strict parser, not yet final): parse failures were truncation at the 32,768-token cap and format drift (single-line answers, `\boxed{}`, different labels); counted as incorrect per the plan.

## Update 2026-10-22T07:55:29Z: third model = Qwen3.5-122B-A10B via the Hugging Face router (replaces the local 9B Colab plan)
- User decision: use an existing hosted model via the HF token. Config slot `qwen122`: model Qwen/Qwen3.5-122B-A10B:novita (MoE, 122B total / 10B active, Apache-2.0, 262k context), HF router OpenAI-compatible endpoint, thinking on by default (thoughts returned as reasoning_content), no sampling parameters sent (provider defaults, as for luna and MiMo), max output 32,768 tokens including thinking (luna: 32,768; MiMo: 131,072). Prices (HF router listing, novita): $0.40 / $3.20 per 1M input / output tokens (deepinfra lists $0.29 / $2.40).
- Runner: unchanged src/run.py (single user message, no system prompt, streaming). src/data.py now finds the pinned renderer clone in vendor/whowhen_pro (ag2ai/whowhen_pro @ 14369dc) by default.
- Smoke test (12 ids: 6 longest + 6 seed-0 random, correctness NOT inspected): both prompts 12/12 completed, 0 API errors, 0 parse failures, 0 truncation. Output tokens: B mean 5,582 (max 10,215), C mean 7,804 (max 14,854). One C response had the provider leave part of the thinking in the answer text before a </think> tag; the final three lines were intact and the last-occurrence parser rule read them correctly. Smoke cost $0.28 (B) + $0.37 (C).
- Projected full-run cost (2,670 traces x B and C), if the full-run mean output is 0.5 / 0.75 / 1.0 x the smoke mean: about $66 / $95 / $124. Awaiting user confirmation of the spend before the full run.
- Local Qwen3.5-9B Colab files (config slot `qwen`, colab/ notebook) are kept but are not part of this plan.

## 2026-10-22T07:57:23Z: Qwen3.5-122B-A10B full run launched (user: "Start Qwen full run")
- MiMo stopped by the user at B 2,021 and C 1,259 of 2,670 (partial; not resumed). Full Qwen run: prompts B and C as separate processes, 16 workers each, spend guard $60 per prompt (about $120 total; the guard is per process, not across restarts).
Hashes at launch:
```
ba3cb5646f5f8efbcb1661eddbd050de1ca759bd6ebcf83c638893aeb7128791  prompts/B.txt
8cf25c968aab6a0de241e225fe1283b326b72b26a7c9bc00fdfba286f3bd4879  prompts/C.txt
f5d77edee84622e025519c31404a5b741ff4e008171392e35df0cceebd41d140  prompts/taxonomy.yaml
d9b345165ee1985b1cc667663bd6d542cf18145d3ee3186a83bb46d55c2a6caa  src/parser.py
e7cc7dee0189b37d79957a5a2412148b9af6191720ca4d7235944f99ebceffd6  src/score.py
c4c448e725b191e283b8e4acdffed10a3ece393b1c7bf15db2c61a64f9e4fa98  src/render.py
506ab4254d2768ba663019b0b8489421295803ac2811d36087a757190ee0a05f  src/data.py
6ed35246e34b98bd2d979cb41df2b0d7139b91ba49687381d43489980cd6b3c9  src/run.py
d745e8d3437686d160550d572ef658c889ac41a772c1712a38474d6f27c7f5ea  config/models.yaml
9b4ef5c86a2e468954051f74a3445a1a1d6b8c0cad46216ede065ae376f3c1b2  config/pricing.yaml
```
