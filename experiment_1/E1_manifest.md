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

## 2026-10-22T08:06:09Z: Qwen3.5-122B run resumed on the :preferred route (provider routing change)
- Novita (pinned :novita) tripped its circuit breaker under 16 workers per prompt (503), which stopped the first launch after 42 good calls. The :deepinfra route was rejected: for a request to Qwen3.5-122B-A10B it answers and BILLS as Qwen3.5-397B-A17B (response model field and estimated_cost match the 397B price exactly), so it is not the 122B.
- Resumed with model_id Qwen/Qwen3.5-122B-A10B:preferred (8/8 probe calls were served by novita, response model qwen/qwen3.5-122b-a10b), 8 workers per prompt, spend guard $60 per prompt. New runner guard (config expect_model): a response whose reported model is not qwen/qwen3.5-122b-a10b is logged as an error, not a result, and is retried.
- The 42 results from the first launch (pinned :novita) are kept: same provider, same reported model.
Hashes at resume:
```
51002a605793f07615c026873d674529a8d97613955da256b1f2cf9da2f1bff4  src/run.py
480e679f144fbaacb1cf617158fc1b8342239ae375229a16dd53524db871fcc7  config/models.yaml
9b4ef5c86a2e468954051f74a3445a1a1d6b8c0cad46216ede065ae376f3c1b2  config/pricing.yaml
```

## 2026-10-22T08:16:57Z: Qwen3.5-122B run STOPPED (provider routing; no wrong-model result kept)
- With Novita's circuit breaker open for the 122B, the :preferred route fell back to deepinfra, which answers and bills as Qwen3.5-397B-A17B. The runner's expect_model guard recorded 54 such calls as errors (finish_reason wrong_model, about $0.82 billed, not part of any result). The :preferred route later returned BadRequestError on every probe. Config reverted to the pinned :novita route.
- Good 122B results kept (qwen/qwen3.5-122b-a10b, novita): B 58, C 32 of 2,670. Everything stopped by the assistant; nothing is running.

## 2026-10-22T08:36:06Z: Qwen3.5-122B run STOPPED by the user (cost)
- Stopped by the user because of the cost. Partial 122B results are kept (not resumed): see outputs/raw/qwen122. Estimated spend on the 122B route in total, including 54 discarded wrong-model calls, is in the assistant's report.

## 2026-10-22T09:09:57Z: third model = Qwen3.5-9B via the Hugging Face router, provider DeepInfra (user decision after the 122B was stopped for cost)
- Why: the 122B (novita) was stopped for cost (about $4.86 spent, partial results in outputs/raw/qwen122 are not used). `:together` is not enabled for this account (HTTP 402, pay-as-you-go not enabled for provider together). Providers listed for Qwen3.5-9B: deepinfra ($0.10 / $0.15 per 1M in/out), ovhcloud ($0.12 / $0.18), featherless-ai.
- Verification before use: DeepInfra reports model label Qwen/Qwen3.5-9B and its response estimated_cost matches the 9B price exactly (this check was needed because the DeepInfra route for the 122B silently serves the 397B). Config guard expect_model qwen/qwen3.5-9b records any other answering model as an error, not a result.
- Smoke (12 ids, correctness NOT inspected): OVHcloud B 12/12 (1 truncated), C 12/12 (4 truncated at the 32,768-token cap); DeepInfra still finishing when this was written (B 6/12, C 4/12, 1 missing_agent format failure). Calls take 8-20 minutes; mean output 5k-17k tokens. OVHcloud entry removed from the config; its smoke outputs are kept in outputs/smoke/qwen9b_ovhcloud.
- Settings: thinking on (default), no sampling parameters sent, max output 32,768 tokens including thinking, retry waits 30/70/150 s (3 retries), spend guard $12 per prompt. The output-cap question (keep 32,768 or raise) is still open with the user.
Hashes at launch:
```
e0d0549f1c90b35c5065e74d37c6140cc91d9004959e3d38bc431ab05c331524  src/run.py
1a9d27815745a805d680210b0e1a33817141d2cf573989f2c4e753a7dafa6e2f  config/models.yaml
a6bb7dcd639eeeb1e49edc80e8602ea64f1ede89b7464827892f2b68e498c248  config/pricing.yaml
```

## 2026-10-22T09:22:04Z: Qwen3.5-9B run restarted at 8 workers per prompt with a longer read timeout (user decision)
- The router's circuit breaker (503 'provider paused') appeared at 64 requests in flight; the user asked for 8 workers per prompt and a longer timeout. src/run.py: the client read timeout (max seconds without a streamed chunk) is now configurable per model (timeout_read); qwen9b uses 1200 s (default 300 s for the other models). Streaming was already on. DeepInfra smoke processes (6 of 12 per prompt done) were stopped to keep the load at 8 + 8.

## 2026-10-22T21:01:22Z: Qwen3.5-9B with THINKING OFF (key qwen9b_nt), hosted on DeepInfra via the HF router (user decision: "put thinking off and continue")
- Reason: with thinking on, hosted calls took 10-40 minutes each (about 25 good results/hour per prompt at 8 workers), i.e. 100+ hours. The thinking-ON partial results (qwen9b: B 105, C 47 of 2,670, DeepInfra) are kept in outputs/raw/qwen9b and are NOT merged with or used as the main Qwen result.
- How thinking is switched off: request field chat_template_kwargs.enable_thinking=false (probe: 13 completion tokens and no reasoning field, versus 442 tokens with thinking; reasoning_effort=none and reasoning={...} also worked; top-level enable_thinking did not). New runner option: per-model extra_body in config. Same route, price ($0.10 / $0.15 per 1M), model-identity guard (expect_model qwen/qwen3.5-9b) and long retry waits.
- This is a different setting from luna and MiMo (reasoning models with thinking on); results are not like-for-like and must be labelled "Qwen3.5-9B, thinking off".
- Smoke (12 ids = 6 longest + 6 seed-0 random, correctness NOT inspected): B 12/12 ok, C 12/12 ok, 0 errors, 0 truncation, mean output 1,248 (B) / 678 (C) tokens, mean latency 59 s / 35 s. Parse failures: B 3/12 and C 2/12 (all missing_agent), which exceeds the plan's smoke rule (at most 1 in 12). The user instructed to continue; failed formats count as incorrect under the strict parser. Note: 6 of the 12 smoke ids are the longest traces.
- Run: 8 workers per prompt (user: keep 8), spend guard $12 per prompt.
Hashes at launch:
```
26fb2415f49e28f398eecbe76e7c66592b33c03708f01955855ad308503f9f34  src/run.py
0070826cb93dd0c9fd4eec0737d8420608e24c8152db63a3cb09f922b9660724  config/models.yaml
23b4ec25490a4bcbd1f5634c0803b51bfc40e6878c7a50ca0d3726cc54dd50d7  config/pricing.yaml
```
