# IJIM Two Cross-Dataset Experiments: Execution Plan for Claude Code

Experiment 1 = **Diagnostic Observability** (does trace format change failure-attribution accuracy?).
Experiment 2 = **Cross-Dataset Reliability Gate** (can 5 repeated diagnoses be turned into an accept-vs-escalate signal?).

This plan implements the "Two Cross-Dataset Experiments: Student Instructions" document exactly. If anything here conflicts with that document, the document wins; stop and tell the user.

## 0. How Claude Code must behave

1. Execute phases in order. At every **STOP** marker, show the user the named file(s) and wait.
2. Never invent a missing field, label, or provenance value. If something expected is absent, record it in `config/schema_notes.md` and ask.
3. Never read correctness during the smoke test (parseability, truncation, latency, cost, context fit only).
4. Never hand-fix, re-prompt, or retry because of a parse failure.
5. Never put any label (`mistake_agent`, `mistake_step`, `mistake_reason`, Pro `ground_truth`, `mode`, `level`, `is_correct`) into any prompt.
6. API keys come from environment variables only (`OPENAI_API_KEY`; `OPENROUTER_API_KEY` or `XIAOMI_API_KEY` only if the optional second judge is enabled). Never write a key to a file or log. If a key is missing, stop and ask.
7. After Phase 3 (freeze) no change to prompt, judge, settings, parser, scorer, renderers, feature set, model class, or thresholds. If a bug is found later, stop, report it, and let the user decide; do not patch silently.

## 1. What was recovered from earlier chats (verify, do not trust blindly)

| Item | Value from earlier chats | Status |
|---|---|---|
| Original W&W dataset (HF) | `Kevin355/Who_and_When`, configs `Algorithm-Generated`, `Hand-Crafted` | verify |
| Original W&W repo | `https://github.com/ag2ai/Agents_Failure_Attribution` (use `git clone`; web fetch of GitHub was blocked in chat) | verify |
| W&W on-disk layout | `Who&When/Algorithm-Generated/*.json`, `Who&When/Hand-Crafted/*.json` (also seen as `Who_and_When/...`) | search for it |
| Uploaded parquet fallbacks | `/mnt/user-data/uploads/Algorithm-Generated.parquet`, `/mnt/user-data/uploads/Hand-Crafted.parquet` | use only if the above are absent |
| Original prompt code | `Automated_FA/Lib/utils.py`, function `all_at_once`; matching logic in `evaluate.py` | read, copy exactly |
| Expected W&W size | 184 = 126 Algorithm-Generated + 58 Hand-Crafted | verify, report any mismatch |
| W&W record fields | `question`, `question_ID`, `ground_truth`, `history` (entries with `content`, `role`, possibly `name`), `mistake_agent`, `mistake_step` (**string**), `mistake_reason` | verify |
| Who&When Pro (HF) | `Leoxx/whowhen_pro`; repo `https://github.com/ag2ai/whowhen_pro` | verify |
| Pro columns | `framework`, `benchmark`, `task`, `trajectory` (JSON string), `ground_truth` (JSON string **= the error label**, e.g. `{"agent": null, "step": 6, "mode": "A.4"}`), `extras` (JSON string) | verify |
| Pro size | card shows ~10.7k rows (text ~6.3k, image ~2.9k, image_gui ~0.7k, video 916) vs 12,326 in paper | record actual counts |
| Pro caveat | step coordinate is framework-specific (some `round`/`position`/`stage`); only `image_gui` is documented as 1-indexed | inventory before use |
| AgentRx (HF) | `microsoft/AgentRx` (paper arXiv 2602.02475, ~170 trajectories in the current version, Magentic-One split is multi-agent) | include only if it stores responsible agent + step |
| Earlier-chat output names (may exist in the working dir) | `config/schema_notes.md`, `config/index_convention.md`, `config/models.yaml`, `prompts/A.txt`, `render_A`, `outputs/example_rendered_prompt_A.txt`, `E1_predictions.csv`, `E1_summary.csv`, `E1_prompts.txt`, `E1_manifest.md` | see 1.1 |
| Current judge | `gpt-5.6-luna`, reasoning effort high, OpenAI Responses API | frozen as J1 |
| Optional second judge | `mimo-v2.6-pro` (`xiaomi/mimo-v2.6-pro` on OpenRouter), open-weight | only if enabled in 4.3 |

**Note on model names:** model names and prices change. Read the provider docs and billing page at run time; do not guess parameter names.

### 1.1 Name collision rule
`E1_*` files belong to an earlier, different study (multi-model x prompt benchmark). **Do not overwrite or reuse their results.** Everything for this study goes under `ijim/` with prefixes `X1_` (Experiment 1) and `X2_` (Experiment 2). If `config/schema_notes.md`, `config/index_convention.md`, `prompts/A.txt` or `render_A` already exist, copy them into `ijim/config/` and `ijim/prompts/`, record their SHA-256, and re-verify them on 5 records instead of re-deriving from scratch.

## 2. Datasets (units for "within each dataset" and "leave-one-dataset-out")

| ID | Source | Notes |
|---|---|---|
| `WW_AG` | W&W Algorithm-Generated | 126 expected. Natural failures, CaptainAgent-built systems |
| `WW_HC` | W&W Hand-Crafted | 58 expected. Natural failures, Magnetic-One |
| `WWPRO_TXT` | Who&When Pro, text modality, filtered and sampled (2.1) | injected (synthetic) errors; exact labels by construction |
| `AGENTRX` | AgentRx multi-agent split | **conditional**; include only if it stores a responsible agent and a single critical step per trajectory |

A dataset is eligible only if it stores (a) an ordered event list, (b) a true responsible agent, (c) a true step coordinate that indexes that event list. Aegis-Bench (no step labels) and TRAIL (span-level, no agent label) are excluded; say so in `dataset_inclusion_report.md`.

If fewer than 3 datasets end up eligible, **STOP** and ask the user, because leave-one-dataset-out gets weak.

### 2.1 Pro filtering and sampling (decide and freeze before any model call)
Keep a Pro row iff: text modality; `ground_truth.agent` is non-null (joint WHO+WHEN needs both); `ground_truth.step` is a single integer that indexes the stored event list under a convention verified by reading at least 10 rows per framework. Drop frameworks with composite coordinates (round+position, stage) unless the coordinate maps one-to-one to a single stored event index without any invented remapping. Write every exclusion count to `outputs/dataset_inclusion_report.md`.

Sample: stratified by `framework`, proportional allocation with a minimum of min(20, available) per framework, seed `20261022`, target n = **300** (or all if fewer). Save the sampled IDs to `config/pro_sample_ids.json`. The user may change the target before the freeze; after the freeze it is fixed.

**Pro leakage trap:** Pro's `ground_truth` column is the error label. It must never reach a prompt. Pro has no stored "final answer" unless `extras` says so (see 4.1).

### 2.2 Question repeats
Compute a `cluster_id` = hash of the normalized question text. Experiment 1 and 2 bootstraps resample by `cluster_id`. W&W Algorithm-Generated and Hand-Crafted can share source tasks; the cluster id is computed across all datasets, but resampling is within the dataset being analysed.

## 3. Repo layout

```
ijim/
  config/        datasets.yaml  models.yaml  schema_notes.md  index_convention.md
                 provenance_inventory.md  pro_sample_ids.json  protocol_freeze.md
  prompts/       judge_with_answer.txt  judge_no_answer.txt
  src/           load_data.py  render.py  parse.py  score.py  run_judge.py
                 stats_exp1.py  features_exp2.py  gate_exp2.py  estimate_cost.py
  data/normalized/   WW_AG.jsonl  WW_HC.jsonl  WWPRO_TXT.jsonl  [AGENTRX.jsonl]
  raw/           calls.jsonl   (append-only: every API request/response)
  tests/         test_*.py
  outputs/exp1/  outputs/exp2/  outputs/dataset_inclusion_report.md
```

Normalized record (one per trajectory): `dataset, traj_id, cluster_id, question, final_answer_or_null, events[ {i, agent_raw, role_raw, content, native_fields{} } ], L, true_agent, true_step_native, step_base (0 or 1)`.

## 4. Phase 0 and 1: Recon and build (no model calls)

### 4.1 Recon
1. Locate or download the data (section 1). Inspect one record per subset. Write real field names, counts, and any mismatch with section 1 to `config/schema_notes.md`.
2. **Step-index convention.** For several records per dataset, read the event at `true_step` and confirm it plausibly belongs to `true_agent`. Is the label 0-based or 1-based, and does it index the event list directly? Write the verified convention to `config/index_convention.md`. If ambiguous, **STOP**.
3. **Original rendering.** Read `Automated_FA/Lib/utils.py::all_at_once` and `evaluate.py`: how each event becomes text (fields, order, separator), the exact prompt string, how agent and step matches are decided. Copy that logic exactly into `render_O0(events)` and `match_agent()/match_step()`. Save a rendered example to `outputs/example_rendered_O0.txt`.
4. **Judge prompt text.** Save the original all-at-once prompt once to `prompts/judge_with_answer.txt` (the version with `The Answer for the problem is: {ground truth}`) and once to `prompts/judge_no_answer.txt` (identical but with that single sentence removed). Use `prompts/A.txt` from earlier work if it exists; otherwise take it from `all_at_once`. Placeholders `{problem}`, `{ground truth}`, `{failure log}` are filled by **plain string replacement**, not `str.format` (the text contains braces). The variant is chosen per dataset by whether a final answer is truly stored (W&W: yes; Pro: only if `extras` stores it), never by results. Record the choice in `datasets.yaml`.
5. **Provenance inventory (for O2).** Per dataset, list native fields on events beyond what `render_O0` already shows (role, action type, tool name, source). Write to `config/provenance_inventory.md` with coverage % per field. O2 is run for a dataset only if at least one such field exists on at least 95% of events. If the fields are already in the O0 rendering, O2 would equal O1, so mark O2 "not applicable" and do not run it.
6. Compute rendered-prompt token counts (state the tokenizer approximation) and report max, median, 5 longest per dataset. If any prompt exceeds the judge's context minus the max output budget, **STOP**; do not truncate.

### 4.2 Trace conditions (implement in `render.py`, frozen at Phase 3)

| Condition | Rule |
|---|---|
| **O0** | `render_O0` exactly as the original code does. No added structure |
| **O1** | O0 text with `[Step k]` inserted before each event, `k` in the dataset's **native label coordinate** (0-based for W&W if verified; if a dataset is natively 1-based, start at 1 and log it). No event content is rewritten or removed |
| **O2** | O1 plus a one-line suffix per event with only the native fields from `provenance_inventory.md`, fixed key order, no placeholders for missing values |
| **O1-A** | O1 where each distinct agent name becomes `Agent A, Agent B, ...` in order of first appearance, per trajectory. Replace the speaker label **and** whole-word occurrences of that name inside event content (deterministic, count replacements, log them). Keep the mapping; predicted `Agent X` is mapped back to the original name before scoring. A prediction with an unknown alias is a wrong agent, not a parse failure |

The judge prompt text is identical across conditions; only `{failure log}` changes.

### 4.3 Judges
- **J1 (primary, used in both experiments):** `gpt-5.6-luna`, highest documented reasoning effort, one user message, no system prompt.
- **J2 (optional, Experiment 1 only):** `mimo-v2.6-pro`. Disabled by default; the user sets `ENABLE_J2: true` in `config/models.yaml` **before** the freeze. After the freeze J2 cannot be added.
- `config/models.yaml` records per judge: provider and base URL, exact model ID, any snapshot string returned by the API, reasoning setting, temperature and top_p (or "provider ignores/rejects"), max output tokens, whether a seed parameter exists (if not: `seed_supported: false`), date of first call. Same provider for every call to a model, fallbacks disabled.

### 4.4 Parser and scorer (`parse.py`, `score.py`), written and unit-tested before any API call
- Parser: from the response, take the first `Agent Name:` line and the first `Step Number:` line (case-insensitive). Agent non-empty, step a base-10 integer. Anything else is a **parse failure**. Truncated or `incomplete` responses are parse failures. Keep the raw text.
- Agent correctness: original `evaluate.py` rule, identical in all conditions (after O1-A mapping back).
- Step correctness: predicted integer equals `true_step_native`.
- Joint = agent correct AND step correct. Parse failures count as incorrect for agent, step, and joint, and are reported separately.
- Normalized step error = `|pred - true| / max(L - 1, 1)`, parseable runs only; report `n_parseable` beside every median.
- Required tests: parser on good, malformed, truncated, and multi-answer outputs; O1 rendering leaves event content byte-identical (strip the added labels and compare to O0); O1-A is reversible; **no-leakage test** that no rendered prompt contains any label field, `mistake_reason` text, or Pro `ground_truth` JSON; feature functions (section 8) take only run outputs and `L`, never labels.

**STOP 1:** show `schema_notes.md`, `index_convention.md`, `provenance_inventory.md`, `dataset_inclusion_report.md`, example rendered prompts for all conditions on one trajectory per dataset, and the test results.

## 5. Phase 2: Smoke test and cost (no correctness)

1. Choose 12 trajectories (6 longest + 6 seeded random across datasets), run **O1 only** on J1 (and J2 if enabled).
2. Report only: parse rate, truncation, latency, tokens (input, output, reasoning), cost, context fit. Never print correctness.
3. Pass rule: at most 1 parse failure out of 12 per judge. If J2 fails and is optional, drop J2 and say so. If J1 fails, **STOP**.
4. `estimate_cost.py` projects total spend:
   `calls = N_total x (conditions per dataset) x judges` for Experiment 1, plus `N_total x 5` for Experiment 2, using measured tokens. Ask the user for a spend cap. If the estimate exceeds roughly $50, **STOP** and ask for confirmation. Note luna price sources disagreed in an earlier chat; use the billing page.

## 6. Phase 3: Freeze

Write `config/protocol_freeze.md`: SHA-256 of both prompt files, parser, scorer, renderers, feature code, gate code, `models.yaml`, `datasets.yaml`, `pro_sample_ids.json`; git commit hash; library versions; the analysis seed `20261022`. **STOP 2:** user confirms the freeze. Then run.

## 7. Experiment 1: Diagnostic Observability

### 7.1 Runs
- Conditions: O0, O1, O1-A on every trajectory of every dataset; O2 only where 4.1 step 5 allows.
- Run each (dataset, judge, condition, trajectory) **once**. Use an async pool with a concurrency limit, a resumable cache keyed by `sha256(dataset|traj_id|condition|judge|replicate)`, append every request and response to `raw/calls.jsonl` **before** parsing, longest prompts first.
- Only transport errors (HTTP 5xx, timeout, connection reset, rate limit) may be retried, up to 3 times with identical parameters. A received response is final.
- Stop automatically if any (judge, condition) parse-failure rate exceeds 20% after 30 calls and report to the user.
- Do the dataset's O0 run before its O1 runs so the order cannot be tuned to results.

### 7.2 Scoring
Per run record: agent, exact-step, joint correctness, parse failure, normalized step error. Save `outputs/exp1/X1_predictions.csv` with `dataset, judge, condition, traj_id, cluster_id, raw_response, parsed_agent, parsed_step, true_agent, true_step, agent_ok, step_ok, joint_ok, parse_fail, norm_err, tokens_in, tokens_out, tokens_reasoning, latency_s, finish_reason, response_id, model_snapshot, timestamp`.

### 7.3 Statistics (`stats_exp1.py`), seed 20261022
- Primary endpoint: **exact joint WHO+WHEN accuracy**.
- Primary comparison, within each dataset and judge: **O1 vs O0**, paired by trajectory. Secondary: **O2 vs O1** (only inside datasets that support O2; never pooled with datasets lacking provenance), **O1-A vs O1**.
- Per comparison report: n, discordant counts (b, c), paired change in percentage points, **95% paired-bootstrap CI** (resample `cluster_id` clusters with replacement, B = 10,000, percentile interval), **McNemar exact test** (two-sided binomial on discordant pairs) raw p, and **Holm-adjusted p** over the pre-specified comparisons in that dataset-judge family (up to 3: O1 vs O0, O1-A vs O1, O2 vs O1).
- Also give Wilson 95% CIs for agent, step, and joint accuracy in the result table.
- Do not pool datasets in any test. A descriptive across-dataset summary is the forest plot only.
- Secondary metrics (agent accuracy, exact-step accuracy, parse failures, normalized step error) get the same paired treatment but are labelled secondary and are not Holm-corrected into the primary family.

### 7.4 Deliverables (in `outputs/exp1/`)
1. **Result table** `X1_results.csv`: `Dataset | Judge | Trace condition | n | Agent % | Step % | Joint % | Median normalized error` (+ Wilson CI columns, `parse_fail %`, `n_parseable`).
2. **Effect table** `X1_effects.csv`: for O1 vs O0, `Dataset | Judge | n | b | c | Delta joint (pp) | 95% CI | raw p | Holm p`. Secondary comparisons in `X1_effects_secondary.csv`.
3. **Forest plot** `X1_forest.png` of the O1-over-O0 effect (pp, 95% CI) per dataset and judge, zero line, n and Holm p annotated. Use the `dataviz` skill palette rules.
4. **Interpretation** `X1_interpretation.md`: 4 to 6 sentences. Claim only what the controlled change supports: *adding explicit step labels to the same judge input changed exact joint accuracy by X pp on these datasets*. Do not say "observability" in general, do not generalize beyond the datasets and judge, and state Pro labels are injected errors while W&W failures are natural.

## 8. Experiment 2: Cross-Dataset Reliability Gate

### 8.1 Repeated diagnoses
- Format **O1 only**, judge **J1 only**, same trajectories as Experiment 1.
- For each trajectory, make **5 fresh API calls** (replicate 1 to 5) with identical prompt and settings. Do not reuse Experiment 1's O1 run as one of the five. Seed handling: use a seed if the API supports one and differs per replicate **only if** the freeze says so; otherwise record `seed_supported: false`.
- Keep each run's complete (agent, step) pair together. Parse failures stay failures; never repair. Save `outputs/exp2/X2_runs.csv` (one row per call).

### 8.2 Features (`features_exp2.py`): exactly these seven, label-free
Let `P` = the parseable (agent, step) pairs among the 5 runs.

| Feature | Definition |
|---|---|
| `joint_modal_agreement` | count of the most frequent identical (agent, step) pair / 5 |
| `agent_agreement` | count of the most frequent agent among parseable runs / 5 |
| `step_agreement` | count of the most frequent step among parseable runs / 5 |
| `unique_pair_count` | number of distinct complete pairs in `P` |
| `norm_step_spread` | (max step - min step over parseable runs) / max(L - 1, 1) |
| `traj_length` | L = stored events |
| `parse_failure_count` | failed parses among the 5 (0 to 5) |

Fixed conventions for undefined cases (pre-specified, record how many trajectories hit them): no parseable run gives agreements 0, `unique_pair_count` 0, spread 0; one parseable run gives spread 0.

### 8.3 Candidate and target
- Candidate = most frequent complete (agent, step) pair. On a tie, take the tied pair that appeared in the earliest run (run 1 to 5); record `tie_occurred` (not a feature).
- Trajectories with no parseable run have no candidate: target 0, kept in the data and counted.
- **Target = 1** only if the candidate is exactly correct on WHO and WHEN.
- Nothing about correctness, true labels, error category, or mode may enter a feature.

### 8.4 Gate (`gate_exp2.py`)
- Model: logistic regression, **only the seven features**, L2, fixed `C = 1.0`, `class_weight=None`, standardization fit on training datasets only. No deep model, no interactions, no feature transforms, no tuning on held-out data.
- Sample weights: each training dataset gets equal total weight (so a large Pro sample does not dominate). Pre-specified; do not change.
- **Leave-one-dataset-out:** for each dataset D, train on all others, predict on D, rotate through all datasets. Save held-out predicted probabilities to `X2_gate_predictions.csv`.
- Pre-specified sensitivity (reported separately, labelled sensitivity): treat `WW_AG` and `WW_HC` as one fold, because they share task sources.

### 8.5 Evaluation
On each held-out dataset:
- **Risk-coverage (primary).** Rank by predicted probability (ties broken by original row order, not by labels). Report joint accuracy for the top 100%, 75%, 50%, 25% retained (retained n = ceil(coverage x N)).
- **Baselines.** Accept-all; consensus-only `joint_modal_agreement` >= 3/5, >= 4/5, = 5/5. Report each baseline's actual coverage and retained n, since coverage is not fixed for thresholds.
- **Calibration.** Brier score for the gate, and for accept-all as the constant equal to the training-datasets' target rate (consensus rules are binary, so Brier is "n/a"). One reliability plot (10 quantile bins, pooled held-out predictions, plus per-dataset small multiples).
- **Discrimination (secondary).** AUROC for the gate with 95% bootstrap CI; also AUROC of `joint_modal_agreement` as a continuous baseline. If a held-out dataset has a single target class, write "AUROC undefined".
- **Uncertainty.** Trajectory-level bootstrap (resample by `cluster_id`, B = 10,000, seed 20261022), re-ranking inside each resample. CIs for retained joint accuracy, Brier, AUROC.
- Do **not** choose a best threshold after seeing test data.

### 8.6 Deliverables (in `outputs/exp2/`)
1. **Result table** `X2_results.csv`: `Held-out dataset | Method | Coverage % | Retained n | Joint % | Brier score | AUROC | 95% CI/note`. Methods: Gate@100/75/50/25, Accept-all, Consensus>=3/5, >=4/5, 5/5.
2. **Figures:** `X2_risk_coverage.png` (gate curve against the three consensus points per held-out dataset), `X2_calibration.png`.
3. **Interpretation** `X2_interpretation.md`: 5 to 7 sentences on the accuracy/coverage trade-off. Say self-consistency is not proof of correctness, report cases where high agreement was still wrong, and make no threshold recommendation.
4. Counts of tie cases, no-parseable-run trajectories, and unique-pair-count distribution per dataset.

## 9. Final stop rules and reporting
- Freeze the protocol and raw outputs (`raw/calls.jsonl` is never edited).
- No post-hoc trace formats, features, thresholds, judges, or sample changes. Anything extra is labelled **exploratory** in a separate file.
- Write `outputs/X_manifest.md`: exact model IDs and returned snapshots, dates, settings, hashes, dataset versions and counts, filtering counts, transformation rules, parse-failure rates, spend.
- Final message to the user: the file list, one sentence per experiment on the main result, anything that could not be done, and the open caveats (injected vs natural errors, small W&W n, judge is a cost-tier model, no seed if unsupported).

## 10. Order of work
1. Phase 0 and 1 (recon, build, tests). **STOP 1.**
2. Phase 2 smoke test and cost estimate. Ask for spend cap.
3. Phase 3 freeze. **STOP 2.**
4. Experiment 1 runs, scoring, stats, deliverables. Show `X1_results.csv` and `X1_effects.csv` before starting Experiment 2.
5. Experiment 2 runs, features, gate, evaluation, deliverables.
6. Manifest and final report.
