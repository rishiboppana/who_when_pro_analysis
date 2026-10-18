# Who&When Pro: schema notes (text split only)

Source: https://huggingface.co/datasets/Leoxx/whowhen_pro (CC-BY-4.0). Downloaded to `data/`: `text.jsonl` (6,257 rows, 75 MB), `taxonomy.yaml`, `README.md`.
Scope decision (user): image, image_gui and video splits are NOT used; the EDA covers error analysis on text traces only.

## Format
JSONL, one trace per row. Top-level columns: `id, framework, benchmark, task, trajectory, ground_truth, extras`. `task`, `trajectory`, `ground_truth`, `extras` are JSON strings (`json.loads`).

## Field map against the plan
| plan field | actual field | status |
|---|---|---|
| trace id | `id` (e.g. `alfworld_alfagent_0001`) | present |
| task / question | `task.query` | present |
| gold answer | `task.answer` | present, but null for alfagent (ALFWorld) and likely others; not yet counted |
| source benchmark | `benchmark` (16 values in text) | present |
| task category (9 in paper) | - | **not present** |
| modality | - | **not present** as a field; implied by split file (all text here) |
| agent framework | `framework` (8 values in text) | present |
| single vs multi-agent | - | **not present**; derivable from framework (+ `extras.agents`) |
| backbone model | - | **not present** |
| ordered steps | `trajectory` list (see below) | present, **structure differs per framework** |
| error agent | `ground_truth.agent` or `.agents` (list, debate/dylan) | null for single-agent (alfagent, smolagents) |
| error step | `ground_truth.step` / `.round` / `.position` / `.stage` depending on framework | present, **5 different shapes** |
| error mode | `ground_truth.mode` (code, e.g. `A.3`) | present |
| family | - | not a field; derivable from the mode prefix |
| injection metadata | per-step `is_injected`, `is_replayed` (mathchat, macnet), `injection_via` (magentic-one) | **partial**; no seed trace id, no relative position |

## Label shapes by framework (text split)
| framework | n | agent | step locator | trajectory element kinds |
|---|---|---|---|---|
| smolagents | 3276 | null | `step` (int) | user, action (reasoning/code/observation/error), planning |
| mathchat | 920 | `agent` | `round` + `position` | user, then untyped turns (agent_id, role, output, is_injected, is_replayed), final_answer |
| magentic-one | 811 | `agent` | `step` as **string**, e.g. "7.1" (round.position?) | user, agent (round, position, output, ledger, tool_actions) |
| debate | 445 | `agents` (list) | `round` | user, round (turns), final_answer |
| alfagent | 311 | null | `step` (int, 1-indexed) | user, action (31 per trace) |
| macnet | 261 | `agent` | `round` + `position` (+ `accepted_predictions`, 109 rows) | user, untyped turns (agent_id, role, stage), final_answer |
| dylan | 152 | `agents` (list) | `round` | user, round, final_answer |
| metagpt | 81 | `agent` | `stage` | user, untyped turns, final_answer |

So a single "error step index" does not exist; it must be built per framework (one mapping rule each). A unified step table needs a decision for round-based frameworks (debate/dylan have only round-level labels, no step within a round).

## Mismatches vs the paper / plan
1. **Taxonomy has 17 modes, not 18** (README and `taxonomy.yaml` agree). The plan's "A.2 Hallucinated tool/action" and "A.5 Looping" do not exist; the yaml's Action codes are A.1 Tool parameter/invocation error, **A.2 Output format/syntax error, A.3 Premature termination, A.4 Repetitive/looping behavior**. The plan's code-to-name list is therefore off for A.2-A.5 and this analysis uses the yaml names (so the top mode "A.3" is Premature termination, not Output format error).
2. **Only 14 of 17 modes occur in text**: absent P.1, P.2 (perception, expected), PL.2. Counts (yaml names): A.3 1207, R.4 1120, R.2 1089, R.3 739, R.1 621, V.2 486, C.3 346, A.2 287, V.1 128, PL.1 90, A.4 53, C.2 33, A.1 31, C.1 27.
3. **Counts:** text has 6,257 traces vs ~12,326 total; 8 frameworks (paper: 15), 16 benchmarks (paper: 26), 1 modality. Expected given text-only; not an error.
4. **Mean length** is not comparable to 7.5 until a unified step definition exists (per-framework means of the raw list: smolagents 5.6, magentic-one 7.1, alfagent 31, debate 4, mathchat 6.6, macnet 6, dylan 5, metagpt 5).

## Missing fields that block parts of the plan
- Task category, backbone, seed id: A4, A9, C4 (cluster bootstrap by seed) cannot be done as specified. Benchmark can stand in for task category; seed could be approximated by identical `task.query`.
- Injection position is not given; `injection_pos_rel` has to be derived from the label locator and trajectory length.
- Modality is constant (text), so A7 reduces to topology only.

## Example (full): alfagent `alfworld_alfagent_0001`
ground_truth = `{"agent": null, "step": 6, "mode": "A.4"}`, extras = `{}`, trajectory = 1 user turn + 30 action steps, each `{kind, step_number, reasoning, action, observation, is_final_answer}`. Task: "find two pencil and put them in drawer". (Step 3 `examine desk 1` instead of taking pencil 2 is already visible before the labelled step 6.)

## Update: how the missing fields were filled (built by `eda/00_build_tables.py`)
Cross-check of the raw file: the released labels are only **agent, step, mode**. `round`/`position`/`stage` are alternative step coordinates; `agent` is non-null in 2,073 traces and `agents` (list) in 597; `step` is present in 4,398 and absent where round/position/stage is used. These are kept as released in `gt_agent, gt_agents, gt_step, gt_round, gt_position, gt_stage` (no unification).

| field | result |
|---|---|
| task_category | **Filled from the paper** (Fig. 1 / ag2ai README, not in the data): Embodied (alfworld), STEM (gpqa, mmlu_pro, math, scibench), Data Science (dabench, dacode, databench, kramabench, tablebench), Coding (bigcodebench, humaneval, livecodebench_pro), Deep Search (gaia, simpleqa_verified, dsqa = DeepSearchQA). All 16 text benchmarks mapped; 5 categories appear in text. |
| backbone | **NA.** Paper App. G.2 says the backbone (GPT-4.1 / Gemini 3 Flash) is "recorded in the trace metadata", but the released `extras` has no such field. |
| seed trace id | **Not released.** `seed_group` = hash of (framework, task query) is a proxy: 2,671 groups for 6,257 traces. |
| injection_pos_rel | **Derived**: (k+1)/n_steps where k is the 0-indexed injected unit. Located per framework following `whowhen_eval/score.py`: `is_injected` flag (mathchat, macnet, metagpt); `step_number` (smolagents, alfagent; step 0 = the planning element, so smolagents counts planning + actions); `round.position` (magentic-one); `round` (debate, dylan, round-level only). 6,253 of 6,257 located; 4 smolagents traces (dsqa_0185, 0412, 0513, 0749) have a labelled step beyond the trajectory length and are left NA. |
| topology | Derived: smolagents and alfagent single-agent, all others multi-agent. |
| modality | constant "text". |
| taxonomy | From the dataset's `taxonomy.yaml` (17 modes) -> `tables/taxonomy_dataset.csv`; family from the code prefix. |

Outputs: `eda/outputs/tables/traces.parquet` (6,257 rows), `steps.parquet` (44,212 rows, `is_decisive_step` set from the located injection), `taxonomy_dataset.csv`.
