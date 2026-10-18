# Who&When (original 184) schema notes, Phase 0
Source: git clone of https://github.com/mingyin1/Agents_Failure_Attribution @ b2bae5c (a clone of ag2ai/Agents_Failure_Attribution under that URL; confirm it is the authoritative copy). Also compared against the local HF copy `error_localization/who_and_when_dataset/` (HF `Kevin355/Who_and_When`).

Counts: Algorithm-Generated 126 + Hand-Crafted 58 = **184** (matches).

| subset | record keys | history entry keys |
|---|---|---|
| Algorithm-Generated | is_correct, question, question_ID, level, ground_truth, history, mistake_agent, mistake_step, mistake_reason, system_prompt (+ `labels` in 1 record) | content, role, name |
| Hand-Crafted | history, question, ground_truth, is_corrected, question_ID, mistake_agent, mistake_step, mistake_reason, (mistake_type in 4, level in 30) | content, role |

- `mistake_step` is a **string** in both subsets. No `step` field exists in the history entries of either raw subset.
- Speaker key for rendering: `name` (AG), `role` (HC), as in the original `all_at_once`.
- Repo copy vs local HF copy: AG identical. HC differs in (a) the key name (`ground_truth` in repo, `groundtruth` in HF), (b) `mistake_agent` for HC 11 and 13 ("WebSurfer" in repo, "Websurfer" in HF). No other label or history differences. HC_025/039/050 (mistake_step 18/30/5 of lengths 20/49/113) show no difference between these two copies; the "24, 4, 15" experiment-file values mentioned in the plan are not in either copy.
- `int(mistake_step) >= len(history)`: none in either subset.
- The pasted example in the chat has `"step": "step - N"` inside each history entry. That field is NOT in the repo or HF raw files; it exists only in the locally derived `error_localization/who_and_when_dataset/step_labeled_trajectories/` (187 files) or was added by the user.
