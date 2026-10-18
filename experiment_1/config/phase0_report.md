# Phase 0 report (stop point 1)

## 1. The Oct 12 run did NOT get lost. It is at `error_localization/prompt_A_benchmark/`
Frozen prompt files, render_A/render_B/render_C, strict parser, scorer, manifest (`config/frozen.md`), model config, and raw responses for 184 trajectories each:
| model | A | B | C | other |
|---|---|---|---|---|
| luna (gpt-5.6-luna, high, cap 32768) | 184 | 184 (+5 reps) | 184 (+5 reps) | old two-stage "C2" (luna only) |
| MiMo-V2.6-Pro-RL:novita (cap 131072) | 184 | 184 | 184 | |
| Kimi-K3:together | 184 | 184 | 184 | to archive |
Prompt hashes: A.txt 03de25bb..., C.txt 20ce2377... (recorded in frozen.md).
The repo-root files `render_B.py`, `render_C.py` are copies of `src/render_B.py` / `src/render_C.py`.

## 2. Render faithfulness (plan 2.2/2.3): FAILS, and cannot pass with the frozen A.txt
The repo's `all_at_once` (Automated_FA/Lib/utils.py L47-93) builds a different string than the frozen A.txt:
- repo: straight quotes, two spaces after "The problem is:", "\n\n" before the conversation, numbered items on separate lines, indented example block, format line `Agent Name: ...\n Step Number: ...\n Reason for Mistake: \n`. frozen A.txt: the paper appendix wording with curly quotes, "Here’s the conversation: {log}" on one line, different line breaks.
- repo also sends a **system message** ("You are a helpful assistant skilled in analyzing conversations."); the plan (rule 3) requires none.
- Character similarity of the two full prompts on trajectory AG/1: 0.98; not byte-identical.
- The log rendering itself (`name: content` joined by "\n") IS identical to the repo.
- Repo scorer (`evaluate.py`) uses **substring** matching (`actual in predicted`), agent regex `[\w_]+`, and no normalized error. The plan's scorer is exact match.

## 3. Items needing the user's decision (see chat)
