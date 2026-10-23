# Schema notes (Phase 0)

W&W source: git clone of https://github.com/ag2ai/Agents_Failure_Attribution @ f4d2b6da464a826580e59b3a0eae15ea2d642d7c (under ijim/vendor, gitignored).
Counts: Algorithm-Generated 126 (expected 126), Hand-Crafted 58 (expected 58), total 184 (expected 184).
Record keys and history entry keys are read from the files (see load_data.load_ww); `mistake_step` is a STRING in the raw files and is converted to int.
Pro source: local copy data/text.jsonl of HF Leoxx/whowhen_pro text modality, 6257 rows (card says text ~6.3k). Frameworks: {'alfagent': 311, 'macnet': 261, 'metagpt': 81, 'smolagents': 3276, 'magentic-one': 811, 'debate': 445, 'dylan': 152, 'mathchat': 920}.
Pro columns: id, framework, benchmark, task (JSON: query, answer), trajectory (JSON), ground_truth (JSON, = the error label), extras (JSON).
Pro final answer: `task.answer`; null for some benchmarks (LiveCodeBench-Pro). `extras` was not used as an answer source.
Scope: only WW_AG, WW_HC, WWPRO_TXT (plan section 2); no other dataset is loaded.
WW_AG event keys beyond content: {'agent_raw': 1099, 'role': 1099}
WW_HC event keys beyond content: {'agent_raw': 2993}
