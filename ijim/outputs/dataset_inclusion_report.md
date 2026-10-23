# Dataset inclusion report

- WW_AG: 126 kept (expected 126). WW_HC: 58 kept (expected 58).
- Pro text rows scanned: 6257. Exclusions and kept counts (candidate frameworks magentic-one, macnet, mathchat, metagpt):
  - excluded_framework_not_in_keep_list:alfagent: 311
  - excluded_framework_not_in_keep_list:smolagents: 3276
  - excluded_round_level_label:debate: 445
  - excluded_round_level_label:dylan: 152
  - kept:macnet: 261
  - kept:magentic-one: 811
  - kept:mathchat: 920
  - kept:metagpt: 81
  - rows_total: 6257
- Pro sample: n=300 (target 300, seed 20261022); per framework {'macnet': 47, 'metagpt': 27, 'magentic-one': 107, 'mathchat': 119}
- Pro stored final answer in 271/300 sampled rows (90%) -> answer_variant should be no_answer (set in config/datasets.yaml)
- AgentRx: microsoft/AgentRx is gated; data files returned HTTP 403 with the available HF token. Its card documents root-cause failures with step_number and failed_agent, so it could qualify; the user must accept the access terms. Excluded for now.
- Aegis-Bench (no step labels) and TRAIL (span-level, no agent label) are excluded as ineligible.
- Eligible datasets: 3 (WW_AG, WW_HC, WWPRO_TXT) -> leave-one-dataset-out works with 3, but each training fold has only 2 datasets.
