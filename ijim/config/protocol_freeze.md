# Protocol freeze
Frozen: 2026-10-23T00:16:41.681449+00:00 (user confirmed 2026-10-22). Analysis seed 20261022.
Git HEAD: 4f99d729e52165cdabaa2eb9f3693163ab8366f4 (the ijim/ folder is untracked/uncommitted at freeze time; file hashes below are authoritative).
Judge: gpt-5.6-luna, Responses API, reasoning effort HIGH (user decision), max_output_tokens 32768, one user message, no system prompt, seed_supported: false (no seed used in Exp 2). Single judge; no J2. Datasets: WW_AG, WW_HC, WWPRO_TXT (300 sampled rows, 4 frameworks). Pro prompt variant: no_answer; W&W: with_answer. O2 only for WW_AG (role).
Exp 1 replicate = 0; Exp 2 replicates 1..5 (fresh calls). W&W O0 is re-run, earlier runs are not reused.
Python 3.11.14; numpy 2.4.6; pandas 3.0.6; scipy 1.17.1; scikit-learn 1.9.1; openai 3.35.1; tiktoken 0.14.0

| file | sha256 |
|---|---|
| prompts/judge_with_answer.txt | 03de25bbfb169875cbbf79fd5baaf06f248c80e6be8fed31cb1a58f0a27a7c31 |
| prompts/judge_no_answer.txt | 8c2498e165075f70c3714c948f914a65790ffb58edb6630a54491343a53139ef |
| src/parse.py | 12103b96290a58f622eafdacaa0269dc773265faa710cb206cb11188966127d6 |
| src/score.py | 78f4c6141557b9d08cae6101d5ec008a142d6e7414b1ee42e247c8195c110c7f |
| src/render.py | bc909960813c0a877dcbf2577acd2cc7e3d4efddc84fd00bb681fcb8113cfcc2 |
| src/load_data.py | 92c967d532e941fc73276579df89f57b7c8cec3865c7dbb87667ba9719a15eef |
| src/run_judge.py | 4b23cb10124125a4ac5cd9ce073ebd7657a7ad514ef0be13b6eb62a9b8b2411b |
| src/features_exp2.py | d6c2d163336c2b464d0f135c80e7485e00bd2ae0ac62987a18b9e0a41e41d98a |
| src/gate_exp2.py | 010e03e35b82dd7161a4bc14c88779c5f82e4e5785860893c8f77e5484844325 |
| src/stats_exp1.py | 71906415692acd5888d1dbb77e3811af9a5d34916aeda4af4174202558c441c8 |
| src/make_tables.py | 71677605bd7b9f31475d61fa50f26661b04350a68ccccae1cf1f06dbbaf3bbc7 |
| config/models.yaml | a23271139a6e0a9bf780b1b774adeb32943163284f220ae72dee609390bddebf |
| config/datasets.yaml | 3a2c30c9ea915d38e354f15d7fa8339333c33e30adb6281a0dedb17e3bd8ce16 |
| config/pro_sample_ids.json | f1e1e21b4f02abd8f6c2a5efc6d64f2f2677bc9cb1cb050efc73a67f7d0b0087 |
