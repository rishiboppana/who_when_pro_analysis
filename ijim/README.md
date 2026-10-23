# IJIM two experiments (plan: ../ijim-two-experiments-plan.md)
Run with the repo venv: `../.venv/bin/python`. Nothing here makes a model call until you run `run_judge.py`.

| Phase | Command |
|---|---|
| 0/1 recon, normalize, reports, examples | `python src/recon.py` |
| tests | `python -m pytest tests -q` |
| 2 smoke (12 trajs, O1, no correctness) | `python src/run_judge.py --stage smoke --judges J1` |
| 2 cost | `python src/estimate_cost.py` (fill `price_usd_per_mtok` in config/models.yaml first) |
| 3 freeze | write config/protocol_freeze.md (hashes, git commit, versions) after STOP 2 |
| Exp 1 | `python src/run_judge.py --stage exp1 --judges J1 --workers 8`, then `python src/make_tables.py && python src/stats_exp1.py` |
| Exp 2 | `python src/run_judge.py --stage exp2 --judges J1 --workers 8`, then `python src/make_tables.py && python src/gate_exp2.py` |

Keys: OPENAI_API_KEY (J1), HF_TOKEN (J2 only). Read from ../.env or the environment; never written anywhere.
Exp 1 replicate = 0, Exp 2 replicates = 1..5 (separate cache keys; Exp 1's O1 run is never reused).
