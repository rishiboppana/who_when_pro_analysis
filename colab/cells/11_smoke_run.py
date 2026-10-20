!cd {E1} && /content/venv/bin/python src/run_local.py --stage smoke --models qwen --prompts B C --workers {max(1, PRESET["seqs"] // 2)} 2>&1 | tail -30
