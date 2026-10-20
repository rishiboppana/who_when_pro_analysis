!rm -rf /content/whowhen_pro && git clone -q https://github.com/ag2ai/whowhen_pro /content/whowhen_pro && git -C /content/whowhen_pro checkout -q 14369dcb079c7cdf86ed2924145d5a28ff5c73b2
os.environ["WHOWHEN_EVAL_REPO"] = "/content/whowhen_pro"      # official per-framework renderers (commit 14369dc) used by experiment_1/src/data.py
os.environ["VLLM_API_KEY"] = "EMPTY"                          # the runner's client sends this; the SERVER is started without it (see the server cell)
# make the E1 config match the model chosen in the first cell (the runner reads model_id and max_model_len from config/models.yaml)
cfg = f"{E1}/config/models.yaml"; s = open(cfg).read(); head, qwen = s.split("  qwen:")
qwen = re.sub(r'model_id:\s*"[^"]+"', f'model_id: "{MODEL}"', qwen); qwen = re.sub(r"max_model_len:\s*\d+", f"max_model_len: {MAX_MODEL_LEN}", qwen)
open(cfg, "w").write(head + "  qwen:" + qwen)
print([l.strip() for l in qwen.splitlines() if "model_id" in l or "max_model_len" in l or "max_output_tokens" in l or "sampling" in l])
!cd {E1} && /content/venv/bin/python -m pytest -q tests
