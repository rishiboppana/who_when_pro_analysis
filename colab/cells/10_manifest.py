import datetime, hashlib
freeze = subprocess.run(["uv", "pip", "freeze", "--python", f"{VENV}/python"], capture_output=True, text=True).stdout
open(f"{E1}/E1_qwen_colab_pip_freeze.txt", "w").write(freeze)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = f"""
## Qwen3.5-9B 4-bit on Colab ({datetime.datetime.utcnow().isoformat(timespec='seconds')}Z)
- Model: {MODEL} (Apache-2.0), served by vLLM; quantization: 4-bit GPTQ INT4 weights (group size 128) for the linear layers of the transformer blocks, activations BF16; linear-attention layers, embeddings and vision encoder stay BF16. NOT the BF16 original.
- GPU: {GPU_NAME}, {MEM_GB:.0f} GB, driver {DRIVER}, python {sys.version.split()[0]}; pip freeze in E1_qwen_colab_pip_freeze.txt
- Server command: `{' '.join(SERVE_CMD)}`
- Thinking: {"ON via --default-chat-template-kwargs enable_thinking=true; --reasoning-parser qwen3 (thoughts excluded from the scored text)" if THINKING else "OFF"}; one setting for prompts B and C
- Sampling (config/models.yaml): temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, seed 0 (Qwen3.5 model card, thinking mode, general tasks); max output tokens 32768 incl. thinking; max_model_len {MAX_MODEL_LEN}; single user message, no system prompt; concurrency {PRESET["seqs"]} sequences
- Hashes: prompts/B.txt {sha(E1+"/prompts/B.txt")[:16]}, prompts/C.txt {sha(E1+"/prompts/C.txt")[:16]}, src/parser.py {sha(E1+"/src/parser.py")[:16]}, src/score.py {sha(E1+"/src/score.py")[:16]}, src/run_local.py {sha(E1+"/src/run_local.py")[:16]}
- vLLM is not fully deterministic. Deviations: 4-bit quantized model; sampling per the model card (not temperature 0); presence_penalty 1.5 is part of the recommended thinking-mode sampling.
"""
open(f"{E1}/E1_manifest.md", "a").write(man); print(man)
