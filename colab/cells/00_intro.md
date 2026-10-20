# Qwen3.5-9B (4-bit) on Google Colab: Experiment 1 inference + Experiment 2 analysis (single self-contained notebook)

Runs **Qwen3.5-9B in 4-bit** (`RedHatAI/Qwen3.5-9B-quantized.w4a16`: GPTQ INT4 weights, BF16 activations; Apache-2.0, not gated) with **vLLM 0.32.0**, thinking **on**, then
1. runs the E1 prompts **B and C** on the 2,670 Who&When Pro multi-agent text traces (same prompts, parser and scorer as the luna and MiMo runs),
2. scores E1 (luna + MiMo + Qwen) and re-runs the E2 analysis with Qwen included.

**Nothing to upload.** The code, prompts and the luna and MiMo result logs are embedded in this notebook; the dataset (`text.jsonl`, `taxonomy.yaml`) is downloaded from Hugging Face `Leoxx/whowhen_pro` and checked against the SHA-256 of the copy used for the earlier runs; the official renderers are cloned from GitHub at a pinned commit.

**Before you start**
- **GPU:** a bigger GPU is recommended for speed (A100 40/80 GB or H100). The model needs only about 11 GB of weights; the notebook scales concurrency to the GPU (24 / 64 / 128 parallel requests for 24 GB / 40 GB / 80 GB+). In VS Code: Select Kernel > Colab > New Colab Server > pick the GPU. Without a GPU the first cell stops with a clear message.
- **Settings used (all in the first cells and in `config/models.yaml`):** thinking on; sampling from the Qwen3.5 model card for thinking mode on general tasks (temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, seed 0); max output 32,768 tokens including thinking (the same cap as luna); context 57,344.
- **4-bit disclosure:** only the weights of the linear layers in the transformer blocks are INT4 (GPTQ, group size 128); the linear-attention layers, embeddings and vision encoder stay in BF16 (checkpoint is 11 GB, not 5 GB). Results are for this quantized model, not the BF16 original; the manifest records it.
- **Time and sessions:** outputs are appended to disk and are **resumable**: after a disconnect, re-run the cells up to the server start and then the full-run cell; finished calls are skipped. Without Drive, results live only on the runtime until the last cell zips them, so **download the results zip regularly**.
- Versions were verified to **resolve together** with `uv` for Linux x86_64 / Python 3.12, and the server cell was written from real Colab failures seen with another model (missing `nvcc`/`ninja`, an API-key variable making the server reject requests). It has **not yet been run with Qwen on a GPU**: if a cell fails, the server cell prints the real error lines.
