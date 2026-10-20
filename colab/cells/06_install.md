## 6. Install (isolated venv, so Colab's own packages are not touched)
Pins are in `colab/requirements-colab.txt`. Key constraints: **vLLM 0.32.0 needs `transformers>=5.16.1,<5.20.0`** (the newest transformers does not work), `numpy<2.4`, `huggingface_hub 1.x`. `uv --torch-backend=auto` picks the torch/CUDA wheel that matches the Colab driver. About 5-10 minutes.
