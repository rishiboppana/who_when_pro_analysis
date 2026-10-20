!pip -q install uv
!uv venv /content/venv --python 3.12
!uv pip install --python /content/venv/bin/python --torch-backend=auto -r {PROJECT}/colab/requirements-colab.txt
!/content/venv/bin/python -c "import torch, vllm, transformers, huggingface_hub, openai, numpy, pandas; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available()); print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| huggingface_hub', huggingface_hub.__version__, '| openai', openai.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__)"
