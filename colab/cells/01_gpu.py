import subprocess, json, os, re, time, sys
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True)
except FileNotFoundError:
    raise SystemExit("This Colab runtime has NO GPU (nvidia-smi not found). Disconnect and reconnect with a GPU machine: "
                     "in the Colab web page use Runtime > Change runtime type > pick a GPU; in VS Code use Select Kernel > Colab > New Colab Server and pick a GPU machine type "
                     "(an A100 or H100 is recommended for speed; a T4 16 GB is too small for a comfortable run).")
print(q.stdout or q.stderr)
assert q.returncode == 0 and q.stdout.strip(), "nvidia-smi failed; no usable GPU on this runtime."
GPU_NAME, MEM_MB, DRIVER, CC = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
MEM_GB = float(MEM_MB) / 1024
print(f"GPU {GPU_NAME} | {MEM_GB:.0f} GB | driver {DRIVER} | compute capability {CC} | python {sys.version.split()[0]}")

# 4-bit checkpoint (vLLM reads the quantization from its config). Alternatives, if this one fails to load:
#   "QuantTrio/Qwen3.5-9B-AWQ"  (AWQ 4-bit)   or   "cyankiwi/Qwen3.5-9B-AWQ-4bit"  (AWQ via compressed-tensors)
MODEL = "RedHatAI/Qwen3.5-9B-quantized.w4a16"
THINKING = True                          # E1 setting for Qwen: thinking ON; one setting for prompts B and C
MAX_MODEL_LEN = 57344                    # longest E1 prompt is 23,927 tokens + 32,768 max output tokens
KV_DTYPE = "fp8"                         # fp8 KV cache (default scales 1.0, uncalibrated): roughly doubles KV capacity so more requests run at once. Set None for the default dtype. Disclose in the write-up.
# concurrency: weights are about 11 GB and the hybrid architecture (24 of 32 layers are linear attention) has a very small KV cache
if MEM_GB >= 75:   PRESET = dict(util=0.90, seqs=128)
elif MEM_GB >= 38: PRESET = dict(util=0.90, seqs=64)
elif MEM_GB >= 22: PRESET = dict(util=0.90, seqs=24)
else: raise SystemExit(f"{MEM_GB:.0f} GB is too small; use a GPU with at least 24 GB (an A100 or H100 is better).")
print("PRESET:", PRESET, "| model:", MODEL, "| thinking:", THINKING)
