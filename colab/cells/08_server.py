import subprocess, urllib.request, signal, re, time, os, sys, json, shutil

def gpu_used_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0

def kill_vllm():
    """Kill every leftover vLLM process (a failed start can leave the engine holding GPU memory) and wait for the memory to be freed."""
    me = os.getpid()
    for l in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
        pid, _, args = l.strip().partition(" ")
        if pid.isdigit() and int(pid) != me and re.search(r"vllm serve|EngineCore|VLLM::|vllm\.v1|vllm\.entrypoints", args):
            try: os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError: pass
    for _ in range(40):
        if gpu_used_mb() < 1500: return True
        time.sleep(2)
    return False

VENV = globals().get("VENV", "/content/venv/bin")
LOGDIR = globals().get("LOGDIR", "/content")
# Colab has no nvcc/ninja, so FlashInfer cannot JIT-compile kernels. Install ninja into the venv and switch the FlashInfer sampler off.
uv = shutil.which("uv")
if uv and not os.path.exists(f"{VENV}/ninja"):
    subprocess.run([uv, "pip", "install", "--python", f"{VENV}/python", "ninja"], capture_output=True, text=True)
# NOTE: an earlier cell sets VLLM_API_KEY="EMPTY" for the runner; vLLM treats that variable as a REQUIRED api key (-> HTTP 401), so it is left out of the server env.
ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "PATH": f"{VENV}:{os.environ.get('PATH', '')}"}

base = ["--port", "8000", "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(PRESET["seqs"]),
        "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", json.dumps({"enable_thinking": THINKING})]
LMO = ["--language-model-only"]
UTIL = ["--gpu-memory-utilization", str(PRESET["util"])]
EAGER = ["--enforce-eager", "--gpu-memory-utilization", "0.85"]
# progressively safer settings; an attempt only counts if the server starts AND really generates a completion
ATTEMPTS = [("default (CUDA graphs), text-only", base + LMO + UTIL),
            ("CUDA graphs capped to the batch size (hybrid-model cache limit)", base + LMO + UTIL + ["--max-cudagraph-capture-size", str(PRESET["seqs"])]),
            ("eager mode (no torch.compile / CUDA graphs), text-only", base + LMO + EAGER),
            ("eager mode, multimodal loader, no prefix caching", base + EAGER + ["--no-enable-prefix-caching"])]

def root_cause(path, first=10, last=6):
    lines = open(path, errors="replace").read().splitlines()
    pat = re.compile(r"(Error|ERROR|error:|out of memory|OutOfMemory|CUDA|nvcc|ninja|not supported|Unsupported|Failed|insufficient|No such file|Cannot|cannot|unrecognized arguments)")
    skip = ("File ", "^", "raise ", "return ", "async ", "with ")
    hits = [(i, l) for i, l in enumerate(lines) if pat.search(l) and not l.lstrip().startswith(skip) and "Engine core initialization failed" not in l
            and "See root cause above" not in l and "Capturing CUDA graphs" not in l and "FlashInfer kernels are disabled" not in l]
    print(f"   {len(lines)} log lines; error-like lines (first {first}, last {last}):")
    for i, l in hits[:first]: print(f"   [{i}] {l[:260]}")
    if len(hits) > first + last: print("   ...")
    for i, l in hits[max(first, len(hits) - last):]: print(f"   [{i}] {l[:260]}")
    frames = [l.strip() for l in lines if re.search(r'File ".*/(vllm|flashinfer|torch|triton)/', l)]
    print("   deepest vllm/flashinfer/torch frames:"); [print("    ", f[:230]) for f in frames[-6:]]

def wait_ready(p, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None: return False
        try:
            if urllib.request.urlopen("http://localhost:8000/health", timeout=5).status == 200: return True
        except Exception: pass
        time.sleep(10)
    return False

def generation_ok():
    """Send a real request. Returns (ok, detail)."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "What is 17*23? Answer in one short sentence."}],
                       "max_tokens": 4000, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}).encode()
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:800]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return ("choices" in r), (json.dumps(r)[:800] if "choices" not in r else "")

ok, proc, SERVE_CMD = False, None, None
for k, (name, args) in enumerate(ATTEMPTS, 1):
    freed = kill_vllm()
    print(f"\n=== attempt {k}/{len(ATTEMPTS)}: {name} | GPU memory in use before start: {gpu_used_mb()} MB (cleanup ok: {freed})")
    SERVE_CMD = [f"{VENV}/vllm", "serve", MODEL] + args
    print(" ".join(SERVE_CMD))
    log = f"{LOGDIR}/vllm_attempt{k}.log"
    proc = subprocess.Popen(SERVE_CMD, stdout=open(log, "w"), stderr=subprocess.STDOUT, env={**{k_: v for k_, v in os.environ.items() if k_ != "VLLM_API_KEY"}, **ENV}, start_new_session=True)
    ok = wait_ready(proc)
    if ok:
        ok, detail = generation_ok()
        if ok:
            shutil.copy(log, f"{LOGDIR}/vllm_server.log"); print(f"SERVER READY AND GENERATING (attempt {k}: {name})"); break
        print(f"attempt {k}: server is up but the test request failed: {detail}")
    else:
        print(f"attempt {k}: server did not start.")
    print(f"Root cause from {log}:"); root_cause(log)
    try: os.killpg(proc.pid, signal.SIGKILL)
    except Exception: pass
if not ok: print("\nALL ATTEMPTS FAILED. Send the text printed above (and `!nvidia-smi`) for diagnosis.")
assert ok, "vLLM server did not start and generate; see the root causes printed above"
