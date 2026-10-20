procs = []
for p in ["B", "C"]:
    log = open(f"{E1}/outputs/full_qwen_{p}.log", "a")
    procs.append(subprocess.Popen([f"{VENV}/python", "src/run_local.py", "--stage", "full", "--models", "qwen", "--prompts", p, "--workers", str(max(1, PRESET["seqs"] // 2))],
                                  cwd=E1, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()))
print("started", [p.pid for p in procs])
