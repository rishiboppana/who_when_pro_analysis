DATA_BASE = "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/main"
os.makedirs(f"{PROJECT}/data", exist_ok=True)
EXPECT = {"data/text.jsonl": "@@TEXT_SHA@@", "taxonomy.yaml": "@@TAX_SHA@@"}   # SHA-256 of the copies used for the luna and MiMo runs
for remote, local in (("data/text.jsonl", "data/text.jsonl"), ("taxonomy.yaml", "data/taxonomy.yaml")):
    dest = f"{PROJECT}/{local}"
    if not os.path.exists(dest):
        subprocess.run(["curl", "-sL", "--fail", "-o", dest, f"{DATA_BASE}/{remote}"], check=True)
    got = hashlib.sha256(open(dest, "rb").read()).hexdigest()
    assert got == EXPECT[remote], f"{remote}: SHA-256 mismatch ({got}); the dataset on the Hub changed. Stop and tell the author."
    print("OK", local, got[:16])
n = sum(1 for _ in open(f"{PROJECT}/data/text.jsonl")); assert n == @@TEXT_LINES@@, n
print("text.jsonl rows:", n)
shutil.copy(f"{PROJECT}/data/taxonomy.yaml", f"{E1}/prompts/taxonomy.yaml")      # the 17-mode taxonomy used by prompt C
