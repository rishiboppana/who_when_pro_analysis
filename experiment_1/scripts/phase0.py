"""Phase 0: data recovery, schema, index convention, label sanity, inventory. Writes into experiment_1/config."""
import json, glob, hashlib, os, collections as C, re
from pathlib import Path
ORIG = Path("/private/tmp/claude-501/-Users-user-Desktop-thesis-who-and-when-pro-analysis/2a21fd4a-fc52-4e4f-9e6b-a0bbdd9c277a/scratchpad/orig/Who&When")
LOCAL = Path("/Users/user/Desktop/thesis/error_localization/who_and_when_dataset/Who&When")
def load(base, sub):
    out = {}
    for f in sorted(glob.glob(str(base/sub/"*.json")), key=lambda x:int(re.sub(r"\D","",os.path.basename(x)) or 0)):
        out[os.path.basename(f)] = json.load(open(f))
    return out
res = {}
for sub in ["Algorithm-Generated","Hand-Crafted"]:
    o, l = load(ORIG, sub), load(LOCAL, sub)
    print(sub, "repo files:", len(o), "local copy files:", len(l), "identical json:", o == l)
    res[sub] = o
    keys = C.Counter(); hk = C.Counter()
    for d in o.values():
        keys.update(d.keys()); [hk.update(e.keys()) for e in d["history"]]
    print("  record keys:", dict(keys)); print("  history entry keys:", dict(hk))
    bad = [(k, d["mistake_step"], len(d["history"])) for k,d in o.items() if int(d["mistake_step"]) >= len(d["history"])]
    print("  mistake_step >= len(history):", bad)
    has_step = sum(1 for d in o.values() if all("step" in e for e in d["history"]))
    ok = sum(1 for d in o.values() if all(e.get("step")==f"step - {i}" for i,e in enumerate(d["history"]) if "step" in e))
    print(f"  records where every entry has a 'step' field: {has_step}; entries' step==f'step - i' in {ok}")
    print("  mistake_step type:", C.Counter(type(d['mistake_step']).__name__ for d in o.values()))
# label-sanity for HC_025/039/050 (file names in the repo)
hc = res["Hand-Crafted"]
for n in (25, 39, 50):
    d = hc.get(f"{n}.json")
    print(f"HC_{n:03d}", None if d is None else (d["mistake_agent"], d["mistake_step"], len(d["history"])))
