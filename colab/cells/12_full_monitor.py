# Monitor (re-run any time). Shows progress, error rows and parse failures.
import glob, pandas as pd
from parser import parse_response
def progress():
    out = {}
    for p in ["B", "C"]:
        f = f"{E1}/outputs/raw/qwen/{p}.jsonl"
        rows = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
        ok = [r for r in rows if not r.get("error")]
        out[p] = {"done": len({r["trajectory_id"] for r in ok}), "of": 2670, "error_rows": len(rows) - len(ok),
                  "parse_fail": sum(1 for r in ok if not parse_response(r["text"], r["finish_reason"])["ok"]),
                  "truncated": sum(1 for r in ok if r["finish_reason"] == "length"), "out_tokens": sum(r.get("output_tokens") or 0 for r in ok)}
    return out
P1 = progress(); t1 = time.time(); print(pd.DataFrame(P1).T)
print("processes alive:", [p.poll() is None for p in procs] if "procs" in globals() else None, "| server alive:", proc.poll() is None)
