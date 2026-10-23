"""Projected spend (plan 5.4). calls = N_total x conditions-per-dataset x judges (Exp 1) + N_total x 5 (Exp 2, J1 O1).
Input tokens from the actual rendered prompts (o200k_base approximation). Output tokens from the smoke test
(raw/smoke_calls.jsonl mean tokens_out per call) unless --out-tokens is given. Prices come from config/models.yaml
(USD per 1M tokens; fill from the billing page). python src/estimate_cost.py [--out-tokens N]"""
import argparse, json, sys
from pathlib import Path
import tiktoken, yaml

sys.path.insert(0, str(Path(__file__).parent))
import load_data as ld
from render import build_prompt

ROOT = ld.ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-tokens", type=float); ap.add_argument("--cap-usd", type=float)
    a = ap.parse_args()
    enc = tiktoken.get_encoding("o200k_base")
    cfg = yaml.safe_load((ROOT / "config/datasets.yaml").read_text())["datasets"]
    models = yaml.safe_load((ROOT / "config/models.yaml").read_text())
    judges = ["J1"]
    sm = ROOT / "raw/smoke_calls.jsonl"
    outs = [d["tokens_out"] for d in map(json.loads, open(sm)) if d.get("tokens_out")] if sm.exists() else []
    out_tok = a.out_tokens or (sum(outs) / len(outs) if outs else None)
    if out_tok is None:
        sys.exit("no smoke output-token measurement; run the smoke stage or pass --out-tokens")
    inp = {}
    for n in ("WW_AG", "WW_HC", "WWPRO_TXT"):
        recs = ld.load_normalized(n)
        for c in ("O0", "O1", "O1-A") + (("O2",) if cfg[n]["o2_fields"] else ()):
            inp[(n, c)] = sum(len(enc.encode(build_prompt(r, c, cfg[n]["answer_variant"], cfg[n]["o2_fields"])[0], disallowed_special=())) for r in recs)
            if c == "O1":
                inp[(n, "exp2")] = 5 * inp[(n, c)]
    print(f"mean output tokens per call (incl. reasoning): {out_tok:.0f}")
    total = 0.0
    for j in judges:
        pr = models["judges"][j]["price_usd_per_mtok"]
        if pr["input"] is None:
            print(f"{j}: price missing in config/models.yaml; token totals only"); pr = {"input": 0, "output": 0}
        calls = {k: (len(ld.load_normalized(k[0])) * (5 if k[1] == "exp2" else 1)) for k in inp if (k[1] != "exp2" or j == "J1")}
        tin = sum(v for k, v in inp.items() if k in calls); tout = sum(calls.values()) * out_tok
        usd = (tin * pr["input"] + tout * pr["output"]) / 1e6
        total += usd
        print(f"{j}: calls={sum(calls.values())} input_tokens={tin:,.0f} output_tokens={tout:,.0f} projected_usd={usd:,.2f}")
    print(f"TOTAL projected USD: {total:,.2f}  (plan: STOP and ask the user if > ~$50)")
    if a.cap_usd and total > a.cap_usd:
        print("EXCEEDS the stated cap")


if __name__ == "__main__":
    main()
