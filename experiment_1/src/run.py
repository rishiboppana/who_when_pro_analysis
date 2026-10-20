"""usage: python src/run.py --stage smoke|full --models luna mimo qwen --prompts B C [--workers 4] [--budget-usd X]
Outputs: outputs/smoke/{model}/{prompt}.jsonl or outputs/raw/{model}/{prompt}.jsonl (append-only, resumable).
Each (model, prompt, trajectory) is run exactly once; only transport errors are retried (3x, identical params)."""
import argparse, hashlib, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import openai, yaml

sys.path.insert(0, str(Path(__file__).parent))
from data import load_records
from render import build_prompt_B, build_prompt_C
from parser import parse_response

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {"B": build_prompt_B, "C": build_prompt_C}
TRANSPORT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
lock = threading.Lock()


def load_env():
    for l in (ROOT.parent / ".env").read_text().splitlines():
        if "=" in l and not l.lstrip().startswith("#"):
            k, _, v = l.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def make_client(m):
    return openai.OpenAI(base_url=m["base_url"], api_key=os.environ[m["api_key_env"]], max_retries=0,
                         timeout=openai.Timeout(1200, connect=30, read=300))


def call_model(client, m, prompt):
    if m["api"] == "openai_responses":
        p = {"model": m["model_id"], "input": prompt, "max_output_tokens": m["max_output_tokens"], "reasoning": m["reasoning"]}
        r = client.responses.create(**p); u = r.usage
        fr = r.status if r.status != "incomplete" else f"incomplete:{getattr(r.incomplete_details, 'reason', None)}"
        return {"text": r.output_text, "finish_reason": fr, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                "reasoning_tokens": u.output_tokens_details.reasoning_tokens, "returned_model": r.model}
    p = {"model": m["model_id"], "messages": [{"role": "user", "content": prompt}], "max_tokens": m["max_output_tokens"],
         "stream": True, "stream_options": {"include_usage": True}}
    text, finish, usage, returned = [], None, None, None
    for ch in client.chat.completions.create(**p):
        returned = ch.model or returned
        usage = ch.usage or usage
        for c in ch.choices:
            if c.delta.content:
                text.append(c.delta.content)
            finish = c.finish_reason or finish
    det = usage.completion_tokens_details if usage else None
    return {"text": "".join(text), "finish_reason": finish, "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None, "returned_model": returned}


def run_one(client, key, m, rec, prompt, pid, retries=3):
    t0, n, base = time.time(), 0, {"trajectory_id": rec["id"], "model_key": key, "prompt_id": pid,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "timestamp": datetime.now(timezone.utc).isoformat()}
    while True:
        try:
            return base | call_model(client, m, prompt) | {"latency_s": round(time.time() - t0, 2), "transport_retries": n, "error": None}
        except TRANSPORT as e:
            if n >= retries:
                return base | {"text": None, "finish_reason": "transport_error", "error": f"{type(e).__name__}: {str(e)[:300]}",
                               "latency_s": round(time.time() - t0, 2), "transport_retries": n}
            n += 1; time.sleep(2 ** n)
        except openai.APIStatusError as e:
            return base | {"text": None, "finish_reason": "api_error", "error": f"{e.status_code}: {str(e)[:500]}",
                           "latency_s": round(time.time() - t0, 2), "transport_retries": n}


def smoke_ids(recs):
    """6 longest (by the larger of the B/C prompt token counts) + 6 random with seed 0 from the rest."""
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    tok = {r["id"]: max(len(enc.encode(f(r), disallowed_special=())) for f in BUILDERS.values()) for r in recs}
    ranked = sorted(tok, key=lambda i: (-tok[i], i))
    return ranked[:6] + random.Random(0).sample(sorted(ranked[6:]), 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "full"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=["B", "C"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=None)
    a = ap.parse_args()
    load_env()
    models = yaml.safe_load((ROOT / "config/models.yaml").read_text())["models"]
    pricing = yaml.safe_load((ROOT / "config/pricing.yaml").read_text())
    recs = load_records(); by_id = {r["id"]: r for r in recs}
    if a.stage == "smoke":
        f = ROOT / "config/smoke_ids.txt"
        if not f.exists():
            f.write_text("\n".join(smoke_ids(recs)) + "\n")
        ids, outdir = f.read_text().split(), ROOT / "outputs/smoke"
    else:
        ids, outdir = sorted(by_id), ROOT / "outputs/raw"
    for key in a.models:
        m, client = models[key], make_client(models[key])
        for pid in a.prompts:
            path = outdir / key / f"{pid}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
            done = {d["trajectory_id"] for d in map(json.loads, path.open()) if not d.get("error")} if path.exists() else set()  # failed (transport/api) rows are not "done"
            order = list(ids)
            if a.stage == "full":
                random.Random(0).shuffle(order)
            todo = [i for i in order if i not in done]
            pi, po = pricing[key]["input"], pricing[key]["output"]
            st = {"usd": 0.0, "stop": False, "n": 0, "nonfmt": 0, "fmt": 0}

            def work(i):
                if st["stop"]:
                    return
                rec = by_id[i]
                res = run_one(client, key, m, rec, BUILDERS[pid](rec), pid)
                pr = None if res.get("error") else parse_response(res.get("text"), res.get("finish_reason"))
                with lock:
                    with path.open("a") as fh:
                        fh.write(json.dumps(res) + "\n")
                    if pi is not None and res.get("input_tokens") is not None and res.get("output_tokens") is not None:
                        st["usd"] += (res["input_tokens"] * pi + res["output_tokens"] * po) / 1e6
                        if a.budget_usd and st["usd"] > a.budget_usd:
                            st["stop"] = True
                    st["n"] += 1
                    if res.get("error") or (pr and pr["failure"] == "truncated"):
                        st["nonfmt"] += 1
                    elif pr and not pr["ok"]:
                        st["fmt"] += 1
                    if a.stage == "full" and st["n"] >= 40 and (st["nonfmt"] + st["fmt"]) / st["n"] > 0.25:
                        st["stop"] = True; print(f"[{key}/{pid}] STOP: >25% failures after {st['n']} calls", flush=True)
                    print(f"[{key}/{pid}] {i} fin={res['finish_reason']} out={res.get('output_tokens')} lat={res['latency_s']}s "
                          f"err={res['error']} ${st['usd']:.3f} n={st['n']} nonfmt={st['nonfmt']} fmt={st['fmt']}", flush=True)

            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(work, todo))
            if st["stop"]:
                print(f"[{key}/{pid}] STOPPED early (guard or budget)")


if __name__ == "__main__":
    main()
