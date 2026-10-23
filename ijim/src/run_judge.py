"""Judge runner (plan 7.1 / 8.1). Resumable, append-only raw/calls.jsonl written BEFORE parsing.
  python src/run_judge.py --stage smoke --judges J1
  python src/run_judge.py --stage exp1 --judges J1 [--datasets WW_AG ...] [--workers 8]
  python src/run_judge.py --stage exp2 --judges J1 [--workers 8]          (O1 only, replicates 1..5)
Rules enforced here: one user message, no system prompt; a received response is final; only transport errors
(5xx, timeout, connection reset, rate limit) are retried, 3 times, identical parameters; longest prompts first;
within a dataset O0 runs before O1, O1-A, O2; auto-stop if a (judge, condition) parse-failure rate exceeds 20% after 30 calls.
Smoke stage never reads correctness. Experiment-1 replicate = 0; Experiment-2 replicates = 1..5 (never reuses Experiment 1)."""
import argparse, asyncio, hashlib, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import openai, yaml

sys.path.insert(0, str(Path(__file__).parent))
import load_data as ld
from parse import parse_response
from render import build_prompt

ROOT = ld.ROOT
RAW = ROOT / "raw"
CFG = yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text())
MODELS = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
TRANSPORT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
ORDER = ["O0", "O1", "O1-A", "O2"]


def load_env():
    p = ROOT.parent / ".env"
    if p.exists():
        for l in p.read_text().splitlines():
            if "=" in l and not l.lstrip().startswith("#"):
                k, _, v = l.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cache_key(dataset, traj_id, condition, judge, replicate):
    return hashlib.sha256(f"{dataset}|{traj_id}|{condition}|{judge}|{replicate}".encode()).hexdigest()


def conditions_for(dataset):
    return [c for c in ORDER if c != "O2" or CFG["datasets"][dataset]["o2_fields"]]


async def call_model(client, m, prompt):
    if m["api"] == "openai_responses":
        r = await client.responses.create(model=m["model_id"], input=prompt, max_output_tokens=m["max_output_tokens"], reasoning=m["reasoning"])
        u = r.usage
        fr = r.status if r.status != "incomplete" else f"incomplete:{getattr(r.incomplete_details, 'reason', None)}"
        return {"text": r.output_text, "finish_reason": fr, "response_id": r.id, "model_snapshot": r.model,
                "tokens_in": u.input_tokens, "tokens_out": u.output_tokens, "tokens_reasoning": u.output_tokens_details.reasoning_tokens}
    text, finish, usage, returned, rid = [], None, None, None, None
    stream = await client.chat.completions.create(model=m["model_id"], messages=[{"role": "user", "content": prompt}], max_tokens=m["max_output_tokens"],
                                                  stream=True, stream_options={"include_usage": True}, **({"extra_body": m["extra_body"]} if m.get("extra_body") else {}))
    async for ch in stream:
        returned, rid, usage = ch.model or returned, ch.id or rid, ch.usage or usage
        for c in ch.choices:
            if c.delta.content:
                text.append(c.delta.content)
            finish = c.finish_reason or finish
    det = usage.completion_tokens_details if usage else None
    return {"text": "".join(text), "finish_reason": finish, "response_id": rid, "model_snapshot": returned,
            "tokens_in": usage.prompt_tokens if usage else None, "tokens_out": usage.completion_tokens if usage else None,
            "tokens_reasoning": getattr(det, "reasoning_tokens", None) if det else None}


async def run_one(client, m, prompt):
    waits, n, t0 = m.get("retry_waits", [2, 4, 8]), 0, time.time()
    while True:
        try:
            out = await call_model(client, m, prompt)
            exp = m.get("expect_model")
            if exp and (out.get("model_snapshot") or "").lower() != exp.lower():
                return out | {"text": None, "finish_reason": "wrong_model", "error": f"wrong_model_served: {out.get('model_snapshot')}", "latency_s": round(time.time() - t0, 2), "transport_retries": n}
            return out | {"error": None, "latency_s": round(time.time() - t0, 2), "transport_retries": n}
        except TRANSPORT as e:
            if n >= len(waits):
                return {"text": None, "finish_reason": "transport_error", "error": f"{type(e).__name__}: {str(e)[:300]}", "latency_s": round(time.time() - t0, 2), "transport_retries": n}
            n += 1
            await asyncio.sleep(waits[n - 1])
        except openai.APIStatusError as e:
            return {"text": None, "finish_reason": "api_error", "error": f"{e.status_code}: {str(e)[:500]}", "latency_s": round(time.time() - t0, 2), "transport_retries": n}


def make_client(m):
    return openai.AsyncOpenAI(base_url=m["base_url"], api_key=os.environ[m["api_key_env"]], max_retries=0,
                              timeout=openai.Timeout(1200, connect=30, read=m.get("timeout_read", 1200)))


def done_keys(path):
    if not path.exists():
        return set()
    return {d["cache_key"] for d in map(json.loads, open(path)) if not d.get("error")}


def prepare(dataset, recs, condition):
    """-> list of (rec, prompt, aliases, n_repl) sorted longest prompt first (character length is the sort key)."""
    cfg = CFG["datasets"][dataset]
    rows = []
    for r in recs:
        p, al, n = build_prompt(r, condition, cfg["answer_variant"], cfg["o2_fields"])
        rows.append((r, p, al, n))
    return sorted(rows, key=lambda x: -len(x[1]))


def smoke_ids(all_recs):
    """6 longest (by event text length) + 6 seeded random of the rest, across datasets."""
    flat = [(len(r["question"]) + sum(len(str(e["content"])) for e in r["events"]), r["dataset"], r["traj_id"]) for rs in all_recs.values() for r in rs]
    flat.sort(key=lambda x: (-x[0], x[2]))
    longest, rest = flat[:6], sorted(flat[6:], key=lambda x: x[2])
    rnd = random.Random(CFG["analysis_seed"]).sample(rest, 6)
    return [(d, t) for _, d, t in longest + rnd]


async def main_async(a):
    load_env()
    judges = a.judges
    for j in judges:
        if j == "J2" and not MODELS.get("ENABLE_J2"):
            sys.exit("J2 is disabled (ENABLE_J2: false in config/models.yaml)")
        if os.environ.get(MODELS["judges"][j]["api_key_env"]) is None:
            sys.exit(f"missing environment variable {MODELS['judges'][j]['api_key_env']}")
    names = a.datasets or ["WW_AG", "WW_HC", "WWPRO_TXT"]
    recs = {n: ld.load_normalized(n) for n in names}
    RAW.mkdir(exist_ok=True)
    path = RAW / ("smoke_calls.jsonl" if a.stage == "smoke" else "calls.jsonl")
    lock, stop = asyncio.Lock(), {"flag": False}
    stats = {}

    # plan: list of phases; a phase = (dataset, condition, replicate, [rows])
    phases = []
    if a.stage == "smoke":
        ids = smoke_ids(recs)
        (ROOT / "config" / "smoke_ids.json").write_text(json.dumps(ids, indent=1))
        for d in names:
            sel = {t for dd, t in ids if dd == d}
            if sel:
                phases.append((d, "O1", 0, prepare(d, [r for r in recs[d] if r["traj_id"] in sel], "O1")))
    elif a.stage == "exp1":
        for d in names:
            for c in (a.conditions or conditions_for(d)):
                phases.append((d, c, 0, prepare(d, recs[d], c)))
    else:
        for rep in range(1, 6):
            for d in names:
                phases.append((d, "O1", rep, prepare(d, recs[d], "O1")))

    for jname in judges:
        m = MODELS["judges"][jname]
        client = make_client(m)
        done = done_keys(path)
        sem = asyncio.Semaphore(a.workers)
        price = m.get("price_usd_per_mtok") or {}
        spend = {"usd": 0.0}

        async def work(d, c, rep, r, prompt, aliases, nrep):
            key = cache_key(d, r["traj_id"], c, jname, rep)
            if key in done or stop["flag"]:
                return
            async with sem:
                if stop["flag"]:
                    return
                res = await run_one(client, m, prompt)
            row = {"cache_key": key, "dataset": d, "traj_id": r["traj_id"], "cluster_id": r["cluster_id"], "condition": c, "judge": jname,
                   "replicate": rep, "stage": a.stage, "timestamp": datetime.now(timezone.utc).isoformat(), "model_id": m["model_id"],
                   "request": {"max_output_tokens": m["max_output_tokens"], "reasoning": m.get("reasoning"), "seed": None, "system_prompt": None},
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_chars": len(prompt), "L": r["L"],
                   "aliases": aliases, "alias_replacements": nrep} | res
            async with lock:
                with path.open("a") as fh:                       # written BEFORE parsing
                    fh.write(json.dumps(row) + "\n")
                pr = None if res.get("error") else parse_response(res.get("text"), res.get("finish_reason"))
                s = stats.setdefault((jname, c), {"n": 0, "bad": 0})
                s["n"] += 1
                s["bad"] += bool(res.get("error") or (pr and not pr["ok"]))
                if price.get("input") and res.get("tokens_in") is not None:
                    spend["usd"] += (res["tokens_in"] * price["input"] + (res["tokens_out"] or 0) * price["output"]) / 1e6
                if a.budget_usd and spend["usd"] > a.budget_usd:
                    stop["flag"] = True; print(f"STOP: budget ${a.budget_usd} exceeded", flush=True)
                if a.stage != "smoke" and s["n"] >= 30 and s["bad"] / s["n"] > 0.20:
                    stop["flag"] = True; print(f"STOP: parse-failure/error rate {s['bad']}/{s['n']} > 20% for {jname}/{c}", flush=True)
                print(f"[{jname}/{d}/{c}/r{rep}] {r['traj_id']} fin={res['finish_reason']} out={res.get('tokens_out')} lat={res['latency_s']}s "
                      f"err={res['error']} ok={None if pr is None else pr['ok']} ${spend['usd']:.2f}", flush=True)

        for d, c, rep, rows in phases:
            await asyncio.gather(*(work(d, c, rep, r, p, al, n) for r, p, al, n in rows))
            if stop["flag"]:
                print("STOPPED early; report to the user", flush=True)
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "exp1", "exp2"], required=True)
    ap.add_argument("--judges", nargs="+", default=["J1"])
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--conditions", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
