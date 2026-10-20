import requests
resp = requests.post("http://localhost:8000/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "user", "content": "What is 17*23? Answer in one short sentence."}],
                     "max_tokens": 4000, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, timeout=900)
r = resp.json()
if "choices" not in r:                      # show the server's error instead of a bare KeyError
    print("HTTP", resp.status_code, "\n", json.dumps(r)[:1500], "\n--- server log tail ---\n", open("/content/vllm_server.log").read()[-3000:])
    raise SystemExit("The server answered with an error, see above.")
msg = r["choices"][0]["message"]
print("content   :", msg.get("content"))
print("reasoning :", ((msg.get("reasoning_content") or msg.get("reasoning") or "")[:300]) or "(none)")
print("usage     :", r["usage"])
assert msg.get("content") and "<think>" not in msg["content"] and "</think>" not in msg["content"], "thoughts leaked into the scored text; check --reasoning-parser qwen3"
if THINKING: assert (msg.get("reasoning_content") or msg.get("reasoning")), "thinking is ON but no reasoning came back"
