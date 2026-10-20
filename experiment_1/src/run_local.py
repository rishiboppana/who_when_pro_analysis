"""Run E1 prompts B and C against a local vLLM OpenAI-compatible server (Google Colab).
Same runner, prompts, parser and scorer as run.py; the only differences are (a) sampling parameters come from
config/models.yaml (`sampling`) and (b) no .env is needed.   usage:
  python src/run_local.py --stage smoke|full --models qwen --prompts B C --workers 32"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run


def load_env():
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")


def call_model(client, m, prompt):
    p = {"model": m["model_id"], "messages": [{"role": "user", "content": prompt}], "max_tokens": m["max_output_tokens"],
         "stream": True, "stream_options": {"include_usage": True}}              # single user message, no system prompt
    s = m.get("sampling") or {}
    for k in ("temperature", "top_p", "presence_penalty", "seed"):
        if k in s:
            p[k] = s[k]
    extra = {k: s[k] for k in ("top_k", "min_p", "repetition_penalty") if k in s}   # vLLM-specific sampling fields
    if extra:
        p["extra_body"] = extra
    text, finish, usage, returned, reasoning_chars = [], None, None, None, 0
    for ch in client.chat.completions.create(**p):
        returned = ch.model or returned
        usage = ch.usage or usage
        for c in ch.choices:
            if c.delta.content:                                                   # final answer only; thoughts arrive separately
                text.append(c.delta.content)
            r_ = getattr(c.delta, "reasoning_content", None) or getattr(c.delta, "reasoning", None)
            if r_:
                reasoning_chars += len(r_)
            finish = c.finish_reason or finish
    det = usage.completion_tokens_details if usage else None
    return {"text": "".join(text), "finish_reason": finish, "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None, "returned_model": returned,
            "reasoning_chars": reasoning_chars, "sampling": s}


run.load_env = load_env
run.call_model = call_model
if __name__ == "__main__":
    run.main()
