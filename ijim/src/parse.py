"""Frozen parser (plan 4.4): the first `Agent Name:` line and the first `Step Number:` line (case-insensitive label at
line start, leading whitespace allowed). Agent non-empty; step a base-10 integer and nothing else. Anything else, and any
truncated / incomplete response, is a parse failure. No markdown repair, no re-prompting."""
import re

LENGTH_FINISH = {"length", "max_tokens", "max_output_tokens", "max_completion_tokens"}
_A = re.compile(r"^\s*agent name\s*:(.*)$", re.I)
_S = re.compile(r"^\s*step number\s*:(.*)$", re.I)


def parse_response(text, finish_reason=None):
    fail = lambda c: {"ok": False, "agent": None, "step": None, "failure": c}
    fr = (finish_reason or "").lower()
    if fr in LENGTH_FINISH or fr.startswith("incomplete"):
        return fail("truncated")
    if text is None or not text.strip():
        return fail("empty_response")
    agent = step = None
    for line in text.splitlines():
        if agent is None:
            m = _A.match(line)
            if m:
                agent = m.group(1).strip()
        if step is None:
            m = _S.match(line)
            if m:
                step = m.group(1).strip()
        if agent is not None and step is not None:
            break
    if not agent:
        return fail("missing_agent")
    if step is None:
        return fail("missing_step")
    if not re.fullmatch(r"[0-9]+", step):
        return fail("bad_step")
    return {"ok": True, "agent": agent, "step": int(step), "failure": None}
