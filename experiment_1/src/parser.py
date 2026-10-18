"""Frozen primary parser (plan section 7), identical for B and C. Final response text only.
Lines starting with the exact labels `Agent Name:`, `Step Number:`, `Reason for Mistake:` (case-sensitive, line start);
the LAST occurrence of each is used. All three required; Step Number must be a single integer after trimming
whitespace; Agent Name non-empty. No markdown fixing, no spelling fixing. Truncated/empty responses are failures."""
import re

LENGTH_FINISH = {"length", "max_tokens", "max_output_tokens", "max_completion_tokens"}


def _last(lines, label):
    hit = None
    for l in lines:
        if l.startswith(label):
            hit = l[len(label):]
    return hit


def parse_response(text, finish_reason=None):
    fail = lambda c: {"ok": False, "agent": None, "step": None, "failure": c}
    fr = (finish_reason or "").lower()
    if fr in LENGTH_FINISH or fr.startswith("incomplete"):
        return fail("truncated")
    if text is None or not text.strip():
        return fail("empty_response")
    lines = text.splitlines()
    a, s, r = _last(lines, "Agent Name:"), _last(lines, "Step Number:"), _last(lines, "Reason for Mistake:")
    if a is None or not a.strip():
        return fail("missing_agent")
    if s is None:
        return fail("missing_step")
    if r is None:
        return fail("missing_reason")
    if not re.fullmatch(r"\d+", s.strip()):
        return fail("bad_step")
    return {"ok": True, "agent": a.strip(), "step": int(s.strip()), "failure": None}
