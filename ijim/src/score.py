"""Scorer (plan 4.4).
Agent: original evaluate.py rule. The predicted agent is the first `[\\w_]+` token after `Agent Name:` and the run is correct
iff the labelled agent string is a SUBSTRING of that token (case-sensitive), identical in all conditions. For O1-A the
predicted `Agent X` alias is first mapped back to the original name; an unknown alias is a wrong agent, not a parse failure.
Pro frameworks may label several acceptable agents: correct iff any matches.
Step: parsed integer equals true_step_native. Joint = both. Parse failure = incorrect on all three.
Normalized step error = |pred - true| / max(L - 1, 1), parseable runs only."""
import re

_ALIAS = re.compile(r"\s*Agent ([A-Z]{1,3})\b")
_TOKEN = re.compile(r"\s*([\w_]+)")


def map_back(pred_agent, aliases):
    """aliases: {'Agent A': original_name}. Returns the original name, or the text unchanged if it is not an alias."""
    if not aliases:
        return pred_agent
    m = _ALIAS.match(pred_agent)
    if m and f"Agent {m.group(1)}" in aliases:
        return aliases[f"Agent {m.group(1)}"]
    return pred_agent


def match_agent(true_agents, pred_agent):
    m = _TOKEN.match(pred_agent or "")
    if not m:
        return False
    tok = m.group(1)
    return any(t in tok for t in true_agents)


def match_step(true_step, pred_step):
    return pred_step == true_step


def score_run(parsed, rec, aliases=None):
    if not parsed["ok"]:
        return {"agent_ok": False, "step_ok": False, "joint_ok": False, "parse_fail": True, "norm_err": None}
    a = match_agent(rec["true_agents"], map_back(parsed["agent"], aliases))
    s = match_step(rec["true_step_native"], parsed["step"])
    return {"agent_ok": a, "step_ok": s, "joint_ok": a and s, "parse_fail": False,
            "norm_err": abs(parsed["step"] - rec["true_step_native"]) / max(rec["L"] - 1, 1)}
