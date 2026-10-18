"""Exact-match scorer. Agent: parsed agent equals a labelled agent (exact, case-sensitive, trimmed). Step: parsed step
is a labelled flat index (debate/dylan label a round, so any turn inside the round is correct, as in the official
scorer). Parse failure = incorrect. Normalised error is computed over parsed responses only."""


def score(parsed, rec):
    if not parsed["ok"]:
        return {"agent_correct": False, "step_correct": False, "joint_correct": False, "norm_error": None}
    a = parsed["agent"] in rec["gt_agents"]
    s = parsed["step"] in rec["gt_steps"]
    d = min(abs(parsed["step"] - g) for g in rec["gt_steps"])
    return {"agent_correct": a, "step_correct": s, "joint_correct": a and s, "norm_error": d / rec["n_entries"]}
