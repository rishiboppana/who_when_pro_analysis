"""Trace conditions O0 / O1 / O2 / O1-A (plan 4.2) and prompt assembly. The judge prompt text is identical across
conditions; only {failure log} changes. Placeholders are filled by single-pass plain replacement (the text has braces).
Only question, final_answer_or_null and events are read: no label field is ever accessible here."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {v: (ROOT / "prompts" / f"judge_{v}.txt").read_text().rstrip("\n") for v in ("with_answer", "no_answer")}
CONDITIONS = ("O0", "O1", "O1-A", "O2")


def event_line(e, speaker=None):
    """Original all_at_once: '<speaker>: <content>' (speaker = `name` for AG, `role` for HC)."""
    return f"{e['agent_raw'] if speaker is None else speaker}: {e['content']}"


def render_O0(events):
    return "\n".join(event_line(e) for e in events)


def step_label(i, base):
    return f"[Step {i + base}]"


def render_O1(events, base=0):
    return "\n".join(f"{step_label(i, base)}\n{event_line(e)}" for i, e in enumerate(events))


def strip_O1(text, L, base=0):
    """Inverse of render_O1 (removes the added labels). Used by tests: result must equal render_O0 byte for byte."""
    pos, parts = 0, []
    for i in range(L):
        lab = step_label(i, base) + "\n"
        assert text.startswith(lab, pos), f"missing label for event {i}"
        pos += len(lab)
        if i < L - 1:
            nxt = text.index("\n" + step_label(i + 1, base) + "\n", pos)
            parts.append(text[pos:nxt]); pos = nxt + 1
        else:
            parts.append(text[pos:])
    return "\n".join(parts)


def render_O2(events, fields, base=0):
    """O1 plus one suffix line per event with only the listed native fields (fixed key order, none for missing values)."""
    out = []
    for i, e in enumerate(events):
        lines = [step_label(i, base), event_line(e)]
        kv = [f"{k}={e['native_fields'][k]}" for k in fields if e["native_fields"].get(k) is not None]
        if kv:
            lines.append("(" + "; ".join(kv) + ")")
        out.append("\n".join(lines))
    return "\n".join(out)


def alias_names(n):
    names = []
    for k in range(n):
        s, x = "", k
        while True:
            s = chr(65 + x % 26) + s
            x = x // 26 - 1
            if x < 0:
                break
        names.append(f"Agent {s}")
    return names


def render_O1A(events, base=0):
    """-> (text, alias_to_name, n_replacements_in_content). Aliases by first appearance of agent_raw; the speaker label
    and whole-word occurrences of each name inside event content are replaced in one pass (longest name first)."""
    order = []
    for e in events:
        if e["agent_raw"] not in order:
            order.append(e["agent_raw"])
    to_alias = dict(zip(order, alias_names(len(order))))
    pat = re.compile("(?<!\\w)(" + "|".join(re.escape(n) for n in sorted(order, key=len, reverse=True)) + ")(?!\\w)") if order else None
    count = [0]

    def sub(m):
        count[0] += 1
        return to_alias[m.group(1)]
    out = []
    for i, e in enumerate(events):
        content = pat.sub(sub, str(e["content"])) if pat else e["content"]
        out.append(f"{step_label(i, base)}\n{to_alias[e['agent_raw']]}: {content}")
    return "\n".join(out), {a: n for n, a in to_alias.items()}, count[0]


def render_log(rec, condition, o2_fields=()):
    """-> (failure_log_text, aliases_or_None, n_alias_replacements)."""
    ev, base = rec["events"], rec.get("step_base", 0)
    if condition == "O0":
        return render_O0(ev), None, 0
    if condition == "O1":
        return render_O1(ev, base), None, 0
    if condition == "O2":
        return render_O2(ev, o2_fields, base), None, 0
    if condition == "O1-A":
        t, al, n = render_O1A(ev, base)
        return t, al, n
    raise ValueError(condition)


def fill(template, problem, answer, log):
    vals = {"{problem}": str(problem), "{ground truth}": str(answer), "{failure log}": log}
    return re.sub(r"\{problem\}|\{ground truth\}|\{failure log\}", lambda m: vals[m.group(0)], template)


def build_prompt(rec, condition, answer_variant, o2_fields=()):
    """-> (prompt, aliases_or_None, n_alias_replacements). answer_variant in {with_answer, no_answer}."""
    log, aliases, n = render_log(rec, condition, o2_fields)
    return fill(TEMPLATES[answer_variant], rec["question"], rec["final_answer_or_null"], log), aliases, n
