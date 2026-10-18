import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from data import load_records
from parser import parse_response
from render import render_A, render_B, render_taxonomy, build_prompt_B, build_prompt_C
from score import score

RECS = load_records()
OK = "Agent Name: coder\nStep Number: 7\nReason for Mistake: x"


def test_count():
    assert len(RECS) == 2670
    assert all(r["gt_agents"] and r["gt_steps"] for r in RECS)


def test_label_agent_at_step():
    for r in RECS:
        if not r["round_level"]:
            assert r["entries"][r["gt_steps"][0]][0] in r["gt_agents"], r["id"]
        else:
            assert any(r["entries"][i][0] in r["gt_agents"] for i in r["gt_steps"]), r["id"]


def test_parser_valid_and_extra_text():
    assert parse_response(OK)["step"] == 7
    assert parse_response("Stage 1...\nblah\n" + OK + "\n")["ok"]


def test_parser_failures():
    assert parse_response("Agent Name: a\nReason for Mistake: r")["failure"] == "missing_step"
    assert parse_response("Step Number: 3\nReason for Mistake: r")["failure"] == "missing_agent"
    assert parse_response("Agent Name: a\nStep Number: 3")["failure"] == "missing_reason"
    for bad in ["3-5", "step 3", "**3**", "3.0", "N/A", ""]:
        assert parse_response(f"Agent Name: a\nStep Number: {bad}\nReason for Mistake: r")["failure"] in ("bad_step",), bad
    assert parse_response("Agent Name: a\nStep Number: 3\nReason for Mistake: r", "length")["failure"] == "truncated"
    assert parse_response(None)["failure"] == "empty_response" and parse_response("   ")["failure"] == "empty_response"


def test_parser_last_wins_and_stage_mentions():
    t = "Agent Name: x\nStep Number: 1\nReason for Mistake: a\nAgent Name: y\nStep Number: 2\nReason for Mistake: b"
    assert parse_response(t)["agent"] == "y" and parse_response(t)["step"] == 2
    t2 = "In Stage 2 I will report the Agent Name: later\nStep 4 is wrong.\n" + OK   # label not at line start in stage text
    assert parse_response(t2)["agent"] == "coder"
    assert not parse_response("**Agent Name:** coder\nStep Number: 7\nReason for Mistake: x")["ok"]  # no markdown fixing


def test_scorer():
    r = {"gt_agents": ["a"], "gt_steps": [3, 4], "n_entries": 10}
    assert score({"ok": True, "agent": "a", "step": 4}, r)["joint_correct"]
    assert not score({"ok": True, "agent": "A", "step": 4}, r)["agent_correct"]  # case-sensitive
    assert score({"ok": False}, r) == {"agent_correct": False, "step_correct": False, "joint_correct": False, "norm_error": None}


def _strip_headers(b, n):
    """Remove the exact header f"step - {i} :\n" for i=0..n-1, sequentially, each exactly once at its own position."""
    pos = 0
    for i in range(n):
        h = f"step - {i} :\n"
        j = b.index(h, pos)
        assert j == pos, (i, "header not at the expected position")  # header sits at the start of its entry
        b = b[:j] + b[j + len(h):]
        # advance past this entry's text up to the separator before the next header
        pos = b.index(f"\n\nstep - {i + 1} :\n", j) + 2 if i + 1 < n else len(b)
    return b


def test_b_equals_a_modulo_headers_and_separator():
    for r in RECS[:: max(1, len(RECS) // 300)]:
        e, n = r["entries"], len(r["entries"])
        texts = [f"{a}: {t}" for a, t in e]
        assert render_A(e) == "\n".join(texts)
        assert _strip_headers(render_B(e), n) == "\n\n".join(texts)           # B minus headers == A with B's separator
        assert render_B(e) == "\n\n".join(f"step - {i} :\n{t}" for i, t in enumerate(texts))
        assert f"step - {n} :" not in render_B(e)                              # exactly N headers, 0..N-1, no gaps


def test_full_prompts_differ_only_by_template_and_headers():
    r = RECS[0]; pb = build_prompt_B(r)
    log_a, log_b = render_A(r["entries"]), render_B(r["entries"])
    assert pb.count(log_b) == 1  # B embeds the labeled log exactly once
    assert build_prompt_C(r).count(log_a) == 1 and log_b not in build_prompt_C(r)


def test_labels_and_leaks_not_in_prompts():
    for r in RECS[:50]:
        for p in (build_prompt_B(r), build_prompt_C(r)):
            for bad in ("mistake_agent", "mistake_step", "is_injected", "ground_truth", "system_prompt", "accepted_predictions"):
                assert bad not in p
    assert "step - 0 :" not in build_prompt_C(RECS[0])  # headers only in B
    assert "step - 0 :\n" in build_prompt_B(RECS[0])


def test_taxonomy():
    t = render_taxonomy(); lines = t.split("\n")
    assert len(lines) == 17 and lines[0].startswith("P.1 - ") and lines[-1].startswith("C.3 - ")
    import yaml
    y = yaml.safe_load((Path(__file__).resolve().parents[1] / "prompts/taxonomy.yaml").read_text())
    for line, (k, v) in zip(lines, y.items()):
        assert line == f"{k} - {v['name']}: " + re.sub(r"\s+", " ", v["description"]).strip()
    assert "{taxonomy}" not in build_prompt_C(RECS[0])


def test_no_unfilled_placeholders():
    for r in RECS[::40]:
        for p in (build_prompt_B(r), build_prompt_C(r)):
            for ph in ("{problem}", "{ground truth}", "{failure log}"):
                assert ph not in p
    nr = next(r for r in RECS if r["ground_truth"] in (None, ""))
    assert "The Answer for the problem is" not in build_prompt_B(nr)
