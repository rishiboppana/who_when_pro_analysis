import inspect, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import features_exp2 as fx, gate_exp2 as gt, load_data as ld, stats_exp1 as st
from parse import parse_response
from render import (alias_names, build_prompt, render_O0, render_O1, render_O1A, render_O2, strip_O1, TEMPLATES)
from score import map_back, match_agent, score_run

DATA = {n: (ld.NORM_DIR / f"{n}.jsonl") for n in ("WW_AG", "WW_HC", "WWPRO_TXT")}
have_data = all(p.exists() for p in DATA.values())
needs_data = pytest.mark.skipif(not have_data, reason="run src/recon.py first")
load = lambda n: [json.loads(l) for l in open(DATA[n])]


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("text,ok,agent,step", [
    ("Agent Name: Orchestrator\nStep Number: 12\nReason for Mistake: x", True, "Orchestrator", 12),
    ("agent name: A\nSTEP NUMBER: 0", True, "A", 0),
    ("  Agent Name: A b\n  Step Number:  7  ", True, "A b", 7),
    ("Agent Name: A\nStep Number: 3\nAgent Name: B\nStep Number: 9", True, "A", 3),            # first of each
    ("**Agent Name:** A\nStep Number: 3", False, None, None),                                  # no markdown repair
    ("Agent Name: A\nStep Number: 8 (the second message)", False, None, None),
    ("Agent Name: A\nStep Number: -1", False, None, None),
    ("Agent Name: A\nStep Number: 3.0", False, None, None),
    ("Agent Name:   \nStep Number: 3", False, None, None),
    ("Step Number: 3", False, None, None),
    ("Agent Name: A", False, None, None),
    ("", False, None, None),
])
def test_parser(text, ok, agent, step):
    r = parse_response(text, "completed")
    assert r["ok"] == ok and r["agent"] == agent and r["step"] == step


def test_parser_truncated():
    for fr in ("length", "incomplete:max_output_tokens", "max_tokens"):
        r = parse_response("Agent Name: A\nStep Number: 3", fr)
        assert not r["ok"] and r["failure"] == "truncated"
    assert parse_response(None, "completed")["failure"] == "empty_response"


# ---------------------------------------------------------------- scorer
REC = {"true_agents": ["Orchestrator"], "true_step_native": 4, "L": 11}


def test_scorer():
    s = score_run({"ok": True, "agent": "Orchestrator (thought)", "step": 4, "failure": None}, REC)
    assert s["joint_ok"] and s["norm_err"] == 0
    s = score_run({"ok": True, "agent": "WebSurfer", "step": 6, "failure": None}, REC)
    assert not s["agent_ok"] and not s["step_ok"] and s["norm_err"] == pytest.approx(0.2)
    s = score_run({"ok": False, "agent": None, "step": None, "failure": "x"}, REC)
    assert s["parse_fail"] and not s["joint_ok"] and s["norm_err"] is None


def test_scorer_alias_mapping():
    al = {"Agent A": "Orchestrator", "Agent B": "WebSurfer"}
    assert score_run({"ok": True, "agent": "Agent A", "step": 4, "failure": None}, REC, al)["joint_ok"]
    assert not score_run({"ok": True, "agent": "Agent B", "step": 4, "failure": None}, REC, al)["agent_ok"]
    assert not score_run({"ok": True, "agent": "Agent Z", "step": 4, "failure": None}, REC, al)["agent_ok"]   # unknown alias: wrong, not parse failure
    assert map_back("Agent B", al) == "WebSurfer"


def test_multi_agent_label():
    assert match_agent(["A_one", "B_two"], "B_two")


# ---------------------------------------------------------------- rendering
EV = [{"i": i, "agent_raw": a, "role_raw": None, "content": c, "native_fields": {"role": "user"}}
      for i, (a, c) in enumerate([("Alice", "hello Alice and Bob\nsecond line"), ("Bob", "[Step 7] weird content"), ("Alice Cooper", "Alice Cooper says Alice"), ("Bob", "")])]


def test_o1_strip_is_o0():
    for base in (0, 1):
        assert strip_O1(render_O1(EV, base), len(EV), base) == render_O0(EV)


def test_o1_labels_native_base():
    assert render_O1(EV, 1).startswith("[Step 1]\n") and "[Step 4]\n" in render_O1(EV, 1)


def test_o1a_reversible_and_longest_first():
    text, al, n = render_O1A(EV)
    assert al == {"Agent A": "Alice", "Agent B": "Bob", "Agent C": "Alice Cooper"}
    assert "Alice Cooper" not in text and "Bob" not in text and n >= 4
    assert "Agent C says Agent A" in text                           # 'Alice Cooper' wins over 'Alice'; whole words only
    inv = text
    for a, name in al.items():
        inv = inv.replace(a, name)
    assert inv == render_O1(EV)


def test_alias_names():
    n = alias_names(28)
    assert n[0] == "Agent A" and n[25] == "Agent Z" and n[26] == "Agent AA" and len(set(n)) == 28


def test_o2_suffix_only_listed_fields():
    t = render_O2(EV, ["role"])
    assert t.count("(role=user)") == len(EV)
    assert render_O2(EV, []) == render_O1(EV)


def test_fill_plain_replacement_with_braces():
    p = build_prompt({"question": "q {failure log} {x}", "final_answer_or_null": "a", "events": EV, "step_base": 0}, "O0", "with_answer")[0]
    assert "q {failure log} {x}" in p and "{problem}" not in p and "{ground truth}" not in p
    assert "The Answer for the problem is: a." in p


def test_no_answer_variant_drops_only_that_sentence():
    a, b = TEMPLATES["with_answer"], TEMPLATES["no_answer"]
    assert "{ground truth}" not in b and a.replace("The Answer for the problem is: {ground truth}.\n", "") == b


# ---------------------------------------------------------------- leakage (real data)
@needs_data
@pytest.mark.parametrize("name", ["WW_AG", "WW_HC", "WWPRO_TXT"])
def test_no_leakage(name):
    import yaml
    cfg = yaml.safe_load((ld.ROOT / "config/datasets.yaml").read_text())["datasets"][name]
    for r in load(name):
        for c in ("O0", "O1", "O1-A"):
            p, _, _ = build_prompt(r, c, cfg["answer_variant"], cfg["o2_fields"])
            lab = r["labels"]
            for key in ("mistake_reason", "ground_truth_json"):
                v = lab.get(key)
                if v and len(str(v)) >= 20:
                    assert str(v) not in p, (r["traj_id"], key)
            for key in ("mode", "mistake_type", "is_correct", "level"):
                assert f'"{key}"' not in p or key in r["question"]
            assert "mistake_agent" not in p and "mistake_step" not in p
            assert "{problem}" not in p and "{failure log}" not in p
        # the renderer must not even be able to see labels
        src = inspect.getsource(sys.modules["render"])
        assert '["labels"]' not in src and "true_step" not in src and "true_agent" not in src


@needs_data
@pytest.mark.parametrize("name", ["WW_AG", "WW_HC", "WWPRO_TXT"])
def test_o1_byte_identical_real(name):
    for r in load(name):
        assert strip_O1(render_O1(r["events"], r["step_base"]), r["L"], r["step_base"]) == render_O0(r["events"])


@needs_data
def test_normalized_sanity():
    assert len(load("WW_AG")) == 126 and len(load("WW_HC")) == 58
    for n in DATA:
        for r in load(n):
            assert 0 <= r["true_step_native"] < r["L"] and r["step_base"] == 0 and r["true_agents"]


# ---------------------------------------------------------------- experiment 2
def test_features_label_free():
    assert list(inspect.signature(fx.features).parameters) == ["runs", "L"]
    assert list(inspect.signature(fx.candidate).parameters) == ["runs"]


def test_features_values():
    runs = [("A", 3), ("a ", 3), ("B", 3), ("A", 5), None]
    f = fx.features(runs, 11)
    assert f["joint_modal_agreement"] == 2 / 5 and f["agent_agreement"] == 3 / 5 and f["step_agreement"] == 3 / 5
    assert f["unique_pair_count"] == 3 and f["norm_step_spread"] == pytest.approx(2 / 10) and f["parse_failure_count"] == 1 and f["traj_length"] == 11


def test_features_undefined_cases():
    f = fx.features([None] * 5, 4)
    assert f["joint_modal_agreement"] == 0 and f["unique_pair_count"] == 0 and f["norm_step_spread"] == 0 and f["parse_failure_count"] == 5
    assert fx.features([("A", 1), None, None, None, None], 1)["norm_step_spread"] == 0
    assert fx.candidate([None] * 5) == (None, False)


def test_candidate_tie_earliest():
    idx, tie = fx.candidate([("B", 2), ("A", 1), ("A", 1), ("B", 2), None])
    assert tie and idx == 0
    idx, tie = fx.candidate([("A", 1), ("B", 2), ("B", 2), None, None])
    assert not tie and idx == 1


def test_gate_table_target_uses_candidate_run():
    rows = []
    for rep, (ag, stp, ok) in enumerate([("A", 1, False), ("B", 2, True), ("B", 2, True), ("A", 1, False), (None, None, False)], 1):
        rows.append({"dataset": "D", "traj_id": "t", "cluster_id": "c", "replicate": rep, "parse_fail": ag is None, "parsed_agent": ag, "parsed_step": stp, "joint_ok": ok, "L": 5})
    t = fx.build_gate_table(pd.DataFrame(rows))
    assert t.target.iloc[0] == 0 and bool(t.tie_occurred.iloc[0])        # tie A/B -> earliest = A (run 1), which is wrong


def _toy():
    rng = np.random.default_rng(0)
    rows = []
    for d in ("D1", "D2", "D3"):
        for i in range(60):
            m = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
            rows.append({"dataset": d, "traj_id": f"{d}{i}", "cluster_id": f"{d}{i}", "joint_modal_agreement": m, "agent_agreement": m, "step_agreement": m,
                         "unique_pair_count": int(round(6 - 5 * m)), "norm_step_spread": 1 - m, "traj_length": 10, "parse_failure_count": 0,
                         "target": int(rng.random() < m), "tie_occurred": False, "no_candidate": False})
    return pd.DataFrame(rows)


def test_gate_loo_and_metrics():
    tab, notes = gt.loo(_toy())
    assert tab.p_gate.notna().all() and set(notes.values()) == {"ok"}
    df = tab[tab.fold == "D1"].reset_index(drop=True)
    y, p = df.target.values, df.p_gate.values
    assert gt.retained(y, p, 0.25)[1] == 15 and gt.retained(y, p, 1.0)[0] == pytest.approx(y.mean())
    assert gt.retained(y, p, 0.25)[0] >= y.mean()                       # informative feature: top quartile beats accept-all
    assert gt.auroc(y, p) > 0.5 and np.isnan(gt.auroc(np.ones(5), np.arange(5.)))
    ci = gt.bootstrap(df, 0.5, b=200)
    assert ci["brier"][0] <= ci["brier"][1]


def test_gate_uses_only_seven_features():
    assert gt.FEATURES == ["joint_modal_agreement", "agent_agreement", "step_agreement", "unique_pair_count", "norm_step_spread", "traj_length", "parse_failure_count"]


# ---------------------------------------------------------------- experiment 1 stats
def test_wilson_and_holm():
    lo, hi = st.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_effects_pipeline():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        base = rng.random() < 0.3
        for c, ok in (("O0", base), ("O1", base or rng.random() < 0.3), ("O1-A", base)):
            rows.append({"dataset": "D", "judge": "J1", "condition": c, "traj_id": f"t{i}", "cluster_id": f"c{i // 2}", "agent_ok": ok, "step_ok": ok, "joint_ok": ok,
                         "parse_fail": False, "norm_err": 0.0 if ok else 0.3})
    df = pd.DataFrame(rows)
    prim, sec = st.effects(df)
    r = prim[prim.Comparison == "O1 vs O0"].iloc[0]
    assert r["Delta joint (pp)"] >= 0 and r["CI lo (pp)"] <= r["Delta joint (pp)"] <= r["CI hi (pp)"] and r["c"] == 0 and 0 <= r["Holm p"] <= 1
    assert len(st.results_table(df)) == 3
