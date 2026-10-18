"""Prompt rendering. B = plan's A text with `step - N :` headers (labels used ONLY here); C = plan's evidence-first
prompt with the unlabeled log. Plain string replacement (templates contain braces)."""
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_B = (ROOT / "prompts" / "B.txt").read_text().rstrip("\n")
TEMPLATE_C = (ROOT / "prompts" / "C.txt").read_text().rstrip("\n")


def render_entry(e):
    agent, body = e
    return f"{agent}: {body}"


def render_A(entries):  # unlabeled log; used by C (same join as the original all_at_once)
    return "\n".join(render_entry(e) for e in entries)


def render_B(entries):
    return "\n\n".join(f"step - {i} :\n{render_entry(e)}" for i, e in enumerate(entries))


def render_taxonomy(path=ROOT / "prompts" / "taxonomy.yaml"):
    tax = yaml.safe_load(Path(path).read_text())
    collapse = lambda d: re.sub(r"\s+", " ", d).strip()
    return "\n".join(f"{k} - {v['name']}: {collapse(v['description'])}" for k, v in tax.items())


ANSWER_SENTENCE = " The Answer for the problem is: {ground truth}."


def _fill(t, rec, log, taxonomy=None):
    if rec["ground_truth"] in (None, ""):  # livecodebench_pro (macnet, metagpt) has no gold answer: drop the sentence
        assert ANSWER_SENTENCE in t
        t = t.replace(ANSWER_SENTENCE, "")
    t = (t.replace("{problem}", str(rec["problem"]))
          .replace("{ground truth}", str(rec["ground_truth"]))
          .replace("{failure log}", log))
    return t if taxonomy is None else t.replace("{taxonomy}", taxonomy)


def build_prompt_B(rec):
    return _fill(TEMPLATE_B, rec, render_B(rec["entries"]))


def build_prompt_C(rec):
    return _fill(TEMPLATE_C, rec, render_A(rec["entries"]), render_taxonomy())
