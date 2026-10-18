"""Prompt C (evidence-first). Same input as A: the log is render_A(history), unchanged. Only the instruction text
after the log differs (prompts/C.txt). Separate file so the frozen src/render.py (prompt A) is untouched."""
from pathlib import Path
from render import render_A

TEMPLATE_C = (Path(__file__).resolve().parents[1] / "prompts" / "C.txt").read_text()


def build_prompt_C(record):
    """Plain string replacement (template contains braces, so no str.format)."""
    log = render_A(record["history"], record["subset"] == "Hand-Crafted")
    return (TEMPLATE_C
            .replace("{problem}", str(record["problem"]))
            .replace("{ground truth}", str(record["ground_truth"]))
            .replace("{failure log}", log))
