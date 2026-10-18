"""Prompt B rendering: prompt A's text with an explicit `step - N :` header before each log entry.
Separate file so the frozen src/render.py (prompt A) is untouched. Uses the same prompts/A.txt template."""
from render import TEMPLATE


def render_entry(entry):
    # Agent name field: Algorithm-Generated records use "name".
    # Fall back to "role" if a subset has no "name" (Hand-Crafted has only "role"; same as render_A).
    agent = entry.get("name") or entry["role"]
    return f"{agent}: {entry['content']}"   # content untouched, no strip()


def render_B(history):
    parts = []
    for i, entry in enumerate(history):
        # Sanity check against the dataset's own label when it exists.
        if "step" in entry:
            assert entry["step"] == f"step - {i}", (i, entry["step"])
        parts.append(f"step - {i} :\n{render_entry(entry)}")
    return "\n\n".join(parts)


def build_prompt_B(record):
    """Plain string replacement (template contains braces, so no str.format)."""
    return (TEMPLATE
            .replace("{problem}", str(record["problem"]))
            .replace("{ground truth}", str(record["ground_truth"]))
            .replace("{failure log}", render_B(record["history"])))
