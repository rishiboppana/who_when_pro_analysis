"""Builds whowhen_qwen35_colab.ipynb from colab/cells/*. Run from the repo root:  .venv/bin/python colab/make_notebook.py
Each cell is a plain file in colab/cells (so it can be tested); code and the luna/MiMo logs are embedded (xz + base64) so the notebook is self-contained."""
import nbformat as nbf, tarfile, io, lzma, base64, hashlib, textwrap
from pathlib import Path
R = Path(__file__).resolve().parents[1]; CELLS = Path(__file__).resolve().parent / "cells"
CODE_FILES = ["experiment_1/src/analyze.py", "experiment_1/src/data.py", "experiment_1/src/parser.py", "experiment_1/src/render.py", "experiment_1/src/run.py",
              "experiment_1/src/run_local.py", "experiment_1/src/score.py", "experiment_1/src/tokens_latency.py", "experiment_1/tests/test_all.py",
              "experiment_1/config/models.yaml", "experiment_1/config/pricing.yaml", "experiment_1/config/smoke_ids.txt", "experiment_1/prompts/B.txt",
              "experiment_1/prompts/C.txt", "experiment_2/src/build.py", "experiment_2/src/gee_lib.py", "experiment_2/src/primary.py", "experiment_2/src/run_all.py",
              "experiment_2/src/breakdown.py", "experiment_2/src/figures.py", "colab/requirements-colab.txt"]
LOG_FILES = [f"experiment_1/outputs/raw/{m}/{p}.jsonl" for m in ("luna", "mimo") for p in ("B", "C")]
def blob(files):
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as t:
        for f in files: t.add(R / f, arcname=f)
    return textwrap.fill(base64.b64encode(lzma.compress(b.getvalue(), preset=9 | lzma.PRESET_EXTREME)).decode(), 100)
sha = lambda f: hashlib.sha256((R / f).read_bytes()).hexdigest()
SUBS = {"@@TEXT_SHA@@": sha("data/text.jsonl"), "@@TAX_SHA@@": sha("data/taxonomy.yaml"), "@@TEXT_LINES@@": str(sum(1 for _ in open(R / "data/text.jsonl")))}
def read(name):
    s = (CELLS / name).read_text()
    for k, v in SUBS.items(): s = s.replace(k, v)
    return s.rstrip("\n")
UNPACK = '''
def unpack(b64, dest):
    data = lzma.decompress(base64.b64decode("".join(b64.split())))
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(dest)
        return t.getnames()
'''
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md(read("00_intro.md")); md("## 1. GPU check and settings"); code(read("01_gpu.py"))
md(read("02_storage.md")); code(read("02_storage.py"))
md("## 3. Embedded project files (code, prompts, configs: 21 files, xz + base64). Do not edit; run to write them to disk.")
code('CODE_B64 = """\n' + blob(CODE_FILES) + '\n"""\n' + UNPACK + 'names = unpack(CODE_B64, PROJECT)\nprint(len(names), "files written under", PROJECT); print(*names, sep="\\n")')
md("## 4. Embedded E1 results for luna and MiMo (needed only for scoring and E2; MiMo is **partial**: B 1,894 and C 1,196 of 2,670 traces)")
code('LOGS_B64 = """\n' + blob(LOG_FILES) + '\n"""\n' + 'names = unpack(LOGS_B64, PROJECT)\nfor f in names:\n    print(f, sum(1 for _ in open(f"{PROJECT}/{f}")), "rows")')
md("## 5. Dataset from Hugging Face (public), with integrity checks"); code(read("05_data.py"))
md(read("06_install.md")); code(read("06_install.py"))
md(read("07_renderer.md")); code(read("07_renderer.py"))
md(read("08_server.md")); code(read("08_server.py"))
md(read("09_sanity.md")); code(read("09_sanity.py"))
md(read("10_manifest.md")); code(read("10_manifest.py"))
md(read("11_smoke.md")); code(read("11_smoke_run.py")); code(read("11_smoke_summary.py"))
md(read("12_full.md")); code(read("12_full_launch.py")); code(read("12_full_monitor.py")); code(read("12_full_eta.py"))
md(read("13_e1.md")); code(read("13_e1.py")); code(read("13_e1_show.py"))
md(read("14_e2.md")); code(read("14_e2_a.py")); code(read("14_e2_b.py")); code(read("14_e2_c.py"))
md(read("15_package.md")); code(read("15_package.py"))
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "A100"}
nbf.write(nb, str(Path(__file__).resolve().parent / "whowhen_qwen35_colab.ipynb"))
print("cells:", len(C))
