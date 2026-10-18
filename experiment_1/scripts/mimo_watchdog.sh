#!/bin/bash
# Keeps mimo B and C running until all 2,670 are done. Restarts only after 6 concurrent test calls all succeed.
cd "$(dirname "$0")/.."
done_n() { grep -c '"error": null' outputs/raw/mimo/$1.jsonl; }
healthy() { ../.venv/bin/python - <<'P' >/dev/null 2>&1
import sys; sys.path.insert(0,'src')
import run, yaml
from concurrent.futures import ThreadPoolExecutor
run.load_env(); m=yaml.safe_load(open('config/models.yaml'))['models']['mimo']; c=run.make_client(m)
def f(_): c.chat.completions.create(model=m['model_id'],messages=[{"role":"user","content":"Say OK."}],max_tokens=200)
with ThreadPoolExecutor(6) as ex: list(ex.map(f, range(6)))
P
}
while [ "$(done_n B)" -lt 2670 ] || [ "$(done_n C)" -lt 2670 ]; do
  if ! pgrep -f "models mimo" >/dev/null; then
    if healthy && healthy; then
      for p in B C; do [ "$(done_n $p)" -lt 2670 ] && nohup ../.venv/bin/python src/run.py --stage full --models mimo --prompts $p --workers 24 --budget-usd 60 >> outputs/full_mimo_${p}.log 2>&1 & done
      echo "$(date -u +%H:%M:%S) restarted mimo (B=$(done_n B) C=$(done_n C))" >> outputs/resume_mimo.log
      sleep 120
    else
      echo "$(date -u +%H:%M:%S) provider unhealthy, waiting" >> outputs/resume_mimo.log; sleep 120
    fi
  else sleep 60; fi
done
echo "$(date -u +%H:%M:%S) mimo complete" >> outputs/resume_mimo.log
