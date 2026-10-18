#!/bin/bash
# wait until the novita provider answers 3 test calls in a row, then resume mimo B and C (24 workers each)
cd "$(dirname "$0")/.."
ok=0
while [ $ok -lt 3 ]; do
  if ../.venv/bin/python - <<'P' >/dev/null 2>&1
import sys; sys.path.insert(0,'src')
import run, yaml
run.load_env(); m=yaml.safe_load(open('config/models.yaml'))['models']['mimo']
run.make_client(m).chat.completions.create(model=m['model_id'],messages=[{"role":"user","content":"Say OK."}],max_tokens=200)
P
  then ok=$((ok+1)); else ok=0; fi
  sleep 20
done
for p in B C; do nohup ../.venv/bin/python src/run.py --stage full --models mimo --prompts $p --workers 24 --budget-usd 60 >> outputs/full_mimo_${p}.log 2>&1 & done
echo "resumed mimo at $(date -u +%H:%M:%S)" >> outputs/resume_mimo.log
