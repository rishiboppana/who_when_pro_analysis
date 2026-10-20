## 9. Smoke test (12 traces x prompts B and C): parseability, truncation, latency only. Correctness is NOT looked at.
Pass rule from the plan: at most 1 parse failure in 12 per prompt (truncation counts as a failure). If it fails, stop and ask; do not edit prompts, parser or settings.
