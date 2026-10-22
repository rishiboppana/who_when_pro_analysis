## 10. Full run (all 2,670 traces, prompts B and C): resumable, appends to disk
One process per prompt, each with half the server's slots, so requests never queue past the client timeout. Stops itself if more than 25% of calls fail after the first 40. Re-running this cell after a disconnect skips finished calls (failed or API-error rows are retried).

**If the run is slow:** check the vLLM log line `Running: N reqs, Waiting: M reqs, GPU KV cache usage: P%`. Near-100% KV usage with a long waiting queue means the KV cache, not compute, is the limit: use an 80 GB GPU, or on 40 GB set `KV_DTYPE = "fp8"` in the first cell and restart the server (finished calls are kept).
