## 10. Full run (all 2,670 traces, prompts B and C): resumable, appends to disk
One process per prompt, each with half the server's slots, so requests never queue past the client timeout. Stops itself if more than 25% of calls fail after the first 40. Re-running this cell after a disconnect skips finished calls (failed or API-error rows are retried).
