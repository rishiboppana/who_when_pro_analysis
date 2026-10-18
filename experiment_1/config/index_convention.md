# Step-index convention (re-verified in Phase 0 on the repo copy)
`mistake_step` is 0-based and indexes `history` directly.
- AG: `history[ms]` belongs to `mistake_agent` in 123/126 (vs 22/126 at `ms-1`); `ms == 0` occurs, so 1-based is impossible.
- HC: 56/58 at `history[ms]` (vs 20/58 at `ms-1`); max index < len(history) for all 184.
- Match test = case-insensitive substring of mistake_agent in the speaker field. NOT done: reading 10 entries by hand (the 5 mismatches were not inspected again; the earlier recon labelled 3 AG mismatches as off-by-one annotation noise).
