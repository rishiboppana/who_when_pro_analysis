time.sleep(120); P2 = progress(); dt = time.time() - t1
for p in ["B", "C"]:
    rate = (P2[p]["done"] - P1[p]["done"]) / dt * 3600
    left = 2670 - P2[p]["done"]
    print(f"prompt {p}: {P2[p]['done']}/2670 | {rate:.0f} calls/h | {P2[p]['out_tokens'] - P1[p]['out_tokens']} new output tokens in {dt:.0f}s | ETA {left / rate:.1f} h" if rate > 0 else f"prompt {p}: {P2[p]['done']}/2670 (no progress in {dt:.0f}s)")
