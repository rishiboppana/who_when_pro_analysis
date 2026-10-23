"""Seven label-free features from 5 repeated diagnoses (plan 8.2) and the candidate (8.3).
`runs` = list of 5 items, each (agent, step) for a parseable run or None for a parse failure, in replicate order 1..5.
Pairs are compared as (agent.strip().casefold(), step). Features take only runs and L, never labels."""
from collections import Counter

FEATURES = ["joint_modal_agreement", "agent_agreement", "step_agreement", "unique_pair_count", "norm_step_spread", "traj_length", "parse_failure_count"]


def _norm(run):
    return (run[0].strip().casefold(), int(run[1]))


def features(runs, L):
    P = [_norm(r) for r in runs if r is not None]
    n_fail = sum(r is None for r in runs)
    f = {"traj_length": L, "parse_failure_count": n_fail}
    if not P:
        return f | {"joint_modal_agreement": 0.0, "agent_agreement": 0.0, "step_agreement": 0.0, "unique_pair_count": 0, "norm_step_spread": 0.0}
    steps = [p[1] for p in P]
    return f | {"joint_modal_agreement": Counter(P).most_common(1)[0][1] / 5,
                "agent_agreement": Counter(p[0] for p in P).most_common(1)[0][1] / 5,
                "step_agreement": Counter(steps).most_common(1)[0][1] / 5,
                "unique_pair_count": len(set(P)),
                "norm_step_spread": (max(steps) - min(steps)) / max(L - 1, 1) if len(P) > 1 else 0.0}


def candidate(runs):
    """-> (index of the run holding the candidate pair, tie_occurred) or (None, False). Most frequent complete pair; on a tie
    the tied pair that appeared in the earliest run."""
    first, cnt = {}, Counter()
    for i, r in enumerate(runs):
        if r is not None:
            p = _norm(r)
            cnt[p] += 1
            first.setdefault(p, i)
    if not cnt:
        return None, False
    top = max(cnt.values())
    tied = [p for p, c in cnt.items() if c == top]
    return min(first[p] for p in tied), len(tied) > 1


def build_gate_table(runs_df, correct_col="joint_ok"):
    """runs_df: X2_runs.csv-like. One row per (dataset, traj_id): features, target (candidate exactly correct on WHO and WHEN),
    tie_occurred, no_candidate, cluster_id, row order preserved by first appearance."""
    import pandas as pd
    rows = []
    for (d, t), g in runs_df.groupby(["dataset", "traj_id"], sort=False):
        g = g.sort_values("replicate")
        by_rep = {int(r.replicate): r for r in g.itertuples()}
        runs = []
        for k in range(1, 6):
            r = by_rep.get(k)
            runs.append(None if r is None or r.parse_fail else (r.parsed_agent, r.parsed_step))
        L = int(g.L.iloc[0])
        idx, tie = candidate(runs)
        target = 0 if idx is None else int(bool(by_rep[idx + 1].joint_ok))   # correctness enters ONLY here, never in features()
        rows.append({"dataset": d, "traj_id": t, "cluster_id": g.cluster_id.iloc[0], **features(runs, L), "target": target,
                     "tie_occurred": tie, "no_candidate": idx is None})
    return pd.DataFrame(rows)
