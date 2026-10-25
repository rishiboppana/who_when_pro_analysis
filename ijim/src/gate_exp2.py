"""Reliability gate (plan 8.4-8.5): logistic regression on the seven features only, L2 (sklearn default penalty), C=1.0, class_weight=None, standardization fit
on training datasets only, equal total weight per training dataset (weights normalised to mean 1). Leave-one-dataset-out.
Risk-coverage, baselines, Brier, AUROC, cluster bootstrap (B=10,000, seed 20261022, re-ranking inside each resample).
python src/gate_exp2.py"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features_exp2 import FEATURES, build_gate_table
import load_data as ld

SEED, B, COVS = 20261022, 10_000, (1.0, 0.75, 0.5, 0.25)


def fit_predict(train, test):
    """-> predicted probabilities on test. Single-class training set -> constant training rate (logged by caller)."""
    y = train.target.values
    if len(set(y)) < 2:
        return np.full(len(test), float(y.mean())), "single_class_training"
    sc = StandardScaler().fit(train[FEATURES].values)
    w = 1.0 / train.groupby("dataset")["target"].transform("size").values
    w = w * len(train) / w.sum()
    lr = LogisticRegression(C=1.0, class_weight=None, max_iter=1000).fit(sc.transform(train[FEATURES].values), y, sample_weight=w)
    return lr.predict_proba(sc.transform(test[FEATURES].values))[:, 1], "ok"


def loo(tab, groups=None):
    """groups: optional {dataset: fold_name} to merge datasets into one fold (sensitivity). Returns tab with column p_gate and fold."""
    tab = tab.copy()
    tab["fold"] = tab.dataset.map(groups) if groups else tab.dataset
    tab["p_gate"], notes = np.nan, {}
    for f in tab.fold.unique():
        tr, te = tab[tab.fold != f], tab[tab.fold == f]
        p, note = fit_predict(tr, te)
        tab.loc[te.index, "p_gate"] = p
        notes[f] = note
    return tab, notes


def retained(y, p, cov):
    n = math.ceil(cov * len(y))
    order = np.argsort(-p, kind="stable")        # ties broken by original row order, never by labels
    return y[order[:n]].mean(), n


def auroc(y, s):
    pos = y == 1
    if pos.all() or (~pos).all():
        return float("nan")
    r = rankdata(s)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def metrics(y, p, cons):
    out = {f"gate@{int(c*100)}": retained(y, p, c)[0] for c in COVS}
    out["brier"] = float(np.mean((p - y) ** 2))
    out["auroc"] = auroc(y, p)
    out["auroc_consensus"] = auroc(y, cons)
    return out


def bootstrap(df, train_rate, seed=SEED, b=B):
    """Cluster bootstrap inside one held-out dataset. Returns {metric: (lo, hi)}."""
    rng = np.random.default_rng(seed)
    cl = {c: np.flatnonzero(df.cluster_id.values == c) for c in df.cluster_id.unique()}
    keys = list(cl)
    y, p, cons = df.target.values, df.p_gate.values, df.joint_modal_agreement.values
    res = {k: [] for k in [f"gate@{int(c*100)}" for c in COVS] + ["brier", "auroc", "auroc_consensus"]}
    for _ in range(b):
        idx = np.concatenate([cl[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        for k, v in metrics(y[idx], p[idx], cons[idx]).items():
            res[k].append(v)
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) if not np.all(np.isnan(v)) else (float("nan"),) * 2 for k, v in res.items()}


def results(tab, notes, boot=True):
    rows = []
    for f in tab.fold.unique():
        df = tab[tab.fold == f].reset_index(drop=True)
        y, p = df.target.values, df.p_gate.values
        train_rate = tab[tab.fold != f].target.mean()
        m = metrics(y, p, df.joint_modal_agreement.values)
        ci = bootstrap(df, train_rate) if boot else {}
        fmt = lambda k: "" if not ci else f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        for c in COVS:
            k = f"gate@{int(c*100)}"
            rows.append({"Held-out dataset": f, "Method": f"Gate@{int(c*100)}", "Coverage %": 100 * c, "Retained n": math.ceil(c * len(y)), "Joint %": 100 * m[k],
                         "Brier score": m["brier"] if c == 1.0 else "", "AUROC": m["auroc"] if c == 1.0 else "",
                         "95% CI/note": f"joint CI {fmt(k)}" + (f"; Brier CI {fmt('brier')}; AUROC CI {fmt('auroc')}" if c == 1.0 else "") + (f"; {notes[f]}" if c == 1.0 and notes[f] != "ok" else "")
                         + ("; AUROC undefined (single target class)" if c == 1.0 and math.isnan(m["auroc"]) else "")})
        rows.append({"Held-out dataset": f, "Method": "Accept-all", "Coverage %": 100.0, "Retained n": len(y), "Joint %": 100 * y.mean(),
                     "Brier score": float(np.mean((train_rate - y) ** 2)), "AUROC": "n/a", "95% CI/note": f"Brier uses training-datasets target rate {train_rate:.3f}"})
        for name, thr in (("Consensus>=3/5", 0.6), ("Consensus>=4/5", 0.8), ("Consensus=5/5", 1.0)):
            keep = df.joint_modal_agreement.values >= thr - 1e-9
            rows.append({"Held-out dataset": f, "Method": name, "Coverage %": 100 * keep.mean(), "Retained n": int(keep.sum()),
                         "Joint %": 100 * y[keep].mean() if keep.any() else "", "Brier score": "n/a",
                         "AUROC": f"{m['auroc_consensus']:.3f} (continuous joint_modal_agreement)" if name == "Consensus>=3/5" else "",
                         "95% CI/note": "" if keep.any() else "no trajectories retained"})
    return pd.DataFrame(rows)


def main():
    out = ld.ROOT / "outputs" / "exp2"
    runs = pd.read_csv(out / "X2_runs.csv")
    tab = build_gate_table(runs)
    tab.to_csv(out / "X2_gate_table.csv", index=False)
    cnt = []
    for d, g in tab.groupby("dataset"):
        cnt.append({"dataset": d, "n": len(g), "ties": int(g.tie_occurred.sum()), "no_parseable_run": int(g.no_candidate.sum()),
                    **{f"unique_pairs={k}": int((g.unique_pair_count == k).sum()) for k in range(0, 6)}, "target_rate": g.target.mean()})
    pd.DataFrame(cnt).to_csv(out / "X2_counts.csv", index=False)
    t1, n1 = loo(tab)
    t1[["dataset", "traj_id", "target", "p_gate"]].to_csv(out / "X2_gate_predictions.csv", index=False)
    results(t1, n1).to_csv(out / "X2_results.csv", index=False)
    t2, n2 = loo(tab, {"WW_AG": "WW_all", "WW_HC": "WW_all", "WWPRO_TXT": "WWPRO_TXT"})
    results(t2, n2).to_csv(out / "X2_results_sensitivity_WW_merged.csv", index=False)
    plots(t1, out)
    print("done")


def plots(tab, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = list(tab.fold.unique())
    fig, axs = plt.subplots(1, len(folds), figsize=(4.2 * len(folds), 3.6), squeeze=False)
    for ax, f in zip(axs[0], folds):
        df = tab[tab.fold == f].reset_index(drop=True)
        y, p = df.target.values, df.p_gate.values
        cov = np.linspace(0.1, 1, 19)
        ax.plot(cov * 100, [100 * retained(y, p, c)[0] for c in cov], color="#2a78d6", label="gate")
        for thr, mk in ((0.6, "o"), (0.8, "s"), (1.0, "^")):
            keep = df.joint_modal_agreement.values >= thr - 1e-9
            if keep.any():
                ax.scatter(100 * keep.mean(), 100 * y[keep].mean(), marker=mk, color="#eb6834", zorder=3, label=f"consensus >= {int(thr*5)}/5")
        ax.set_title(f"{f} (n={len(y)})"); ax.set_xlabel("coverage %"); ax.set_ylabel("joint accuracy % of retained")
    axs[0][-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "X2_risk_coverage.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, len(folds) + 1, figsize=(3.6 * (len(folds) + 1), 3.6), squeeze=False)
    for ax, (name, df) in zip(axs[0], [("pooled held-out", tab)] + [(f, tab[tab.fold == f]) for f in folds]):
        q = pd.qcut(df.p_gate.rank(method="first"), min(10, len(df)), labels=False)
        g = df.groupby(q).agg(p=("p_gate", "mean"), y=("target", "mean"))
        ax.plot([0, 1], [0, 1], color="grey", lw=1); ax.plot(g.p, g.y, marker="o", color="#2a78d6")
        ax.set_title(name); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed joint accuracy")
    fig.tight_layout(); fig.savefig(out / "X2_calibration.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
