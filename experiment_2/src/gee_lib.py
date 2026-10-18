"""GEE helpers. Primary spec (plan section 4): binomial/logit, exchangeable working correlation, robust covariance,
groups = seed_group (proxy for the shared seed; plan used trajectory_id)."""
import warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

REF = {"subset": "MC", "model": "luna", "prompt": "B"}      # references (plan: AG, luna, A -> here MC, luna, B)


def prep(d, length_col="log2_length"):
    d = d.copy(); d["lc"] = d[length_col] - d[length_col].mean()     # centred at the analysis-sample mean
    d["grp"] = pd.factorize(d.seed_group)[0]
    return d


def fit(formula, d, outcome, cov_type="robust", fam=None):
    """Returns (result or None, warnings list, error text). Never alters the model on failure."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            m = smf.gee(f"{outcome} ~ {formula}", groups="grp", data=d.sort_values("grp"), family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
            res = m.fit(cov_type=cov_type)
        except Exception as e:  # noqa
            return None, [str(x.message) for x in w], f"{type(e).__name__}: {e}"
    msgs = [f"{x.category.__name__}: {x.message}" for x in w]
    return res, msgs, None


def contrast(res, vec_by_name, label):
    """Linear contrast on the robust covariance. vec_by_name: {param name: weight}."""
    v = np.zeros(len(res.params)); idx = list(res.params.index)
    for k, wt in vec_by_name.items(): v[idx.index(k)] = wt
    t = res.t_test(v.reshape(1, -1)); b = float(t.effect[0]); se = float(t.sd[0][0]); p = float(t.pvalue)
    lo, hi = b - 1.959964 * se, b + 1.959964 * se
    return {"Effect": label, "coef": b, "se": se, "OR": np.exp(b), "CI_lo": np.exp(lo), "CI_hi": np.exp(hi), "p_raw": p}


def slopes(res, inter_name):
    """Doubling ORs per subset (reference MC; MA = beta + interaction) and the interaction."""
    return [contrast(res, {"lc": 1}, "doubling of length, MC (reference)"),
            contrast(res, {"lc": 1, inter_name: 1}, "doubling of length, MA"),
            contrast(res, {inter_name: 1}, "interaction MA - MC (difference in slopes)")]


def stratum_ok(d, outcome, min_traces=30, min_events=10):
    ev = int(d[outcome].sum()); ne = int((1 - d[outcome]).sum())
    return d.trajectory_id.nunique() >= min_traces and ev >= min_events and ne >= min_events, ev, ne
