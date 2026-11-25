"""Paired bootstrap CIs, fixed-effects regressions with article-clustered SEs, agreement stats."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(a, b, B=1000, seed=42):
    """Mean of (a − b) over paired units, 95% percentile CI, two-sided bootstrap p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) < 5:
        return dict(n=int(len(d)), diff=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"))
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(n=int(len(d)), diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), p=float(min(1.0, p)))


def bootstrap_mean(x, B=1000, seed=42):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n=int(len(x)))


def fe_ols(df, formula, term, cluster="uid"):
    """OLS with fixed effects in the formula; SEs clustered by article. Returns β, SE, p (1- and 2-sided)."""
    import statsmodels.formula.api as smf
    try:
        cols = [c for c in df.columns if c in formula]
        d = df.dropna(subset=cols)
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
        b, p2 = float(m.params[term]), float(m.pvalues[term])
        return dict(beta=b, se=float(m.bse[term]), p_two=p2, p_one_pos=p2 / 2 if b > 0 else 1 - p2 / 2,
                    p_one_neg=p2 / 2 if b < 0 else 1 - p2 / 2, n=int(m.nobs))
    except Exception as e:
        return dict(error=repr(e)[:200])


def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    a, b = list(a), list(b)
    return float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else float("nan")
