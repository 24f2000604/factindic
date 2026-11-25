"""Numbers added after the review: python paper/extra_numbers.py → paper/extra_numbers.json   (numpy only, CPU)
Article-clustered bootstrap intervals, the Gujarati-vs-Hindi difference of the Romanization loss, agreement of the Stage 4
verifiers with FactCG, and the Kaggle replication. Reads only files that are in the repository."""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
B, TEST = 4000, ("native", "script_mixed", "roman")
ALL = TEST + ("natural_cm",)


def load(folder):
    out = {}
    for f in sorted((REPO / folder).glob("eval_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            x = json.loads(line)
            out.setdefault(x["system"], {})[x["item"]] = x
    return out


RUNS = {"laptop": load("results/final_run/4_evaluation/eval"), "kaggle": load("results/kaggle_run/eval")}


def boot(d, groups, seed=42):
    """Mean of paired differences; resamples articles, because the items of one article are not independent."""
    d, groups = np.asarray(d, float), np.asarray(groups)
    ok = ~np.isnan(d)
    d, g = d[ok], np.unique(groups[ok], return_inverse=True)[1]
    n = g.max() + 1
    s, c = np.bincount(g, d, n), np.bincount(g, minlength=n)
    ix = np.random.default_rng(seed).integers(0, n, (B, n))
    b = s[ix].sum(1) / c[ix].sum(1)
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return dict(diff=float(d.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), p=float(min(1, max(p, 1 / B))),
                n=int(len(d)), articles=int(n))


def pair(run, a, b, metric, variants=ALL):
    X, Y = RUNS[run][a], RUNS[run][b]
    items = sorted(i for i in X if i in Y and X[i]["variant"] in variants)
    return boot([X[i][metric] - Y[i][metric] for i in items], [X[i]["uid"] for i in items])


def roman_loss(run, system):
    by = {}
    for x in RUNS[run][system].values():
        by.setdefault((x["lang"], x["uid"]), {})[x["variant"]] = x["P_fact"]
    d = {l: np.array([v["roman"] - v["native"] for (k, _), v in sorted(by.items()) if k == l and "roman" in v]) for l in ("hi", "gu")}
    d = {l: v[~np.isnan(v)] for l, v in d.items()}
    rng = np.random.default_rng(42)
    b = [d["gu"][rng.integers(0, len(d["gu"]), len(d["gu"]))].mean() - d["hi"][rng.integers(0, len(d["hi"]), len(d["hi"]))].mean() for _ in range(B)]
    mixed = {l: float(np.nanmean([v["script_mixed"] - v["native"] for (k, _), v in by.items() if k == l and "script_mixed" in v])) for l in ("hi", "gu")}
    return dict(hi=boot(d["hi"], np.arange(len(d["hi"]))), gu=boot(d["gu"], np.arange(len(d["gu"]))), mixed=mixed,
                gu_minus_hi=dict(diff=float(d["gu"].mean() - d["hi"].mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        r[s == v] = r[s == v].mean()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po, pe = (a == b).mean(), a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def verifiers():
    """Sentences of candidate 0 (Anchor-CAD + DoLa), the only candidate that FactCG graded."""
    E, rows = RUNS["laptop"]["dec_anchor_cad_dola"], []
    for line in open(REPO / "results/final_run/4_evaluation/verify/stage4.jsonl", encoding="utf-8"):
        r = json.loads(line)
        c, e = r["candidates"][0]["verdicts"], E[r["item"]]
        if len(c) == len(e["p_claims"]):
            rows += [(p, v["p_minicheck"], v["p_nli"], v["p"], v["route"] == "nli") for v, p in zip(c, e["p_claims"])]
    a = np.array(rows, float)
    y = (a[:, 0] >= .5).astype(int)
    out = dict(n=len(a), factcg_supported=float(y.mean()), to_nli=float(a[:, 4].mean()))
    for k, name in ((1, "minicheck"), (2, "nli"), (3, "router")):
        pred = (a[:, k] >= .5).astype(int)
        out[name] = dict(auroc=auroc(y, a[:, k]), kappa=kappa(y, pred), supported=float(pred.mean()))
    return out


def replication(run, base):
    R = RUNS[run]
    P = lambda s: float(np.nanmean([x["P_fact"] for x in R[s].values() if x["variant"] in TEST]))
    return dict(base=base, systems=len([s for s in R if s != "indicbartss"]) + ("indicbartss" in R),
                P={s: P(s) for s in ("gemma_A", "gemma_B1", "gemma_B3", "dec_anchor_cad_dola", "factindic", "drop_only")},
                roman={s: roman_loss(run, s) for s in ("gemma_A", "gemma_B1", "factindic")},
                h4={ref: {m: pair(run, "dec_anchor_cad_dola", ref, m, TEST) for m in ("P_fact", "F1_anchor_ent", "F1_anchor_num")}
                    for ref in ("dec_static_cad", "dec_adacad")},
                pairs={f"{a}-{b}": {m: pair(run, a, b, m) for m in ("P_fact", "F1_anchor")}
                       for a, b in (("factindic", "gemma_A"), ("factindic", base), ("factindic", "dec_anchor_cad_dola"),
                                    ("dec_anchor_cad_dola", base), ("gemma_B1", "gemma_A"), ("factindic", "drop_only"))},
                pairs_testbed={f"{a}-{b}": {m: pair(run, a, b, m, TEST) for m in ("P_fact", "F1_anchor")}
                               for a, b in (("factindic", "gemma_A"), ("factindic", base), ("gemma_B1", "gemma_A"))})


out = dict(verifiers=verifiers(), laptop=replication("laptop", "gemma_B1"), kaggle=replication("kaggle", "gemma_B3"))
(REPO / "paper" / "extra_numbers.json").write_text(json.dumps(out, indent=1))
f = lambda t: f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]" + (f" p={t['p']:.4f} n={t['n']}" if "p" in t else "")
print(out["verifiers"])
for r in ("laptop", "kaggle"):
    x = out[r]
    print(r, {k: round(v, 3) for k, v in x["P"].items()}, x["systems"])
    for s, v in x["roman"].items(): print("  roman", s, "hi", f(v["hi"]), "gu", f(v["gu"]), "gu-hi", f(v["gu_minus_hi"]), v["mixed"])
    for k, v in x["h4"].items(): print("  h4", k, {m: f(t) for m, t in v.items()})
    for k, v in x["pairs"].items(): print("  440", k, {m: f(t) for m, t in v.items()})
    for k, v in x["pairs_testbed"].items(): print("  360", k, {m: f(t) for m, t in v.items()})
