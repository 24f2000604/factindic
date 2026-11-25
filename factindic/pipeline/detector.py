"""Lookback-Lens detector: unsupported sentences from attention alone. The labels are FactCG verdicts (there is
no human annotation), so it learns to predict the held-out grader. Writes eval/lookback.json."""
from __future__ import annotations

import numpy as np

from .. import viz
from ..config import CFG, DIRS
from ..io import jl_read_glob, json_dump
from ..text import W, split_sentences


def lookback_detector():
    """Train on half of the ablation articles, test on the other half (split by article)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    from ..lookback import train_detector
    feats = jl_read_glob(DIRS["verify"] / "lookback_feats.jsonl")
    if not feats:
        return None
    ev = {}
    for name in sorted({f.name.split(".")[0] for f in DIRS["eval"].glob("eval_*.jsonl")}):     # shard files included
        for r in jl_read_glob(DIRS["eval"] / f"{name}.jsonl"):
            ev[(r["system"], r["item"])] = r
    X, y, groups = [], [], []
    gen_cache = {}
    for fr in feats:
        e = ev.get((fr["system"], fr["item"]))
        if not e:
            continue
        if fr["system"] not in gen_cache:
            gen_cache[fr["system"]] = {g["item"]: g for g in jl_read_glob(DIRS["outputs"] / f"{fr['system']}.jsonl")}
        g = gen_cache[fr["system"]].get(fr["item"])
        if not g:
            continue
        sents = split_sentences(g["summary"])
        keep = [i for i, s in enumerate(sents) if W(s) >= 3]
        if len(keep) != len(e["p_claims"]) or len(sents) != len(fr["feats"]):
            continue
        for i, p in zip(keep, e["p_claims"]):
            X.append(fr["feats"][i]); y.append(int(p < 0.5)); groups.append(e["uid"])
    if len(set(y)) < 2 or len(X) < 50:
        return None
    X, y = np.array(X), np.array(y)
    uids = sorted(set(groups))
    rng = np.random.default_rng(CFG["seed"])
    tr_u = set(rng.choice(uids, len(uids) // 2, replace=False))
    tr = np.array([g in tr_u for g in groups])
    clf = train_detector(X[tr], y[tr])
    s = clf.predict_proba(X[~tr])[:, 1]
    auc = roc_auc_score(y[~tr], s)
    fpr, tpr, _ = roc_curve(y[~tr], s)
    base = X[~tr][:, :-1].mean(1)              # mean lookback ratio alone (lower → more hallucination)
    auc_b = roc_auc_score(y[~tr], -base)
    fpr_b, tpr_b, _ = roc_curve(y[~tr], -base)
    fig = viz.roc({"Lookback-Lens LR (per-head)": (fpr, tpr, auc), "Mean lookback ratio": (fpr_b, tpr_b, auc_b)},
                  "lookback_roc", "Detecting unsupported sentences from attention alone",
                  caption=f"Labels: held-out FactCG; train/test split by article; n_test={int((~tr).sum())}.")
    return dict(auroc=float(auc), auroc_mean_ratio=float(auc_b), n_train=int(tr.sum()), n_test=int((~tr).sum()),
                fig=str(fig), labels="factcg")


def run():
    res = lookback_detector()
    json_dump(res, DIRS["eval"] / "lookback.json")
    return res
