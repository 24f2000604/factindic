"""H3 (verifier agreement by modality), inter-annotator κ, FactCG-vs-human validation, τ tuning,
and the Lookback-Lens hallucination detector. Writes eval/human_results.json."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..claims import sentence_claims
from ..config import CFG, DIRS
from ..data import load_set
from ..io import free_gpu, jl_read_glob, json_dump
from ..stats import cohen_kappa, macro_f1
from ..text import W, split_sentences

VERIFIERS = {"tv_minicheck": "Translate→MiniCheck", "tv_factcg": "Translate→FactCG",
             "native_nli": "Native mDeBERTa-XNLI", "router": "Round-trip router"}


def export_sheets():
    from ..annotate import export_claim_sheets, export_naturalness
    recs = {r["uid"]: r for s in ("testbed", "natural_cm") for r in load_set(s)}
    pool = []
    for s in ("gemma_A", "gemma_B1", "gemma_B2", "gemma_B3", "gemma_sft", "factindic", "indicbartss", "llama_A"):
        for g in jl_read_glob(DIRS["outputs"] / f"{s}.jsonl"):
            if g["uid"] in recs:
                pool.append(dict(system=s, item=g["item"], uid=g["uid"], lang=g["lang"], variant=g["variant"],
                                 summary=g["summary"]))
    n_s, n_c = export_claim_sheets(pool, recs)
    n_nat = export_naturalness(load_set("testbed"))
    return dict(summaries=n_s, claims=n_c, naturalness_items=n_nat)


def score_annotated(lab):
    """Deployment-mode scores from every verifier for each human-labelled claim (GPU)."""
    from ..mt import Translator
    from ..verify import FactCG, MiniCheck, NativeNLI, Router, max_over_chunks
    recs = {r["uid"]: r for s in ("testbed", "natural_cm") for r in load_set(s)}
    router = Router(Translator(), MiniCheck(), NativeNLI())
    fc = FactCG()
    rows = []
    for (uid, variant), d in lab.groupby(["uid", "variant"]):
        rec = recs[uid]
        claims = list(d.claim)
        v = router.verify(rec, variant, claims)
        src = router.prepare_source(rec, variant)
        p_fc = max_over_chunks(fc, [c for c in src["en_chunks"]], [x["claim_en"] for x in v])
        for (_, r), x, pf in zip(d.iterrows(), v, p_fc):
            rows.append(dict(sid=r.sid, claim_idx=r.claim_idx, split=r.split, lang=r.lang, variant=variant,
                             system=r.system, gold=int(r.gold), error_type=r.error_type,
                             tv_minicheck=x["p_minicheck"], native_nli=x["p_nli"], tv_factcg=pf,
                             router=x["p"], route=x["route"]))
    router.mt.close()
    del router, fc
    free_gpu()
    return pd.DataFrame(rows)


def tune_tau(df, col):
    dev = df[df.split == "dev"]
    if dev.empty or dev.gold.nunique() < 2:
        return CFG["tau_ver"]
    grid = np.linspace(0.1, 0.9, 17)
    return float(max(grid, key=lambda t: macro_f1(dev.gold, (dev[col] >= t).astype(int))))


def h3_with_labels(scored):
    taus = {k: tune_tau(scored, k) for k in VERIFIERS}
    test = scored[scored.split == "test"]
    rows = []
    rng = np.random.default_rng(CFG["seed"])
    for v in sorted(test.variant.unique()):
        d = test[test.variant == v]
        for k, lab in VERIFIERS.items():
            pred = (d[k] >= taus[k]).astype(int)
            kap = cohen_kappa(d.gold, pred)
            bk = []
            for _ in range(CFG["bootstrap"]):
                idx = rng.integers(0, len(d), len(d))
                bk.append(cohen_kappa(d.gold.values[idx], pred.values[idx]))
            bk = np.array([x for x in bk if x == x])
            rows.append(dict(variant=v, verifier=lab, kappa=kap, lo=np.percentile(bk, 2.5) if len(bk) else np.nan,
                             hi=np.percentile(bk, 97.5) if len(bk) else np.nan, macro_f1=macro_f1(d.gold, pred), n=len(d)))
    res = pd.DataFrame(rows)
    # H3 test: on code-mixed variants, does the router close the translate→MiniCheck gap? (paired correctness)
    tests = {}
    for v in ("script_mixed", "roman", "natural_cm"):
        d = test[test.variant == v]
        if len(d) < 10:
            continue
        a = ((d.router >= taus["router"]).astype(int) == d.gold).astype(float).values
        b = ((d.tv_minicheck >= taus["tv_minicheck"]).astype(int) == d.gold).astype(float).values
        from ..stats import paired_bootstrap
        tests[f"router−translate_minicheck accuracy ({v})"] = paired_bootstrap(a, b, CFG["bootstrap"])
    fig = viz.grouped_bars(res, "h3_agreement", "H3: verifier agreement with human claim labels (Cohen's κ)",
                           "Cohen's κ vs. human gold", "variant", "verifier", "kappa", err=("lo", "hi"),
                           order_x=[v for v in ["native", "script_mixed", "roman", "natural_cm"] if v in set(res.variant)],
                           order_g=list(VERIFIERS.values()),
                           caption="Thresholds tuned on the 100-summary dev split; κ on the 200-summary test split.")
    return dict(taus=taus, table=res.to_dict("records"), tests=tests, fig=str(fig))


def h3_proxy():
    """Label-free proxy: does verification on the (back-transliterated) code-mixed source agree with
    verification on the aligned native source for the same claims?"""
    rows = []
    for r in jl_read_glob(DIRS["verify"] / "stage4.jsonl"):
        if not r.get("oracle_verdicts_c0"):
            continue
        dep = r["candidates"][0]["verdicts"]
        for a, b in zip(dep, r["oracle_verdicts_c0"]):
            for k, lab in (("p_minicheck", "Translate→MiniCheck"), ("p_nli", "Native mDeBERTa-XNLI"), ("p", "Round-trip router")):
                rows.append(dict(variant=r["variant"], lang=r["lang"], verifier=lab,
                                 agree=int((a[k] >= CFG["tau_ver"]) == (b["p"] >= CFG["tau_ver"]))))
    if not rows:
        return None
    df = pd.DataFrame(rows).groupby(["variant", "verifier"]).agree.mean().reset_index()
    fig = viz.grouped_bars(df, "h3_proxy_oracle_agreement",
                           "H3 proxy: agreement with verification on the aligned native source",
                           "agreement rate", "variant", "verifier", "agree", pct=True,
                           caption="No human labels needed; the native-source router decision is the reference.")
    return dict(table=df.to_dict("records"), fig=str(fig))


def lookback_detector(labels_from="factcg", scored=None):
    """Train on half of the ablation articles, test on the other half (split by article)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    from ..lookback import train_detector
    feats = jl_read_glob(DIRS["verify"] / "lookback_feats.jsonl")
    if not feats:
        return None
    ev = {}
    for f in DIRS["eval"].glob("eval_*.jsonl"):
        if ".shard" in f.name:
            continue
        for r in jl_read_glob(f):
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
                fig=str(fig), labels=labels_from)


def run(gpu=True):
    from ..annotate import import_claim_labels
    res = {}
    try:
        lab, kappa = import_claim_labels()
    except FileNotFoundError:
        lab, kappa = None, {}
    res["inter_annotator"] = kappa
    if lab is not None and gpu:
        scored = score_annotated(lab)
        scored.to_csv(DIRS["eval"] / "h3_scored_claims.csv", index=False)
        res["h3"] = h3_with_labels(scored)
        test = scored[scored.split == "test"]
        res["factcg_vs_human"] = dict(kappa=cohen_kappa(test.gold, (test.tv_factcg >= 0.5).astype(int)),
                                      n=len(test))
        res["error_types"] = test[test.gold == 0].error_type.value_counts().to_dict()
    res["h3_proxy"] = h3_proxy()
    res["lookback"] = lookback_detector()
    json_dump(res, DIRS["eval"] / "human_results.json")
    return res
