"""Stage 0/1 data preparation: sanitation, testbed/dev/natural-CM sets with normalized anchors,
mFACT-style faithfulness weights, and the data figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, DIRS
from ..io import JsonlAppender, free_gpu, jl_read, jl_read_glob, jl_write, json_dump, json_load
from ..text import chunk_sentences, split_sentences


def build_all():
    from ..data import build_eval_sets, build_natural_cm, build_roman_vocab, build_train_pool, stamp, upgrade
    from ..ner import Tagger
    upgrade()
    build_train_pool()
    build_roman_vocab()
    tagger = Tagger()
    build_eval_sets(tagger)
    build_natural_cm(tagger)
    stamp()
    del tagger
    free_gpu()


def _key(r):
    return f"{r['lang']}:{r['id']}"


def prepare_weights():
    """Fix which pool pairs get faithfulness scores: pairs scored earlier first (scores of changed pairs
    were dropped with the data upgrade), then random. Run once before weights_job."""
    import random
    path = DIRS["data"] / "score_subset.json"
    if path.exists():
        return
    pool = jl_read(DIRS["data"] / "train_pool.jsonl")
    base = DIRS["data"] / "weights.jsonl"
    scored = {w["key"]: w for w in jl_read_glob(base)}
    rng, subset = random.Random(CFG["seed"]), []
    for lang in ("hi", "gu"):
        rows = [r for r in pool if r["lang"] == lang]
        first = [r for r in rows if _key(r) in scored]
        rest = [r for r in rows if _key(r) not in scored]
        n = min(CFG["n_score_per_lang"], len(rows))
        subset += (first + rng.sample(rest, min(len(rest), max(0, n - len(first)))))[:n]
    keep = [scored[_key(r)] for r in subset if _key(r) in scored]
    for sh in base.parent.glob("weights.shard*.jsonl"):
        sh.unlink()
    jl_write(base, keep)
    json_dump([_key(r) for r in subset], path)
    print(f"{len(subset)} pairs to score, {len(keep)} earlier scores still valid")


def weights_job(shard, n_shards, **_):
    """s_MC(D_i, R_i) = mean over reference sentences of max_j MiniCheck(D̃_j, r̃). Discard < 0.30,
    weight w_i = clip(s, 0.3, 1.0)."""
    from ..mt import Translator
    from ..verify import MiniCheck, max_over_chunks
    by_key = {_key(r): r for r in jl_read(DIRS["data"] / "train_pool.jsonl")}
    pool = [by_key[k] for k in json_load(DIRS["data"] / "score_subset.json")]
    out = JsonlAppender(DIRS["data"] / "weights.jsonl", key_fn=lambda r: r["key"], shard=shard)
    todo = [r for i, r in enumerate(pool) if i % n_shards == shard and _key(r) not in out.done]
    if not todo:
        return
    mt = Translator()
    mc = MiniCheck()
    for k, r in enumerate(todo):
        src_en = mt.translate(split_sentences(r["text"]), r["lang"])
        ref_en = [x for x in mt.translate(split_sentences(r["summary"]), r["lang"]) if x]
        chunks = chunk_sentences(src_en, 250)
        ps = max_over_chunks(mc, chunks, ref_en) if ref_en else [0.0]
        s = float(np.mean(ps))
        out.add(dict(key=_key(r), lang=r["lang"], source=r["source"], s_mc=s,
                     w=float(np.clip(s, CFG["w_min"], 1.0)), keep=s >= CFG["w_min"], mt=mt.backend))
        if k % 200 == 0:
            print(f"[shard {shard}] {k}/{len(todo)} s_mc={s:.2f}", flush=True)
    out.close(); mt.close()


def weighted_pool():
    pool = {_key(r): r for r in jl_read(DIRS["data"] / "train_pool.jsonl")}
    rows = []
    for w in jl_read_glob(DIRS["data"] / "weights.jsonl"):
        if w["keep"] and w["key"] in pool:
            rows.append(dict(pool[w["key"]], w=w["w"], s_mc=w["s_mc"]))
    return rows


def figures():
    from .. import viz
    rep = json_load(DIRS["data"] / "sanitation_report.json", [])
    published = {"hi": dict(raw=7957, duplicate_pair=23, duplicate_summary=15, article_prefix_copy=2518,
                            summary_gt_50pct=11),
                 "gu": dict(raw=8457, duplicate_pair=0, duplicate_summary=113, article_prefix_copy=135,
                            summary_gt_50pct=37)}
    rows = []
    for r in rep:
        for rule in ["empty_or_short", "duplicate_pair", "duplicate_summary", "article_prefix_copy", "summary_gt_50pct"]:
            rows.append(dict(lang=r["lang"], rule=rule, source="this run", pct=100 * r["removed"].get(rule, 0) / max(1, r["raw"])))
            if rule in published[r["lang"]]:
                rows.append(dict(lang=r["lang"], rule=rule, source="CEUR Vol-3395 audit",
                                 pct=100 * published[r["lang"]][rule] / published[r["lang"]]["raw"]))
    if rows:
        df = pd.DataFrame(rows)
        for lang in ("hi", "gu"):
            viz.grouped_bars(df[df.lang == lang], f"data_sanitation_{lang}",
                             f"ILSUM {viz.LANG_LABEL[lang]} train: pairs removed by each sanitation rule",
                             "% of raw training pairs", "rule", "source", "pct",
                             caption="Prefix copies are counted even after a dateline. The published audit may define some rules differently.")
    w = pd.DataFrame(jl_read_glob(DIRS["data"] / "weights.jsonl"))
    if not w.empty:
        viz.histogram({f"{viz.LANG_LABEL[l]}": w[w.lang == l].s_mc.values for l in ("hi", "gu")},
                      "data_faithfulness_scores", "Reference faithfulness s_MC(D, R) in the training pool",
                      "MiniCheck support of reference sentences", vline=CFG["w_min"], vlabel="discard < 0.30",
                      caption="Soft weighting keeps abstractive pairs (0.3–0.75) instead of hard NLI filtering.")
    tb = jl_read(DIRS["data"] / "testbed.jsonl")
    if tb:
        rows = [dict(lang=r["lang"], variant=v, cmi=c) for r in tb for v, c in r["cmi"].items()]
        nc = jl_read(DIRS["data"] / "natural_cm.jsonl")
        rows += [dict(lang=r["lang"], variant="natural_cm", cmi=c) for r in nc for c in r["cmi"].values()]
        df = pd.DataFrame(rows).groupby(["lang", "variant"]).cmi.mean().reset_index()
        viz.grouped_bars(df, "data_cmi", "Code-mixing index of each input modality", "mean CMI",
                         "variant", "lang", "cmi", order_x=[v for v in ["native", "script_mixed", "roman", "natural_cm"] if v in set(df.variant)],
                         order_g=["hi", "gu"], color_fn=viz.color_of,
                         caption="Word-level language ID. Romanizing changes the script, not the language mix, so its CMI should stay near 0.")
