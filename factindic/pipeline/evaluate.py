"""Non-circular evaluation. FactCG-DeBERTa-L is held out: it never weights training
data and never reranks candidates. Ground truth source = the aligned native article D_mono (facts are
invariant across variants by construction), so every system and modality is graded against the
same facts.

  P_fact = mean_i I[FactCG(D̃, c̃_i) ≥ τ]
  R_fact = mean_j I[FactCG(S̃, r̃_j) ≥ τ] over C_grounded(R) = {r_j : FactCG(D̃, r̃_j) ≥ τ}
"""
from __future__ import annotations

from ..claims import sentence_claims
from ..config import CFG, DIRS
from ..data import load_set
from ..io import JsonlAppender, free_gpu, jl_read_glob
from ..metrics import compression_ratio, fact_prf, rouge
from ..normalizer import anchor_prf, anchors
from ..text import W, chunk_sentences, split_sentences

TAU_EVAL = 0.5


def _records():
    return {r["uid"]: r for s in ("testbed", "natural_cm") for r in load_set(s)}


def _sub(a, kinds):
    return dict(numbers=a.get("numbers", []) if "num" in kinds else [],
                dates=a.get("dates", []) if "date" in kinds else [],
                entities=a.get("entities", []) if "ent" in kinds else [])


def ref_job(shard, n_shards, **_):
    """Translate sources + references once; mark source-grounded reference claims."""
    from ..mt import Translator
    from ..verify import FactCG, max_over_chunks
    recs = list(_records().values())
    out = JsonlAppender(DIRS["eval"] / "refs.jsonl", key_fn=lambda r: r["uid"], shard=shard)
    todo = [r for i, r in enumerate(recs) if i % n_shards == shard and r["uid"] not in out.done]
    if not todo:
        return
    mt, fc = Translator(), FactCG()
    for r in todo:
        d_en = mt.translate(split_sentences(r["text"]), r["lang"])
        rc = sentence_claims(r["reference"])
        rc_en = mt.translate(rc, r["lang"]) if rc else []
        p = max_over_chunks(fc, chunk_sentences(d_en, 400), rc_en) if rc_en else []
        out.add(dict(uid=r["uid"], d_en=d_en, ref_claims=rc, ref_en=rc_en, p_grounded=p,
                     grounded=[x >= TAU_EVAL for x in p], mt=mt.backend))
    out.close(); mt.close()
    del mt, fc
    free_gpu()


def eval_job(shard, n_shards, systems, **_):
    from ..mt import Translator
    from ..ner import Tagger
    from ..verify import FactCG, max_over_chunks
    recs = _records()
    refs = {r["uid"]: r for r in jl_read_glob(DIRS["eval"] / "refs.jsonl")}
    mt, fc, ner = Translator(), FactCG(), Tagger()
    cache = {}
    for system in systems:
        out = JsonlAppender(DIRS["eval"] / f"eval_{system}.jsonl", key_fn=lambda r: r["item"], shard=shard)
        gens = [g for g in jl_read_glob(DIRS["outputs"] / f"{system}.jsonl") if g["uid"] in recs and g["uid"] in refs]
        todo = [g for i, g in enumerate(gens) if i % n_shards == shard and g["item"] not in out.done]
        for n, g in enumerate(todo):
            rec, ref = recs[g["uid"]], refs[g["uid"]]
            S = g.get("summary") or ""
            cl = sentence_claims(S)
            cl_en = mt.translate(cl, rec["lang"]) if cl else []
            p_claims = max_over_chunks(fc, chunk_sentences(ref["d_en"], 400), cl_en, cache) if cl_en else []
            g_ref = [e for e, ok in zip(ref["ref_en"], ref["grounded"]) if ok]
            s_en = " ".join(cl_en)
            p_rec = max_over_chunks(fc, [s_en], g_ref, cache) if (g_ref and s_en) else ([0.0] * len(g_ref))
            P, R, F = fact_prf([p >= TAU_EVAL for p in p_claims], [p >= TAU_EVAL for p in p_rec])
            if not cl:
                P = float("nan")
            a_s = anchors(S, rec["lang"], ner.entities(S) if S else [])
            row = dict(item=g["item"], uid=g["uid"], lang=rec["lang"], variant=g["variant"], system=system,
                       n_claims=len(cl), p_claims=p_claims, P_fact=P, R_fact=R, F1_fact=F,
                       n_ref_grounded=len(g_ref), words=W(S), CR=compression_ratio(S, rec["text"]),
                       empty=not S.strip(), **rouge(rec["reference"], S))
            row.update(anchor_prf(a_s, rec["anchors"], rec["ref_anchors"], rec["text"], rec["lang"]))
            for kind, ks in (("num", ("num", "date")), ("ent", ("ent",))):
                sub = anchor_prf(_sub(a_s, ks), _sub(rec["anchors"], ks), _sub(rec["ref_anchors"], ks), rec["text"], rec["lang"])
                row.update({f"{k}_{kind}": v for k, v in sub.items() if k in ("P_anchor", "R_anchor", "F1_anchor")})
            out.add(row)
            if n % 200 == 0:
                print(f"[eval {system} shard {shard}] {n}/{len(todo)} P={P:.2f} R={R:.2f}", flush=True)
        out.close()
    mt.close()
    del mt, fc, ner
    free_gpu()


def bertscore_job(shard, n_shards, systems, **_):
    import random
    from ..metrics import BertScorer
    recs = _records()
    out = JsonlAppender(DIRS["eval"] / "bertscore.jsonl", key_fn=lambda r: (r["system"], r["item"]), shard=shard)
    bs = BertScorer()
    for lang in ("hi", "gu"):
        bs.fit_baseline(lang, [r["reference"] for r in recs.values() if r["lang"] == lang])
    for system in systems:
        gens = [g for g in jl_read_glob(DIRS["outputs"] / f"{system}.jsonl")
                if g["uid"] in recs and (system, g["item"]) not in out.done]
        gens = [g for i, g in enumerate(gens) if i % n_shards == shard]
        for lang in ("hi", "gu"):
            gl = [g for g in gens if g["lang"] == lang]
            for i in range(0, len(gl), 64):
                b = gl[i:i + 64]
                sc = bs.score([g["summary"] for g in b], [recs[g["uid"]]["reference"] for g in b], lang)
                for g, s in zip(b, sc):
                    out.add(dict(system=system, item=g["item"], bertscore=s, baseline=bs.baseline[lang]))
    out.close()
    del bs
    free_gpu()
