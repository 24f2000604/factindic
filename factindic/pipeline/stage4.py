"""Stage 4: Hybrid Verification Router + single-round candidate reranking (K=5),
capped pruning, drop-only baseline; also logs verdicts against the aligned native source (kept in the logs, not used in any result)."""
from __future__ import annotations

import time

from ..claims import sentence_claims
from ..config import CFG, DIRS
from ..data import items, load_set
from ..io import JsonlAppender, free_gpu, jl_read_glob, jl_write
from ..text import W


def _load_candidates():
    from .stage3 import selected
    sel = selected()
    greedy = {g["item"]: g for g in jl_read_glob(DIRS["outputs"] / f"{sel['system']}.jsonl")}
    acd = {g["item"]: g for g in jl_read_glob(DIRS["outputs"] / "dec_anchor_cad_dola.jsonl")}
    smp = {g["item"]: g for g in jl_read_glob(DIRS["outputs"] / "stage3_samples.jsonl")}
    return sel, greedy, acd, smp


def verify_job(shard, n_shards, **_):
    from ..mt import Translator
    from ..ner import Tagger
    from ..rerank import J_score, drop_only, prune
    from ..verify import MiniCheck, NativeNLI, Router
    sel, greedy, acd, smp = _load_candidates()
    tau, lam = CFG["tau_ver"], CFG["lambda_len"]
    out = JsonlAppender(DIRS["verify"] / "stage4.jsonl", key_fn=lambda r: r["item"], shard=shard)
    its = items(load_set("testbed")) + items(load_set("natural_cm"))
    todo = [x for i, x in enumerate(its) if i % n_shards == shard and x[0] not in out.done and x[0] in acd]
    if not todo:
        return
    router = Router(Translator(), MiniCheck(), NativeNLI())
    ner = Tagger(latin=False)
    for n, (item, rec, v) in enumerate(todo):
        t0 = time.perf_counter()
        cands = [dict(k=0, source="anchor_cad_dola", summary=acd[item]["summary"])]
        cands += [dict(k=i + 1, source="sample", summary=s) for i, s in enumerate(smp.get(item, {}).get("samples", []))]
        seen = {}
        for c in cands:
            cl = sentence_claims(c["summary"])
            key = " ".join(cl)
            if key not in seen:
                seen[key] = router.verify(rec, v, cl) if cl else []
            c["verdicts"] = seen[key]
            c.update(J_score(c["summary"], c["verdicts"], rec, tau, lam, ner))
        best = max(cands, key=lambda c: (c["J"], -c["k"]))
        final, pruned = prune(best["verdicts"], tau) if best["verdicts"] else (best["summary"], False)
        verify_s = time.perf_counter() - t0
        g = greedy.get(item)
        g_cl = sentence_claims(g["summary"]) if g else []
        g_ver = router.verify(rec, v, g_cl) if g_cl else []
        oracle = router.verify(rec, v, sentence_claims(cands[0]["summary"]), mode="oracle") if v in ("roman", "script_mixed") else None
        out.add(dict(item=item, uid=rec["uid"], lang=rec["lang"], variant=v,
                     candidates=[{k: c[k] for k in ("k", "source", "summary", "J", "frac_supported", "R_anchor", "words", "verdicts")} for c in cands],
                     selected_k=best["k"], final_summary=final, pruned=pruned, verify_seconds=verify_s,
                     greedy_summary=g["summary"] if g else "", greedy_verdicts=g_ver,
                     drop_only_summary=drop_only(g_ver, tau) if g else "",
                     oracle_verdicts_c0=oracle, mt=router.mt.backend))
        if n % 50 == 0:
            print(f"[stage4 shard {shard}] {n}/{len(todo)} J*={best['J']:.2f} k*={best['k']} pruned={pruned}", flush=True)
    out.close()
    router.mt.close()
    del router, ner
    free_gpu()


def materialize():
    """Write outputs/factindic.jsonl and outputs/drop_only.jsonl as ordinary system outputs."""
    sel, greedy, acd, smp = _load_candidates()
    rows = jl_read_glob(DIRS["verify"] / "stage4.jsonl")
    fi, do = [], []
    for r in rows:
        gen_lat = acd.get(r["item"], {}).get("latency_s", 0) + smp.get(r["item"], {}).get("latency_s", 0)
        base = dict(item=r["item"], uid=r["uid"], lang=r["lang"], variant=r["variant"])
        fi.append(dict(base, system="factindic", summary=r["final_summary"], sum_words=W(r["final_summary"]),
                       latency_s=gen_lat + r["verify_seconds"], selected_k=r["selected_k"], pruned=r["pruned"],
                       input_fertility=acd.get(r["item"], {}).get("input_fertility")))
        g = greedy.get(r["item"], {})
        do.append(dict(base, system="drop_only", summary=r["drop_only_summary"], sum_words=W(r["drop_only_summary"]),
                       latency_s=g.get("latency_s", 0) + r["verify_seconds"] / max(1, len(r["candidates"])),
                       input_fertility=g.get("input_fertility")))
    jl_write(DIRS["outputs"] / "factindic.jsonl", fi)
    jl_write(DIRS["outputs"] / "drop_only.jsonl", do)
    return len(fi)
