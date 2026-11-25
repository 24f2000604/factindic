"""Stage 3: anchor-aware context-contrasted decoding, K=5 candidates, CoVe baseline,
perplexity, and Lookback-Lens features."""
from __future__ import annotations

import random

import numpy as np
import torch

from ..config import CFG, DIRS, MODELS
from ..data import items, load_set
from ..io import JsonlAppender, free_gpu, jl_read_glob, json_dump, json_load, peak_vram_gb
from ..text import W
from .stage2 import prompt_for

SELECTED = DIRS["outputs"] / "stage2_selected.json"
TUNED = DIRS["outputs"] / "stage3_params.json"

TUNE_GRID = [  # (alpha_base, alpha_anchor, beta) — small grid inside the spec's ranges
    (0.4, 0.9, 0.2), (0.2, 0.6, 0.2), (0.4, 0.9, 0.0), (0.4, 1.2, 0.4),
][: CFG["n_tune"]]


def selected():
    sel = json_load(SELECTED)
    return sel or dict(system="gemma_B1", model_key="gemma", cond="B1", adapter=None)   # if the dev selection failed


def ablation_uids():
    rng = random.Random(CFG["seed"])
    recs = load_set("testbed")
    out = []
    for lang in ("hi", "gu"):
        lr = sorted(r["uid"] for r in recs if r["lang"] == lang)
        out += rng.sample(lr, min(CFG["n_decoding_ablation_per_lang"], len(lr)))
    return sorted(out)


def _items(sets, variants=None, uids=None):
    its = []
    for s in sets:
        its += items(load_set(s), variants)
    if uids is not None:
        uids = set(uids)
        its = [x for x in its if x[1]["uid"] in uids]
    return its


def decode_job(shard, n_shards, method, system, params=None, sets=("testbed",), variants=None, uids=None,
               batch_size=6, **_):
    from ..decoding import anchor_token_ids, contrastive_generate
    from ..llm import chat_text, load_causal, n_tokens
    from ..prompts import bare_prior_prompt
    sel = selected()
    params = params or json_load(TUNED, {}) or {}
    out = JsonlAppender(DIRS["outputs"] / f"{system}.jsonl", key_fn=lambda r: r["item"], shard=shard)
    todo = [x for i, x in enumerate(_items(sets, variants, uids)) if i % n_shards == shard and x[0] not in out.done]
    if not todo:
        return
    model, tok, desc = load_causal(MODELS[sel["model_key"]], adapter=sel.get("adapter"))
    torch.cuda.reset_peak_memory_stats()
    kw = dict(alpha_base=params.get("alpha_base", 0.4), alpha_anchor=params.get("alpha_anchor", 0.9),
              beta=params.get("beta", 0.2), eta=CFG["cad"]["eta"], static_alpha=CFG["cad"]["static_alpha"],
              max_new_tokens=CFG["gen"]["max_new_tokens"])
    for k in range(0, len(todo), batch_size):
        batch = todo[k:k + batch_size]
        ctx = [chat_text(tok, prompt_for(sel["cond"], r, v)) for _, r, v in batch]
        unc = [chat_text(tok, bare_prior_prompt(r["lang"], r["L_target"])) for _, r, v in batch]
        anc = [anchor_token_ids(tok, r) for _, r, _ in batch]
        torch.cuda.synchronize()
        import time
        t = time.perf_counter()
        texts, info = contrastive_generate(model, tok, ctx, unc, anc, method, **kw)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / len(batch)
        for j, ((item, r, v), s) in enumerate(zip(batch, texts)):
            out.add(dict(item=item, uid=r["uid"], lang=r["lang"], variant=v, system=system, method=method,
                         params=kw, summary=s, mean_alpha=info["mean_alpha"][j], dola_layers=info["layers"],
                         latency_s=dt, latency_mode=f"batch{batch_size}", precision=desc,
                         input_fertility=n_tokens(tok, r["variants"][v]) / max(1, W(r["variants"][v])),
                         sum_tokens=n_tokens(tok, s), sum_words=W(s), peak_vram_gb=peak_vram_gb()))
        if (k // batch_size) % 10 == 0:
            print(f"[{system} shard {shard}] {k}/{len(todo)} :: {texts[0][:90]}", flush=True)
    out.close()
    del model
    free_gpu()


def tune_generate_job(shard, n_shards, **_):
    """Generate every grid config on the dev tuning subset (graded afterwards in a separate step)."""
    rng = random.Random(CFG["seed"])
    dev = load_set("dev")
    uids = []
    for lang in ("hi", "gu"):
        lr = sorted(r["uid"] for r in dev if r["lang"] == lang)
        uids += rng.sample(lr, min(CFG["cad"]["tune_docs"] // 2, len(lr)))
    for gi, (ab, aa, b) in enumerate(TUNE_GRID):
        decode_job(shard, n_shards, "anchor_cad_dola", f"tune_g{gi}", params=dict(alpha_base=ab, alpha_anchor=aa, beta=b),
                   sets=("dev",), variants=("native", "roman"), uids=uids)


def pick_params():
    from .stage2 import dev_grade, release_dev_router
    rows = []
    for gi, (ab, aa, b) in enumerate(TUNE_GRID):
        g = dev_grade(f"tune_g{gi}")
        rows.append(dict(grid=gi, alpha_base=ab, alpha_anchor=aa, beta=b, **{k: g[k] for k in ("score", "frac", "R_anchor", "rougeL")}))
    release_dev_router()
    best = max(rows, key=lambda r: (r["score"], r["rougeL"]))
    json_dump(dict(alpha_base=best["alpha_base"], alpha_anchor=best["alpha_anchor"], beta=best["beta"], grid=rows), TUNED)
    return best, rows


def samples_job(shard, n_shards, sets=("testbed", "natural_cm"), **_):
    """K−1 = 4 nucleus samples (T=0.7, top-p=0.9) per input with the selected Stage-2 setup."""
    from ..llm import chat_text, generate, load_causal
    sel = selected()
    K1 = CFG["K"] - 1
    out = JsonlAppender(DIRS["outputs"] / "stage3_samples.jsonl", key_fn=lambda r: r["item"], shard=shard)
    todo = [x for i, x in enumerate(_items(sets)) if i % n_shards == shard and x[0] not in out.done]
    if not todo:
        return
    model, tok, desc = load_causal(MODELS[sel["model_key"]], adapter=sel.get("adapter"))
    bs = max(1, CFG["gen"]["batch_size"] // 2)
    for k in range(0, len(todo), bs):
        batch = todo[k:k + bs]
        prompts = [chat_text(tok, prompt_for(sel["cond"], r, v)) for _, r, v in batch]
        res, dt, _ = generate(model, tok, prompts, max_new_tokens=CFG["gen"]["max_new_tokens"], do_sample=True,
                              num_return_sequences=K1, temperature=CFG["sample_T"], top_p=CFG["sample_top_p"],
                              seed=CFG["seed"] + k)
        for (item, r, v), cands in zip(batch, res):
            out.add(dict(item=item, uid=r["uid"], lang=r["lang"], variant=v, samples=cands,
                         latency_s=dt / len(batch)))
    out.close()
    del model
    free_gpu()


def cove_job(shard, n_shards, **_):
    from ..cove import cove_batch
    from ..llm import load_causal
    sel = selected()
    rng = random.Random(CFG["seed"])
    base = {g["item"]: g for g in jl_read_glob(DIRS["outputs"] / f"{sel['system']}.jsonl")}
    tb = _items(("testbed",), uids=ablation_uids())
    tb = rng.sample(tb, min(CFG["n_cove"], len(tb)))
    out = JsonlAppender(DIRS["outputs"] / "cove.jsonl", key_fn=lambda r: r["item"], shard=shard)
    todo = [x for i, x in enumerate(tb) if i % n_shards == shard and x[0] not in out.done and x[0] in base]
    if not todo:
        return
    model, tok, desc = load_causal(MODELS[sel["model_key"]], adapter=sel.get("adapter"))
    import time
    for k in range(0, len(todo), 4):
        batch = todo[k:k + 4]
        drafts = [base[i]["summary"] for i, _, _ in batch]
        torch.cuda.synchronize(); t = time.perf_counter()
        revised, traces = cove_batch(model, tok, [(r, v) for _, r, v in batch], drafts)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / len(batch)
        for (item, r, v), d, s, tr in zip(batch, drafts, revised, traces):
            out.add(dict(item=item, uid=r["uid"], lang=r["lang"], variant=v, system="cove", summary=s,
                         draft=d, qa=tr, latency_s=dt + base[item].get("latency_s", 0), sum_words=W(s)))
    out.close()
    del model
    free_gpu()


def ppl_job(shard, n_shards, systems, **_):
    from ..decoding import conditional_ppl
    from ..llm import chat_text, load_causal
    sel = selected()
    recs = {r["uid"]: r for r in load_set("testbed")}
    out = JsonlAppender(DIRS["eval"] / "ppl.jsonl", key_fn=lambda r: (r["system"], r["item"]), shard=shard)
    jobs = []
    for s in systems:
        for g in jl_read_glob(DIRS["outputs"] / f"{s}.jsonl"):
            if g["uid"] in recs and (s, g["item"]) not in out.done:
                jobs.append((s, g))
    jobs = [j for i, j in enumerate(jobs) if i % n_shards == shard]
    if not jobs:
        return
    model, tok, _ = load_causal(MODELS[sel["model_key"]], adapter=sel.get("adapter"))
    for s, g in jobs:
        r = recs[g["uid"]]
        p = chat_text(tok, prompt_for(sel["cond"], r, g["variant"]))
        out.add(dict(system=s, item=g["item"], ppl=conditional_ppl(model, tok, [p], [g["summary"]])[0]))
    out.close()
    del model
    free_gpu()


def lookback_job(shard, n_shards, systems, **_):
    from ..llm import chat_text, load_causal
    from ..lookback import sentence_features
    sel = selected()
    recs = {r["uid"]: r for s in ("testbed", "dev") for r in load_set(s)}
    keep = set(ablation_uids()) | {r["uid"] for r in load_set("dev")}
    out = JsonlAppender(DIRS["verify"] / "lookback_feats.jsonl", key_fn=lambda r: (r["system"], r["item"]), shard=shard)
    jobs = [(s, g) for s in systems for g in jl_read_glob(DIRS["outputs"] / f"{s}.jsonl")
            if g["uid"] in keep and (s, g["item"]) not in out.done and g["summary"]]
    jobs = [j for i, j in enumerate(jobs) if i % n_shards == shard]
    if not jobs:
        return
    model, tok, _ = load_causal(MODELS[sel["model_key"]], adapter=sel.get("adapter"), attn_implementation="eager")
    for s, g in jobs:
        r = recs[g["uid"]]
        p = chat_text(tok, prompt_for(sel["cond"], r, g["variant"]))
        try:
            feats = sentence_features(model, tok, p, g["summary"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        out.add(dict(system=s, item=g["item"], feats=[f.round(4).tolist() for f in feats]))
    out.close()
    del model
    free_gpu()
