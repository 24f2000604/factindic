"""Stage 2: baselines and structured prompting.

notebook 02 : Seq2Seq baselines (IndicBARTSS, mT5-base) — train + beam-4 generation.
notebook 03 : Gemma-4 zero-shot conditions A/B1/B2/B3, faithfulness-weighted Gemma LoRA SFT,
              Llama-3.2-3B Cond A, latency probe, dev-set selection of the Stage-3 base.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from ..config import CFG, DIRS, IS_LOCAL, MODELS, max_new_tokens
from ..data import items, load_set
from ..io import JsonlAppender, free_gpu, jl_read_glob, json_dump, json_load, peak_vram_gb
from ..text import W


def _eval_items(sets=("testbed", "natural_cm"), variants=None):
    out = []
    for s in sets:
        out += items(load_set(s), variants)
    return out


# ============================================================================ Seq2Seq
def _s2s_io(name, tok, lang, article, summary=None):
    if name == "indicbartss":
        tag = f"<2{lang}>"
        end = tok.convert_tokens_to_ids(["</s>", tag])          # kept when a long article is cut
        src = tok(article, add_special_tokens=False, truncation=True, max_length=510)["input_ids"] + end
        tgt = end[1:] + tok(summary, add_special_tokens=False, truncation=True, max_length=126)["input_ids"] + end[:1] if summary else None
    else:
        src = tok("summarize: " + article, truncation=True, max_length=512)["input_ids"]
        tgt = tok(summary, truncation=True, max_length=128)["input_ids"] if summary else None
    return src, tgt


def _s2s_load(name, path=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    mid = MODELS[name]
    if name == "indicbartss":
        tok = AutoTokenizer.from_pretrained(mid, do_lower_case=False, use_fast=False, keep_accents=True)
    else:
        tok = AutoTokenizer.from_pretrained(mid)
    if name == "indicbartss":            # wrong accents or language tags would corrupt the baseline silently
        for t in ("<2hi>", "<2gu>", "</s>"):
            i = tok.convert_tokens_to_ids(t)
            got = tok(t, add_special_tokens=False)["input_ids"]
            assert got == [i] and i != tok.unk_token_id, f"IndicBARTSS tokenizer splits {t}: {got}"
        for t in ("हिन्दी की किताब", "ગુજરાતી ભાષાની ચોપડી"):
            back = tok.decode(tok(t, add_special_tokens=False)["input_ids"], clean_up_tokenization_spaces=False)
            assert back.split() == t.split(), f"IndicBARTSS tokenizer changes the text: {t!r} -> {back!r}"
    # explicit fp32: transformers 5 otherwise loads the checkpoint's own dtype, which breaks GradScaler
    model = AutoModelForSeq2SeqLM.from_pretrained(path or mid, dtype=torch.float32).cuda()
    return model, tok


def train_seq2seq(name, epochs=None, lr=None, bs=8):
    """Standard (unweighted) CE on the sanitized pool — the baseline has no faithfulness weighting."""
    from transformers import get_linear_schedule_with_warmup
    from ..io import jl_read
    out_dir = DIRS["models"] / name
    if (out_dir / "config.json").exists():
        return out_dir
    epochs = epochs or CFG["s2s_epochs"]
    lr = lr or (5e-4 if name == "indicbartss" else 3e-4)
    model, tok = _s2s_load(name)
    pool = jl_read(DIRS["data"] / "train_pool.jsonl")
    if len(pool) > CFG["n_s2s_train"]:
        pool = random.Random(CFG["seed"]).sample(pool, CFG["n_s2s_train"])
    data = []
    for r in pool:
        s, t = _s2s_io(name, tok, r["lang"], r["text"], r["summary"])
        data.append((s, t))
    amp = name == "indicbartss" and not IS_LOCAL     # fp16 with loss scaling; mT5 overflows in fp16 → fp32
    dtype = torch.bfloat16 if IS_LOCAL else torch.float16
    if name == "mt5":                    # Adam's two fp32 copies of 580 M weights do not fit
        opt = torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if IS_LOCAL or name == "mt5":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    steps = epochs * math.ceil(len(data) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pad = tok.pad_token_id
    model.train()
    rng = random.Random(CFG["seed"])
    t0, step = time.time(), 0
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = [data[j] for j in order[i:i + bs]]
            Ls, Lt = max(len(s) for s, _ in batch), max(len(t) for _, t in batch)
            src = torch.full((len(batch), Ls), pad); att = torch.zeros_like(src)
            tgt = torch.full((len(batch), Lt), pad)
            for k, (s, t) in enumerate(batch):
                src[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1; tgt[k, :len(t)] = torch.tensor(t)
            src, att, tgt = src.cuda(), att.cuda(), tgt.cuda()
            micro, n_tok = (2 if IS_LOCAL else 4 if name == "mt5" else bs), (tgt != pad).sum()
            for m in range(0, len(batch), micro):            # weighted by tokens, so the size of micro changes nothing
                s, a, y = src[m:m + micro], att[m:m + micro], tgt[m:m + micro]
                with torch.autocast("cuda", dtype=dtype, enabled=amp or IS_LOCAL):
                    if name == "indicbartss":
                        dec_in, labels = y[:, :-1], y[:, 1:].clone()
                        labels[labels == pad] = -100
                        loss = model(input_ids=s, attention_mask=a, decoder_input_ids=dec_in, labels=labels).loss
                    else:
                        labels = y.clone(); labels[labels == pad] = -100
                        loss = model(input_ids=s, attention_mask=a, labels=labels).loss
                scaler.scale(loss * (y != pad).sum() / n_tok).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 100 == 0:
                print(f"{name} ep{ep} step {step}/{steps} loss {loss.item():.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    model.config.use_cache = True
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model, opt
    free_gpu()
    return out_dir


def seq2seq_gen_job(shard, n_shards, name, **_):
    model, tok = _s2s_load(name, DIRS["models"] / name)
    model.eval()
    out = JsonlAppender(DIRS["outputs"] / f"{name}.jsonl", key_fn=lambda r: r["item"], shard=shard)
    todo = [x for i, x in enumerate(_eval_items()) if i % n_shards == shard and x[0] not in out.done]
    torch.cuda.reset_peak_memory_stats()
    for k in range(0, len(todo), 8):
        batch = todo[k:k + 8]
        enc = [_s2s_io(name, tok, r["lang"], r["variants"][v])[0] for _, r, v in batch]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), tok.pad_token_id); att = torch.zeros_like(ids)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e); att[i, :len(e)] = 1
        kw = dict(num_beams=4, max_length=96, early_stopping=True)
        if name == "indicbartss":
            # one language per call because the decoder start tag is language-specific
            outs = [None] * len(batch)
            for lang in {r["lang"] for _, r, _ in batch}:
                idx = [i for i, (_, r, _) in enumerate(batch) if r["lang"] == lang]
                start = tok.convert_tokens_to_ids(f"<2{lang}>")
                torch.cuda.synchronize(); t = time.perf_counter()
                g = model.generate(input_ids=ids[idx].cuda(), attention_mask=att[idx].cuda(),
                                   decoder_start_token_id=start, pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.convert_tokens_to_ids("</s>"), **kw)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t) / len(idx)
                for i, row in zip(idx, g):
                    outs[i] = (tok.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False), dt)
        else:
            torch.cuda.synchronize(); t = time.perf_counter()
            g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), **kw)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) / len(batch)
            outs = [(tok.decode(row, skip_special_tokens=True), dt) for row in g]
        for (item, r, v), e, (s, dt) in zip(batch, enc, outs):
            s = s.replace(f"<2{r['lang']}>", "").strip()
            out.add(dict(item=item, uid=r["uid"], lang=r["lang"], variant=v, system=name, summary=s,
                         latency_s=dt, latency_mode="batch8", input_tokens=len(e),
                         input_fertility=len(e) / max(1, W(r["variants"][v])),
                         sum_tokens=len(tok(s, add_special_tokens=False)["input_ids"]), sum_words=W(s),
                         peak_vram_gb=peak_vram_gb()))
    out.close()


# ============================================================================ LLM prompting
def prompt_for(cond, rec, variant):
    from ..prompts import CONDITIONS
    fn = CONDITIONS["B1" if cond == "B2" else cond]
    return fn(rec["lang"], variant, rec["variants"][variant], rec["L_target"], rec=rec)


def _missing_anchors(rec, draft):
    from ..normalizer import anchors, match_anchors, salient_anchors
    sal = salient_anchors(rec["anchors"], rec["text"], rec["lang"])
    a_s = anchors(draft, rec["lang"])
    have = {k for k, ok in match_anchors(a_s, rec["anchors"], rec["text"], rec["lang"]) if ok}
    have_text = draft
    surf = []
    for k in sal:
        if k in have:
            continue
        kind, val = k.split(":", 1)
        if kind == "ent":
            ent = next((e["text"] for e in rec["anchors"]["entities"] if e["key"] == val), None)
            if ent and ent not in have_text:
                surf.append(ent)
        elif kind == "num" and val not in have_text:
            surf.append(val)
        elif kind == "date":
            surf.append(val)
    return surf[:4]


def llm_gen_job(shard, n_shards, system, model_key, cond, sets=("testbed", "natural_cm"), variants=None,
                adapter=None, **_):
    from ..llm import chat_text, generate, load_causal, n_tokens
    from ..prompts import cond_B2_step2
    path = DIRS["outputs"] / f"{system}.jsonl"
    out = JsonlAppender(path, key_fn=lambda r: r["item"], shard=shard)
    all_items = []
    for s in sets:
        all_items += items(load_set(s), variants)
    todo = [x for i, x in enumerate(all_items) if i % n_shards == shard and x[0] not in out.done]
    if not todo:
        return
    model, tok, desc = load_causal(MODELS[model_key], adapter=adapter)
    bs = CFG["gen"].get("batch_size_by_model", {}).get(model_key, CFG["gen"]["batch_size"])
    torch.cuda.reset_peak_memory_stats()
    for k in range(0, len(todo), bs):
        batch = todo[k:k + bs]
        prompts = [chat_text(tok, prompt_for(cond, r, v)) for _, r, v in batch]
        res, dt, _ = generate(model, tok, prompts, max_new_tokens=max_new_tokens(model_key))
        summ = [x[0] for x in res]
        drafts = None
        if cond == "B2":     # 2-step CoD: densify with deterministic missing-anchor list
            drafts = summ
            p2 = [chat_text(tok, cond_B2_step2(r["lang"], v, r["variants"][v], r["L_target"], r, d,
                                               _missing_anchors(r, d))) for (_, r, v), d in zip(batch, drafts)]
            res2, dt2, _ = generate(model, tok, p2, max_new_tokens=max_new_tokens(model_key))
            summ, dt = [x[0] for x in res2], dt + dt2
        for j, ((item, r, v), p, s) in enumerate(zip(batch, prompts, summ)):
            row = dict(item=item, uid=r["uid"], lang=r["lang"], variant=v, system=system, summary=s,
                       latency_s=dt / len(batch), latency_mode=f"batch{bs}", precision=desc,
                       input_tokens=n_tokens(tok, r["variants"][v]),
                       input_fertility=n_tokens(tok, r["variants"][v]) / max(1, W(r["variants"][v])),
                       prompt_tokens=n_tokens(tok, p), sum_tokens=n_tokens(tok, s), sum_words=W(s),
                       peak_vram_gb=peak_vram_gb())
            if drafts:
                row["draft"] = drafts[j]
            out.add(row)
        if (k // bs) % 10 == 0:
            print(f"[{system} shard {shard}] {k}/{len(todo)} :: {summ[0][:90]}", flush=True)
    out.close()
    del model
    free_gpu()


def latency_probe(system, model_key, cond, adapter=None, n=None):
    """Per-summary wall-clock latency at batch size 1 (the mandatory inference-time metric)."""
    from ..llm import chat_text, generate, load_causal
    path = DIRS["outputs"] / f"latency_{system}.json"
    if path.exists():
        return json_load(path)
    n = n or CFG["gen"]["latency_probe_n"]
    its = _eval_items(("testbed",))
    rng = random.Random(CFG["seed"])
    its = rng.sample(its, min(n, len(its)))
    model, tok, desc = load_causal(MODELS[model_key], adapter=adapter)
    generate(model, tok, [chat_text(tok, prompt_for(cond, its[0][1], its[0][2]))], max_new_tokens=8)  # warm-up
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _, r, v in its:
        _, dt, _ = generate(model, tok, [chat_text(tok, prompt_for(cond, r, v))],
                            max_new_tokens=max_new_tokens(model_key))
        times.append(dt)
    res = dict(system=system, n=len(times), mean_s=float(np.mean(times)), p50_s=float(np.median(times)),
               p90_s=float(np.percentile(times, 90)), peak_vram_gb=peak_vram_gb(), precision=desc)
    json_dump(res, path)
    del model
    free_gpu()
    return res


def sft_examples(tok, n_per_lang=None):
    from ..prompts import cond_A
    from .stage1 import weighted_pool
    rows = weighted_pool()
    rng = random.Random(CFG["seed"])
    n = n_per_lang or CFG["n_sft_per_lang"]
    picked = []
    for lang in ("hi", "gu"):
        lr = [r for r in rows if r["lang"] == lang]
        if len(lr) < n:
            print(f"WARNING: only {len(lr)} kept weighted pairs for {lang} (< {n}); raise n_score_per_lang")
        picked += rng.sample(lr, min(n, len(lr)))
    for r in picked:
        r["article"] = r["text"]
        r["n_words"] = int(np.clip(W(r["summary"]), 25, 45))
    from ..sft import build_examples
    return build_examples(tok, picked, lambda r, art: cond_A(r["lang"], "native", art, r["n_words"]),
                          CFG["sft"]["max_len"])


# ============================================================================ dev selection
def dev_grade(system, tau=None):
    """Dev-only tuning signal (NOT the evaluation grader): router support on dev summaries, salient
    anchor recall, ROUGE-L. Score = mean(frac_supported × R_anchor_salient)."""
    from ..claims import sentence_claims
    from ..metrics import rouge
    from ..normalizer import anchor_recall_salient, anchors, salient_anchors
    path = DIRS["verify"] / f"dev_{system}.jsonl"
    tau = tau or CFG["tau_ver"]
    gens = jl_read_glob(DIRS["outputs"] / f"{system}.jsonl")
    recs = {r["uid"]: r for r in load_set("dev")}
    gens = [g for g in gens if g["uid"] in recs]
    out = JsonlAppender(path, key_fn=lambda r: r["item"])
    todo = [g for g in gens if g["item"] not in out.done]
    if todo:
        router = _dev_router()
        for g in todo:
            rec = recs[g["uid"]]
            cl = sentence_claims(g["summary"])
            v = router.verify(rec, g["variant"], cl) if cl else []
            frac = sum(x["p"] >= tau for x in v) / len(v) if v else 0.0
            sal = salient_anchors(rec["anchors"], rec["text"], rec["lang"])
            ra = anchor_recall_salient(anchors(g["summary"], rec["lang"]), rec["anchors"], sal, rec["text"], rec["lang"])
            out.add(dict(item=g["item"], system=system, variant=g["variant"], lang=g["lang"], frac=frac,
                         R_anchor=ra, score=frac * ra, rougeL=rouge(rec["reference"], g["summary"])["rougeL"],
                         words=W(g["summary"])))
    out.close()
    rows = jl_read_glob(path)
    return dict(system=system, n=len(rows), score=float(np.mean([r["score"] for r in rows])) if rows else float("nan"),
                frac=float(np.mean([r["frac"] for r in rows])) if rows else float("nan"),
                R_anchor=float(np.mean([r["R_anchor"] for r in rows])) if rows else float("nan"),
                rougeL=float(np.mean([r["rougeL"] for r in rows])) if rows else float("nan"))


_ROUTER = None


def _dev_router():
    global _ROUTER
    if _ROUTER is None:
        from ..mt import Translator
        from ..verify import MiniCheck, NativeNLI, Router
        _ROUTER = Router(Translator(cache_name="mt_cache_dev.jsonl"), MiniCheck(), NativeNLI())
    return _ROUTER


def release_dev_router():
    global _ROUTER
    if _ROUTER is not None:
        _ROUTER.mt.close()
    _ROUTER = None
    free_gpu()
