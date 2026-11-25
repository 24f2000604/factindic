"""Stage 3: Anchor-Aware Context-Contrasted Decoding (spec §8.3), batched dual-KV-cache loop.

score_t(w) = (1+α_t)·log P_L(w|D,A,y<t) − α_t·log P_L(w|∅,y<t) − β·log P_l*(w|D,A,y<t)
restricted to V_head = {w : P_L(w|D,A,y<t) ≥ η·max P_L}.

methods
  static_cad       α_t = α (fixed), β = 0                         (Shi et al., 2024)
  adacad           α_t = JSD₂(P(·|D) ‖ P(·|∅)) per token, β = 0     (Wang et al., 2025)
  dola             α_t = 0, β > 0, l* = argmax_l JSD(P_L ‖ P_l)    (Chuang et al., 2024)
  anchor_cad       α_t = α_anchor if argmax P_L ∈ V_anchor else α_base, β = 0
  anchor_cad_dola  anchor_cad + DoLa term                          (FactIndic, H4)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .llm import DEV, lm_parts

METHODS = ("static_cad", "adacad", "dola", "anchor_cad", "anchor_cad_dola")


def jsd(p, q, base2=True):
    m = 0.5 * (p + q)
    lm = m.clamp_min(1e-12).log()
    kl_pm = (p * (p.clamp_min(1e-12).log() - lm)).sum(-1)
    kl_qm = (q * (q.clamp_min(1e-12).log() - lm)).sum(-1)
    d = 0.5 * (kl_pm + kl_qm)
    return d / math.log(2) if base2 else d


def eos_ids(model, tok):
    ids = set()
    ge = getattr(model, "generation_config", None)
    e = getattr(ge, "eos_token_id", None) if ge is not None else None
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids |= set(e)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for t in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<turn|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def anchor_token_ids(tok, rec, variant_text=None):
    """V_anchor(D): subword ids of normalized anchors' surface forms, digits, and embedded English."""
    surf = [e["text"] for e in rec["anchors"].get("entities", [])]
    surf += rec["anchors"].get("numbers", []) + list(rec.get("english_terms", []))
    ids = set()
    for s in surf:
        for form in (s, " " + s):
            ids |= set(tok(form, add_special_tokens=False)["input_ids"])
    for d in "0123456789":
        ids |= set(tok(d, add_special_tokens=False)["input_ids"])
    # drop whitespace/punctuation-only pieces that would fire on every step
    junk = {i for i in ids if not tok.decode([i]).strip() or tok.decode([i]).strip() in ",.।:;-–()"}
    return ids - junk


def _cache(out):
    """transformers 5.24 renamed past_key_values to cache."""
    return out["cache"] if "cache" in out else out.past_key_values


def _cache_arg():
    import transformers
    from packaging.version import parse
    return "cache" if parse(transformers.__version__) >= parse("5.24.0") else "past_key_values"


CACHE_ARG = _cache_arg()


def _prefill(model, ids, mask, want_hidden):
    """Prefill all but the last prompt token, then step the last one (hidden states only for it)."""
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    past = None
    if ids.shape[1] > 1:                 # logits_to_keep: full-prompt logits over a 262k vocabulary need gigabytes
        out = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1],
                    use_cache=True, logits_to_keep=1)
        past = _cache(out)
    out = model(input_ids=ids[:, -1:], attention_mask=mask, position_ids=pos[:, -1:],
                **{CACHE_ARG: past}, use_cache=True, output_hidden_states=want_hidden)
    return out, pos[:, -1:]


def contrastive_generate(model, tok, ctx_texts, unc_texts, anchor_sets, method, **kw):
    """Returns (texts, info with per-row mean alpha and the premature layers). A batch that does not fit
    in memory is split in two."""
    try:
        return _contrastive_generate(model, tok, ctx_texts, unc_texts, anchor_sets, method, **kw)
    except torch.cuda.OutOfMemoryError:
        if len(ctx_texts) == 1:
            raise
    torch.cuda.empty_cache()
    m = len(ctx_texts) // 2
    a, b = (contrastive_generate(model, tok, ctx_texts[s], unc_texts[s], anchor_sets[s], method, **kw)
            for s in (slice(None, m), slice(m, None)))
    return a[0] + b[0], dict(mean_alpha=a[1]["mean_alpha"] + b[1]["mean_alpha"], layers=a[1]["layers"])


@torch.no_grad()
def _contrastive_generate(model, tok, ctx_texts, unc_texts, anchor_sets, method,
                         alpha_base=0.4, alpha_anchor=0.9, beta=0.3, eta=0.1, static_alpha=0.5,
                         layers=None, max_new_tokens=160):
    assert method in METHODS, method
    use_unc = method in ("static_cad", "adacad", "anchor_cad", "anchor_cad_dola")
    use_dola = method in ("dola", "anchor_cad_dola")
    if method == "dola":
        alpha_base = alpha_anchor = static_alpha = 0.0
    if not use_dola:
        beta = 0.0
    B = len(ctx_texts)
    tok.padding_side = "left"
    ec = tok(ctx_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    m_c = ec["attention_mask"]
    out_c, pos_c = _prefill(model, ec["input_ids"], m_c, use_dola)
    if use_unc:
        eu = tok(unc_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        m_u = eu["attention_mask"]
        out_u, pos_u = _prefill(model, eu["input_ids"], m_u, False)
    if use_dola:
        n_layers, norm, lm_head, softcap = lm_parts(model)
        layers = layers or sorted({max(1, int(n_layers * f)) for f in (0.35, 0.45, 0.55, 0.65)})
    V = out_c.logits.shape[-1]
    amask = torch.zeros((B, V), dtype=torch.bool, device=DEV)
    for b, s in enumerate(anchor_sets or [set()] * B):
        idx = [i for i in s if i < V]
        if idx:
            amask[b, idx] = True
    stop = torch.tensor(eos_ids(model, tok), device=DEV)
    finished = torch.zeros(B, dtype=torch.bool, device=DEV)
    gen = [[] for _ in range(B)]
    alpha_sum = torch.zeros(B, device=DEV)
    n_steps = torch.zeros(B, device=DEV)
    pad = tok.pad_token_id
    for t in range(max_new_tokens):
        logp_c = F.log_softmax(out_c.logits[:, -1].float(), -1)
        p_c = logp_c.exp()
        score = logp_c.clone()
        a = torch.zeros(B, device=DEV)
        if use_unc:
            logp_u = F.log_softmax(out_u.logits[:, -1].float(), -1)
            if method == "static_cad":
                a = torch.full((B,), static_alpha, device=DEV)
            elif method == "adacad":
                a = jsd(p_c, logp_u.exp())
            else:
                top = p_c.argmax(-1)
                is_anchor = amask[torch.arange(B, device=DEV), top]
                a = torch.where(is_anchor, torch.tensor(alpha_anchor, device=DEV),
                                torch.tensor(alpha_base, device=DEV))
            score = (1 + a)[:, None] * logp_c - a[:, None] * logp_u
        if use_dola and beta > 0:
            hs = out_c.hidden_states
            best_j = torch.full((B,), -1.0, device=DEV)
            early = torch.zeros_like(logp_c)
            for l in layers:
                lg = lm_head(norm(hs[l][:, -1])).float()
                if softcap:
                    lg = torch.tanh(lg / softcap) * softcap
                lp = F.log_softmax(lg, -1)
                j = jsd(p_c, lp.exp())
                upd = j > best_j
                best_j = torch.where(upd, j, best_j)
                early = torch.where(upd[:, None], lp, early)
            score = score - beta * early
        plaus = p_c >= eta * p_c.max(-1, keepdim=True).values
        score = score.masked_fill(~plaus, float("-inf"))
        nxt = score.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
        alive = ~finished
        alpha_sum += a * alive
        n_steps += alive.float()
        for b in torch.nonzero(alive).flatten().tolist():
            gen[b].append(int(nxt[b]))
        finished = finished | torch.isin(nxt, stop)
        if bool(finished.all()):
            break
        ones = torch.ones((B, 1), dtype=m_c.dtype, device=DEV)
        m_c = torch.cat([m_c, ones], 1)
        pos_c = pos_c + 1
        out_c = model(input_ids=nxt[:, None], attention_mask=m_c, position_ids=pos_c,
                      **{CACHE_ARG: _cache(out_c)}, use_cache=True, output_hidden_states=use_dola)
        if use_unc:
            m_u = torch.cat([m_u, ones], 1)
            pos_u = pos_u + 1
            out_u = model(input_ids=nxt[:, None], attention_mask=m_u, position_ids=pos_u,
                          **{CACHE_ARG: _cache(out_u)}, use_cache=True)
    stop_set = set(stop.tolist())
    texts = []
    from .llm import clean_output
    for g in gen:
        g = [x for x in g if x not in stop_set and x != pad]
        texts.append(clean_output(tok.decode(g, skip_special_tokens=True)))
    info = dict(mean_alpha=(alpha_sum / n_steps.clamp(min=1)).tolist(), layers=layers if use_dola else None)
    return texts, info


@torch.no_grad()
def conditional_ppl(model, tok, prompt_texts, outputs):
    """Perplexity of each output under the plain model given its prompt (fluency/drift proxy)."""
    res = []
    for p, o in zip(prompt_texts, outputs):
        if not o:
            res.append(float("nan"))
            continue
        p_ids = tok(p, add_special_tokens=False, return_tensors="pt")["input_ids"]
        o_ids = tok(o, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([p_ids, o_ids], 1).to(DEV)
        K = o_ids.shape[1] + 1
        try:
            lg = model(input_ids=ids, logits_to_keep=K).logits
        except TypeError:
            lg = model(input_ids=ids).logits[:, -K:]
        lp = F.log_softmax(lg[:, :-1].float(), -1)
        nll = -lp.gather(-1, ids[:, -o_ids.shape[1]:, None]).squeeze(-1).mean()
        res.append(float(nll.exp()))
    return res
