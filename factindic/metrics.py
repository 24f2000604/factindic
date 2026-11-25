"""Evaluation metrics (spec §10.2): claim-level and anchor-level P/R/F1, Indic-tokenized ROUGE,
rescaled IndicBERTv2 BERTScore (semantic similarity only), length / compression."""
from __future__ import annotations

import random

import numpy as np
import torch

from .text import PUNCT, W


class IndicTok:
    def tokenize(self, text):
        from indicnlp.tokenize import indic_tokenize
        return [t.lower() for t in indic_tokenize.trivial_tokenize(text or "")
                if not all(ch in PUNCT for ch in t)]


_ROUGE = None


def rouge(ref, hyp):
    """Default rouge-score strips non-ASCII; an Indic tokenizer is injected (0–100 scale)."""
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=IndicTok())
    if not hyp:
        return dict(rouge1=0.0, rouge2=0.0, rougeL=0.0)
    s = _ROUGE.score(ref, hyp)
    return {k: 100 * v.fmeasure for k, v in s.items()}


class BertScorer:
    """Greedy cosine matching on a middle-upper IndicBERTv2 layer, rescaled with a baseline b
    estimated from random unrelated (candidate, reference) pairs: F' = (F − b)/(1 − b)."""

    def __init__(self, layer=9):
        from transformers import AutoModel, AutoTokenizer
        from .config import MODELS
        self.tok = AutoTokenizer.from_pretrained(MODELS["indicbert"])
        self.model = AutoModel.from_pretrained(MODELS["indicbert"], output_hidden_states=True, dtype=torch.float32).cuda().eval()
        self.layer = layer
        self.baseline = {}

    @torch.no_grad()
    def _emb(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        hs = self.model(**enc).hidden_states[self.layer]
        hs = torch.nn.functional.normalize(hs.float(), dim=-1)
        special = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
        for sid in self.tok.all_special_ids:
            special |= enc["input_ids"] == sid
        mask = enc["attention_mask"].bool() & ~special
        return [hs[i][mask[i]] for i in range(len(texts))]

    def _f(self, cands, refs, bs=32):
        out = []
        for i in range(0, len(cands), bs):
            ec, er = self._emb(cands[i:i + bs]), self._emb(refs[i:i + bs])
            for c, r in zip(ec, er):
                if len(c) == 0 or len(r) == 0:
                    out.append(0.0)
                    continue
                sim = c @ r.T
                p = sim.max(1).values.mean().item()
                rr = sim.max(0).values.mean().item()
                out.append(2 * p * rr / (p + rr) if p + rr > 0 else 0.0)
        return out

    def fit_baseline(self, lang, texts, n=500, seed=42):
        rng = random.Random(seed)
        a = [rng.choice(texts) for _ in range(n)]
        b = [rng.choice(texts) for _ in range(n)]
        self.baseline[lang] = float(np.mean(self._f(a, b)))
        return self.baseline[lang]

    def score(self, cands, refs, lang):
        f = self._f([c or "" for c in cands], refs)
        b = self.baseline.get(lang, 0.0)
        return [(x - b) / (1 - b) for x in f]


def fact_prf(p_claims, r_grounded_hits):
    """P_fact = mean I_eval(D, c_i); R_fact = mean I_eval(S, r_j) over C_grounded(R)."""
    P = float(np.mean(p_claims)) if len(p_claims) else float("nan")
    R = float(np.mean(r_grounded_hits)) if len(r_grounded_hits) else float("nan")
    F = 2 * P * R / (P + R) if (P == P and R == R and P + R > 0) else (0.0 if P == 0 or R == 0 else float("nan"))
    return P, R, F


def compression_ratio(summary, source):
    return 100.0 * W(summary) / max(1, W(source))
