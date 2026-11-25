"""Lookback Lens (Chuang et al., 2024) on Gemma-4: per-head lookback ratios per summary sentence.

LR(l,h)_t = Σ_ctx Attn(t,i) / (Σ_ctx Attn(t,i) + Σ_new Attn(t,j)); averaged over each sentence's
tokens, concatenated with the sentence's relative position. An L2 logistic regression detects
unsupported sentences (labels: the held-out verifier).
"""
from __future__ import annotations

import numpy as np
import torch

from .llm import DEV
from .text import split_sentences


@torch.no_grad()
def sentence_features(model, tok, prompt_text, summary):
    sents = split_sentences(summary)
    if not sents:
        return []
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    pieces, spans, cur = [], [], len(p_ids)
    for i, s in enumerate(sents):
        ids = tok((" " if i else "") + s, add_special_tokens=False)["input_ids"]
        pieces += ids
        spans.append((cur, cur + len(ids)))
        cur += len(ids)
    ids = torch.tensor([p_ids + pieces], device=DEV)
    out = model(input_ids=ids, output_attentions=True, logits_to_keep=1)
    P = len(p_ids)
    feats = []
    att = [a[0].float() for a in out.attentions if a is not None]   # each [H, T, T]
    for k, (a, b) in enumerate(spans):
        per_layer = []
        for A in att:
            rows = A[:, a:b, :]                          # [H, n, T]
            ctx = rows[:, :, :P].sum(-1)
            new = torch.stack([rows[:, i, P:a + i].sum(-1) for i in range(b - a)], 1)
            lr = (ctx / (ctx + new).clamp_min(1e-9)).mean(1)   # [H]
            per_layer.append(lr)
        f = torch.cat(per_layer).cpu().numpy()
        feats.append(np.concatenate([f, [k / max(1, len(spans) - 1)]]))
    return feats


def train_detector(X, y, C=1.0, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", max_iter=2000,
                                                             class_weight="balanced", random_state=seed))
    clf.fit(X, y)
    return clf
