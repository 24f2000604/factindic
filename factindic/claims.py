"""Claim units. Default: native summary sentences (MiniCheck/FactCG are sentence-level verifiers).
Optional: Qwen2.5-1.5B molecular decomposition of the English projection (Gunjal & Durrett, 2024)."""
from __future__ import annotations

import re

from .text import W, split_sentences


def sentence_claims(summary, min_words=3):
    return [c for c in split_sentences(summary) if W(c) >= min_words]


MOLECULAR_PROMPT = (
    "Split the sentence into the smallest set of self-contained factual claims. Each claim must be "
    "understandable on its own (replace pronouns with the names they refer to), keep multi-word "
    "predicates intact, and add nothing that is not in the sentence. One claim per line, no "
    "numbering.\n\nSentence: {s}\nClaims:")


class MolecularDecomposer:
    def __init__(self):
        from .config import MODELS
        from .llm import load_causal
        self.model, self.tok, _ = load_causal(MODELS["qwen"])

    def __call__(self, sentences_en, bs=8):
        from .llm import chat_text, generate
        out = []
        for i in range(0, len(sentences_en), bs):
            batch = sentences_en[i:i + bs]
            res, _, _ = generate(self.model, self.tok,
                                 [chat_text(self.tok, MOLECULAR_PROMPT.format(s=s)) for s in batch],
                                 max_new_tokens=128)
            for s, r in zip(batch, res):
                raw = r[0] if r else ""
                lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in raw.split("\n")]
                lines = [l for l in lines if len(l.split()) >= 3]
                out.append(lines or [s])
        return out
