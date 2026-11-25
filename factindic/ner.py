"""Named entities in native script (IndicNER) and Latin script (wikineural), offset-safe.

The IndicNER tokenizer strips Indic vowel signs from `entity["word"]` (वेस्टइंडीज → सटइडीज), which
silently broke anchor matching in H1 run 1. Spans are therefore cut from the original text by
character offsets, widened to whole words, and stripped of outer punctuation.
"""
from __future__ import annotations

from .text import INDIC_RE, LATIN_RE, split_sentences, strip_punct

KEEP = {"PER", "LOC", "ORG"}


def _span(S, a, b):
    while a > 0 and not S[a - 1].isspace():
        a -= 1
    while b < len(S) and not S[b].isspace():
        b += 1
    return strip_punct(S[a:b].strip())[1]


class Tagger:
    def __init__(self, device=0, latin=True):
        from transformers import pipeline
        from .config import MODELS
        self.native = pipeline("token-classification", model=MODELS["indicner"],
                               aggregation_strategy="first", device=device)
        self.latin = (pipeline("token-classification", model=MODELS["latin_ner"],
                               aggregation_strategy="first", device=device) if latin else None)

    def _run(self, pipe, text):
        out = []
        for sent in split_sentences(text) or [text]:
            if not sent.strip():
                continue
            for e in pipe(sent[:2000]):
                grp = e.get("entity_group", "").replace("B-", "").replace("I-", "")
                if grp in KEEP:
                    s = _span(sent, e["start"], e["end"])
                    if s:
                        out.append(s)
        return out

    def entities(self, text):
        ents = []
        if INDIC_RE.search(text):
            ents += self._run(self.native, text)
        if self.latin is not None and LATIN_RE.search(text):
            ents += [e for e in self._run(self.latin, text) if LATIN_RE.search(e)]
        return list(dict.fromkeys(ents))
