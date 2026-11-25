"""Word-level language ID and the Code-Mixing Index (Gambäck & Das).

AI4Bharat IndicLID needs fairseq + large fastText assets; this lightweight tagger uses script for
native tokens and an English frequency lexicon (wordfreq) plus a romanized-Indic function-word list
for Latin tokens. Tags: hin_Deva, guj_Gujr, hin_Latn, guj_Latn, eng_Latn, univ (digits/punct).
"""
from __future__ import annotations

from .text import INDIC_RE, LATIN_RE, strip_punct

ROMAN_INDIC_STOP = {
    "hi": set("hai hain ka ki ke ko se mein me main aur ya bhi nahi nahin tha thi the ho raha rahi rahe "
              "kar karna kiya gaya gayi wajah liye par pe jo jab tab kya kyun ek do teen log sab koi "
              "apne apna unke unki uska uski yeh ye woh vo hum tum aap".split()),
    "gu": set("chhe che chhu hatu hati hata ane na ni nu no ne ma thi pan ke e aa te tame ame hu "
              "thai thay rahyu rahi karvu karyu kar ek be tran loko badha koi lidhe mate sathe "
              "upar niche satat".split()),
}

_ZIPF = None


def _zipf(w):
    global _ZIPF
    if _ZIPF is None:
        try:
            from wordfreq import zipf_frequency
            _ZIPF = zipf_frequency
        except ImportError:
            _ZIPF = False
    return _ZIPF(w, "en") if _ZIPF else 0.0


_VOCAB = {}


def roman_vocab(lang):
    """{roman form: [most frequent native spelling, zipf]} built by notebook 01 from the training pool."""
    if lang not in _VOCAB:
        from .config import DIRS
        from .io import json_load
        _VOCAB.update(json_load(DIRS["data"] / "roman_vocab.json", {"hi": {}, "gu": {}}))
    return _VOCAB[lang]


def tag_word(tok, lang, romanized=False):
    core = strip_punct(tok)[1]
    if not core or core.replace(".", "").replace(",", "").isdigit():
        return "univ"
    if INDIC_RE.search(core):
        return {"hi": "hin_Deva", "gu": "guj_Gujr"}[lang]
    if LATIN_RE.search(core):
        w = core.lower()
        if w in ROMAN_INDIC_STOP[lang]:
            return f"{'hin' if lang == 'hi' else 'guj'}_Latn"
        if romanized:                    # में → "men": in Romanized text the Indic reading wins when it is at least as common
            hit = roman_vocab(lang).get(w)
            if hit and hit[1] + 1.0 >= _zipf(w):
                return f"{'hin' if lang == 'hi' else 'guj'}_Latn"
        return "eng_Latn" if _zipf(w) >= 3.0 else f"{'hin' if lang == 'hi' else 'guj'}_Latn"
    return "univ"


def tag_tokens(text, lang):
    toks = text.split()
    latin = sum(bool(LATIN_RE.search(t)) for t in toks)
    indic = sum(bool(INDIC_RE.search(t)) for t in toks)
    return [tag_word(t, lang, romanized=latin > indic) for t in toks]


def mixed_cmi(text, lang, max_run):
    """CMI without runs of more than max_run Latin-script words: a caption, tweet or headline in English
    is a separate passage, not mixing inside a sentence. Digits and punctuation do not end a run."""
    toks, keep, run = text.split(), [], []
    for t in toks + ["\u0915"]:
        if LATIN_RE.search(t) and not INDIC_RE.search(t):
            run.append(t)
        elif INDIC_RE.search(t):
            keep += (run if len(run) <= max_run else []) + [t]
            run = []
    return cmi(" ".join(keep[:-1]), lang)


def cmi(text, lang, tags=None):
    """CMI = 100·(1 − max_L w(L) / (N − u)) with languages {Indic (either script), English}."""
    tags = tags or tag_tokens(text, lang)
    n = len(tags)
    u = sum(t == "univ" for t in tags)
    if n <= u:
        return 0.0
    eng = sum(t == "eng_Latn" for t in tags)
    ind = n - u - eng
    return 100.0 * (1 - max(eng, ind) / (n - u))
