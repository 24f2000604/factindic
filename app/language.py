"""Which of the five input kinds a document is. Script decides Hindi/Gujarati; Latin text is separated into
English, Romanized Hindi and Romanized Gujarati by function words, the project's Roman vocabulary and English frequency."""
from __future__ import annotations

import re

from factindic.lid import ROMAN_INDIC_STOP, _zipf, roman_vocab
from factindic.text import LATIN_RE, strip_punct

KINDS = {  # name shown to the user → (matrix language, input variant of the pipeline)
    "Hindi": ("hi", "native"), "Gujarati": ("gu", "native"), "Hinglish": ("hi", None), "Gujlish": ("gu", None),
    "English": ("en", "native"),
}
_DEVA, _GUJ = re.compile("[ऀ-ॿ]"), re.compile("[઀-૿]")
# spellings people really use; the project's lists hold the Romanizer's own
_HI = ROMAN_INDIC_STOP["hi"] | set("haan nahi nhi kyunki lekin agar mujhe hume usne unhone isse iske inke uske jisse hua hue hui "
                                   "diya liya gaye sakta sakte sakti chahiye wala wale wali yahan wahan kaha baad pehle saath taaki".split())
_GU = ROMAN_INDIC_STOP["gu"] | set("che chhe che. hatu hato hata hati nathi ane pan mate thi thayu thayo thaya karyu karyo kahyu hova "
                                   "tena tene teni temna temne amara tamara shu kem kyare jyare tyare pachi pahela sathe chhu chu rahyo rahi rahya aa aav aavi".split())
_SHARED = (_HI & _GU) | {"a", "e", "te", "to", "the", "me", "ma", "na", "no", "do", "so", "par", "us", "is", "in", "use", "log", "ek"}
_EN = set("the of and to in is was that for on with as by at from are were has have had been will which this it an or not be their its "
          "his her he she they who after also more than into about said".split())


def _latin_scores(words):
    hi = gu = en = 0.0
    for w in words:
        if w in _EN and w not in ("me", "to", "is", "in", "he", "the", "an", "at", "or", "has", "by", "said", "also"):
            en += 1.0
        elif w in ("the", "to", "is", "in", "an", "has"):       # also Romanized words
            en += 0.5
        if w not in _SHARED and w not in _EN:
            hi += 2.0 * (w in _HI)
            gu += 2.0 * (w in _GU)
        z = _zipf(w)
        h, g = roman_vocab("hi").get(w), roman_vocab("gu").get(w)
        if h or g:
            best = max(h[1] if h else 0.0, g[1] if g else 0.0)
            if best + 1.0 >= z and w not in _EN:
                d = (h[1] if h else 0.0) - (g[1] if g else 0.0)
                hi += max(0.0, min(d, 2.0)) * 0.5
                gu += max(0.0, min(-d, 2.0)) * 0.5
        elif z >= 3.0:
            en += 0.3
    return hi, gu, en


def detect(text: str) -> dict:
    """{'kind', 'lang', 'variant', 'latin_share'}"""
    cores = [c for c in (strip_punct(t)[1] for t in text.split()) if c]
    deva = sum(bool(_DEVA.search(c)) for c in cores)
    guj = sum(bool(_GUJ.search(c)) for c in cores)
    latin = sum(bool(LATIN_RE.search(c)) and not (_DEVA.search(c) or _GUJ.search(c)) for c in cores)
    total = max(1, deva + guj + latin)
    if (deva + guj) / total >= 0.35:
        lang, share = ("hi" if deva >= guj else "gu"), latin / total
        mixed = share >= 0.03
        kind = {"hi": ("Hindi", "Hinglish"), "gu": ("Gujarati", "Gujlish")}[lang][mixed]
        return dict(kind=kind, lang=lang, variant="script_mixed" if mixed else "native", latin_share=round(share, 3))
    words = [c.lower() for c in cores if LATIN_RE.search(c)]
    hi, gu, en = _latin_scores(words)
    n = max(1, len(words))
    if max(hi, gu) / n < 0.25 and en / n > 0.15 or max(hi, gu) / n < 0.08:
        return dict(kind="English", lang="en", variant="native", latin_share=1.0)
    lang = "hi" if hi >= gu else "gu"
    return dict(kind={"hi": "Hinglish", "gu": "Gujlish"}[lang], lang=lang, variant="roman", latin_share=round(latin / total, 3))


def resolve(text: str, kind: str | None) -> dict:
    """Detection, or the user's own choice; Hinglish and Gujlish are Romanized or script-mixed depending on the script."""
    d = detect(text)
    if not kind or kind == "Auto" or kind == d["kind"]:
        return d
    lang, variant = KINDS[kind]
    if variant is None or (variant == "native" and lang != "en"):
        latin = d["latin_share"] if d["variant"] != "roman" else 1.0
        variant = "roman" if latin > 0.65 else ("script_mixed" if latin >= 0.03 else "native")
    return dict(kind=kind, lang=lang, variant=variant, latin_share=d["latin_share"])
