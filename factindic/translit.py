"""Romanization, script-mixing, cross-script mapping, and back-transliteration.

IndicXlit needs fairseq, which does not install on current Kaggle/Colab Python. Forward Romanization
here is rule-based ITRANS with schwa deletion (casual Hinglish/Gujlish style). Back-transliteration
uses IndicXlit when importable, otherwise the most frequent native spelling of that Roman form in the
training pool (data/roman_vocab.json), and an ITRANS reverse for unseen words.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .text import INDIC_RE, LATIN_RE, strip_punct

_CONS = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
GU_SUFFIX = sorted(["માં", "થી", "ને", "નો", "ની", "નું", "ના", "નાં", "એ"], key=len, reverse=True)


def _scheme(lang):
    from indic_transliteration import sanscript
    return {"hi": sanscript.DEVANAGARI, "gu": sanscript.GUJARATI}[lang]


_V, _C = "aAiIuUeEoO", "bcdfghjklmnpqrstvwxyzDTNLSJKGR"
_MEDIAL = re.compile(rf"(?<=[{_V}])([{_C}]h?)a(?=[{_C}]h?[{_V}])")
# ITRANS has no casual form for these: chandra vowels stayed in native script, nukta letters became ".D", "K"
_PRE = [(re.compile("[ऑ]"), "ओ"), (re.compile("[ॉॊ]"), "ो"), (re.compile("[ऍ]"), "ए"), (re.compile("[ॅ]"), "े"),
        (re.compile("[ઑ]"), "ઓ"), (re.compile("[ૉ]"), "ો"), (re.compile("[ઍ]"), "એ"), (re.compile("[ૅ]"), "ે"),
        (re.compile("([कखगडढयरन])़"), r"\1"), (re.compile("([કખગડઢ])઼"), r"\1")]


def _drop_medial_schwa(s):
    """VCaCV → VCCV, right to left (sabase → sabse, musalamAn → musalmAn)."""
    while True:
        hits = list(_MEDIAL.finditer(s))
        if not hits:
            return s
        m = hits[-1]
        s = s[:m.end() - 1] + s[m.end():]


def _casual(s):
    for a, b in [("R^i", "ri"), ("R^I", "ri"), ("L^i", "li"), (".N", "n"), ("~N", "n"), ("~n", "n"),
                 (".h", ""), (".a", ""), ("{}", ""), (".c", ""), ("^", ""), ("~", "n")]:
        s = s.replace(a, b)
    s = s.replace("|", ".")
    # final schwa goes (long A survives), except after a cluster ending in r/y/v (rashtra, rajya)
    if len(s) > 2 and s.endswith("a") and s[-2] in _CONS and not (s[-2] in "ryv" and s[-3] in _CONS):
        s = s[:-1]
    s = _drop_medial_schwa(s)
    s = s.replace("M", "n").replace("H", "h").replace("x", "ksh").lower()
    for v in "aiueo":
        s = re.sub(v + "{2,}", v, s)
    return s


@lru_cache(maxsize=500_000)
def _romanize_word(core, lang):
    """Romanizes each part of एस.डी. or पति-पत्नी alone, so schwa deletion sees its word ends."""
    parts = re.split(r"([^\u0900-\u0AFF\u200c\u200d]+)", core)
    if len(parts) > 1:
        return "".join(_romanize_word(p, lang) if INDIC_RE.search(p) else p for p in parts)
    from indic_transliteration import sanscript
    core = unicodedata.normalize("NFC", core)        # NFC keeps nukta as a separate mark (क़ → क + ़)
    for pat, rep in _PRE:
        core = pat.sub(rep, core)
    out = _casual(sanscript.transliterate(core, _scheme(lang), sanscript.ITRANS))
    return re.sub("[ऀ-૿]", "", out)


def romanize(text, lang):
    out = []
    for tok in text.split():
        pre, core, post = strip_punct(tok)
        if INDIC_RE.search(core):
            core = _romanize_word(core, lang)
        out.append((pre + core + post).replace("।", ".").replace("॥", "."))
    return " ".join(out)


def gu_to_deva(s):
    """Gujarati U+0A80–U+0AFF sits exactly 0x180 above Devanagari U+0900–U+097F."""
    return "".join(chr(ord(c) - 0x180) if 0x0A80 <= ord(c) <= 0x0AFF else c for c in s)


def deva_to_gu(s):
    return "".join(chr(ord(c) + 0x180) if 0x0900 <= ord(c) <= 0x097F else c for c in s)


def trim_gu(w):
    for suf in GU_SUFFIX:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            return w[: -len(suf)]
    return w


# --------------------------------------------------------------------------- script-mixed variant
LEX_RAW = {
    "hi": {"सरकार": "government", "प्रदूषण": "pollution", "स्तर": "level", "चुनाव": "election",
           "अस्पताल": "hospital", "मंत्री": "minister", "प्रधानमंत्री": "prime minister",
           "मुख्यमंत्री": "chief minister", "अदालत": "court", "न्यायालय": "court",
           "विश्वविद्यालय": "university", "छात्र": "student", "छात्रों": "students", "शिक्षा": "education",
           "स्वास्थ्य": "health", "बाज़ार": "market", "बाजार": "market", "कंपनी": "company",
           "रिपोर्ट": "report", "फ़ैसला": "decision", "फैसला": "decision", "फ़ैसले": "decision",
           "फैसले": "decision", "सुरक्षा": "security", "कानून": "law", "क़ानून": "law", "नीति": "policy",
           "अर्थव्यवस्था": "economy", "विकास": "development", "परियोजना": "project", "समस्या": "problem",
           "जानकारी": "information", "मामला": "case", "मामले": "case", "जांच": "investigation",
           "जाँच": "investigation", "घटना": "incident", "दुर्घटना": "accident", "मौत": "death",
           "देश": "country", "शहर": "city", "गांव": "village", "गाँव": "village", "बिजली": "electricity",
           "सड़क": "road", "किसान": "farmer", "किसानों": "farmers", "युद्ध": "war", "सेना": "army",
           "हमला": "attack", "हमले": "attack", "विरोध": "protest",
           "बयान": "statement", "अधिकारी": "officer", "अधिकारियों": "officers", "कर्मचारी": "employee",
           "कर्मचारियों": "employees", "नौकरी": "job", "वैज्ञानिक": "scientist", "शोध": "research",
           "अध्ययन": "study", "तकनीक": "technology", "फ़िल्म": "film", "फिल्म": "film",
           "खिलाड़ी": "player", "मैच": "match", "टीम": "team", "जलवायु": "climate",
           "तापमान": "temperature", "बारिश": "rain", "बाढ़": "flood", "वायरस": "virus",
           "वैक्सीन": "vaccine", "मरीज़": "patient", "मरीज": "patient", "मरीज़ों": "patients",
           "मरीजों": "patients", "डॉक्टर": "doctor", "इलाज": "treatment", "संक्रमण": "infection",
           "आंकड़े": "data", "आंकड़ों": "data", "सदस्य": "member", "संसद": "parliament",
           "पार्टी": "party", "नेता": "leader", "उम्मीदवार": "candidate", "वोट": "vote",
           "मतदान": "voting", "पुलिस": "police", "लगातार": "continuously"},
    "gu": {"સરકાર": "government", "પ્રદૂષણ": "pollution", "સ્તર": "level", "ચૂંટણી": "election",
           "હોસ્પિટલ": "hospital", "મંત્રી": "minister", "વડાપ્રધાન": "prime minister",
           "મુખ્યમંત્રી": "chief minister", "કોર્ટ": "court", "અદાલત": "court",
           "યુનિવર્સિટી": "university", "વિદ્યાર્થી": "student", "વિદ્યાર્થીઓ": "students",
           "શિક્ષણ": "education", "આરોગ્ય": "health", "બજાર": "market", "કંપની": "company",
           "રિપોર્ટ": "report", "નિર્ણય": "decision", "સુરક્ષા": "security", "કાયદો": "law",
           "કાયદા": "law", "નીતિ": "policy", "અર્થતંત્ર": "economy", "વિકાસ": "development",
           "પ્રોજેક્ટ": "project", "સમસ્યા": "problem", "માહિતી": "information", "કેસ": "case",
           "તપાસ": "investigation", "ઘટના": "incident", "અકસ્માત": "accident", "મૃત્યુ": "death",
           "દેશ": "country", "શહેર": "city", "ગામ": "village", "વીજળી": "electricity",
           "રસ્તો": "road", "ખેડૂત": "farmer", "ખેડૂતો": "farmers", "યુદ્ધ": "war", "સેના": "army",
           "હુમલો": "attack", "વિરોધ": "protest", "નિવેદન": "statement",
           "અધિકારી": "officer", "અધિકારીઓ": "officers", "કર્મચારી": "employee",
           "કર્મચારીઓ": "employees", "નોકરી": "job", "વૈજ્ઞાનિક": "scientist", "સંશોધન": "research",
           "અભ્યાસ": "study", "ટેકનોલોજી": "technology", "ફિલ્મ": "film", "ખેલાડી": "player",
           "મેચ": "match", "ટીમ": "team", "તાપમાન": "temperature", "વરસાદ": "rain", "પૂર": "flood",
           "વાયરસ": "virus", "રસી": "vaccine", "દર્દી": "patient", "દર્દીઓ": "patients",
           "ડૉક્ટર": "doctor", "સારવાર": "treatment", "ચેપ": "infection", "આંકડા": "data",
           "સભ્ય": "member", "સંસદ": "parliament", "પાર્ટી": "party", "નેતા": "leader",
           "ઉમેદવાર": "candidate", "મતદાન": "voting", "પોલીસ": "police", "સતત": "continuously"},
}
_LEX = {}


def lexicon(lang):
    if lang not in _LEX:
        from .text import clean
        _LEX[lang] = {clean(k, lang): v for k, v in LEX_RAW[lang].items()}
    return _LEX[lang]


def script_mix(text, lang, entities=()):
    """Embed Latin-script English content words inside the native matrix frame; keep Gujarati
    case enclitics as separate tokens ("pollution નું"). Words inside named entities stay as they are
    (विकास दुबे, સુરક્ષા દળ), so A_core does not change. Returns (text, embedded_english_terms)."""
    lex, out, emb = lexicon(lang), [], []
    toks = text.split()
    cores = [strip_punct(t)[1] for t in toks]
    keep = set()
    for e in entities:
        ew = [strip_punct(w)[1] for w in e.split()]
        for i in range(len(cores) - len(ew) + 1):
            if cores[i:i + len(ew)] == ew:
                keep.update(range(i, i + len(ew)))
    for i, tok in enumerate(toks):
        pre, core, post = strip_punct(tok)
        if i in keep:
            out.append(tok)
            continue
        rep = lex.get(core)
        if rep is None and lang == "gu":
            for suf in GU_SUFFIX:
                if core.endswith(suf) and core[: -len(suf)] in lex:
                    rep = lex[core[: -len(suf)]] + " " + suf
                    emb.append(lex[core[: -len(suf)]])
                    break
        elif rep is not None:
            emb.append(rep)
        out.append(pre + rep + post if rep else tok)
    return " ".join(out), sorted(set(emb))


# --------------------------------------------------------------------------- back-transliteration
_XLIT = {}


def _xlit_engine(lang):
    if lang in _XLIT:
        return _XLIT[lang]
    try:
        from ai4bharat.transliteration import XlitEngine
        _XLIT[lang] = XlitEngine(lang, beam_width=4, src_script_type="roman")
    except Exception:
        _XLIT[lang] = None
    return _XLIT[lang]


def back_transliterate_word(w, lang):
    eng = _xlit_engine(lang)
    if eng is not None:
        try:
            res = eng.translit_word(w, topk=1)
            cand = res[lang][0] if isinstance(res, dict) else res[0]
            if cand:
                return cand
        except Exception:
            pass
    from .lid import roman_vocab
    hit = roman_vocab(lang).get(w.lower())
    if hit:
        return hit[0]
    from indic_transliteration import sanscript
    s = w.lower()
    if s and s[-1] in _CONS.lower():
        s = s + "a"                     # restore the inherent vowel dropped by schwa deletion
    return sanscript.transliterate(s, sanscript.ITRANS, _scheme(lang))


def back_transliterate(text, lang, tags=None):
    """Romanized Indic tokens → native script; English tokens (per LID) are kept in Latin."""
    from .lid import tag_tokens
    toks = text.split()
    tags = tags or tag_tokens(text, lang)
    out = []
    for tok, tag in zip(toks, tags):
        pre, core, post = strip_punct(tok)
        if tag.endswith("_Latn") and not tag.startswith("eng") and LATIN_RE.search(core):
            core = back_transliterate_word(core, lang)
        out.append(pre + core + post)
    return " ".join(out)


def xlit_backend():
    return "IndicXlit" if _xlit_engine("hi") is not None else "pool-vocabulary + ITRANS-reverse"
