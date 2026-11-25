"""Deterministic Numeral, Date, and Cross-Script Entity Normalizer (A-5) and the Core Anchor Set.

A_core(D) = {Norm(E_D), Norm(N_D), Norm(T_D)} is represented as a set of canonical strings:
    "num:150000", "date:08-15", "date:2019-08", "ent:<vowel-collapsed ascii key>".
Embedded English terms X_D are tracked separately so A_core stays invariant across variants.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .text import INDIC_RE, strip_punct
from .translit import GU_SUFFIX, gu_to_deva, trim_gu

DIGITS = str.maketrans("०१२३४५६७८९૦૧૨૩૪૫૬૭૮૯", "01234567890123456789")
_nfc = lambda s: unicodedata.normalize("NFC", s)

MULT = {_nfc(k): v for k, v in {
    "सौ": 1e2, "સો": 1e2, "hundred": 1e2, "हज़ार": 1e3, "हजार": 1e3, "હજાર": 1e3, "hazar": 1e3,
    "hazaar": 1e3, "thousand": 1e3, "लाख": 1e5, "લાખ": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5,
    "करोड़": 1e7, "करोड": 1e7, "કરોડ": 1e7, "crore": 1e7, "crores": 1e7, "karod": 1e7,
    "million": 1e6, "मिलियन": 1e6, "મિલિયન": 1e6, "billion": 1e9, "अरब": 1e9, "અબજ": 1e9,
    "बिलियन": 1e9, "બિલિયન": 1e9}.items()}
WORDNUM = {_nfc(k): v for k, v in {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "बीस": 20, "पचास": 50, "डेढ़": 1.5, "ढाई": 2.5, "सवा": 1.25,
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8, "નવ": 9,
    "દસ": 10, "વીસ": 20, "પચાસ": 50, "દોઢ": 1.5, "અઢી": 2.5, "સવા": 1.25,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "dedh": 1.5, "dhai": 2.5}.items()}
MONTHS = {_nfc(k): v for k, v in {
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7,
    "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11,
    "दिसंबर": 12, "दिसम्बर": 12,
    "જાન્યુઆરી": 1, "ફેબ્રુઆરી": 2, "માર્ચ": 3, "એપ્રિલ": 4, "મે": 5, "જૂન": 6, "જુલાઈ": 7,
    "ઑગસ્ટ": 8, "ઓગસ્ટ": 8, "સપ્ટેમ્બર": 9, "ઑક્ટોબર": 10, "ઓક્ટોબર": 10, "નવેમ્બર": 11,
    "ડિસેમ્બર": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12}.items()}


def _lookup(table, w):
    w = _nfc(w).lower()
    if w in table:
        return table[w]
    for suf in GU_SUFFIX + ["ों", "के", "की", "का"]:
        if w.endswith(suf) and w[: -len(suf)] in table:
            return table[w[: -len(suf)]]
    return None


def _fmt(v):
    return ("%.4f" % v).rstrip("0").rstrip(".")


def _tokens(text):
    toks = [strip_punct(t)[1] for t in (text or "").translate(DIGITS).split()]
    return [t for t in toks if t]


_NUM_RE = re.compile(r"[₹$]?(\d[\d,]*(?:\.\d+)?)")


def numbers(text):
    """Canonical numeric values; bare एक/એક ('a/one') counts only before a multiplier."""
    toks, vals, i = _tokens(text), set(), 0
    while i < len(toks):
        w, v = toks[i], None
        m = _NUM_RE.match(w)
        nxt = _lookup(MULT, toks[i + 1]) if i + 1 < len(toks) else None
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                v = None
        elif _lookup(WORDNUM, w) is not None and nxt is not None:
            v = _lookup(WORDNUM, w)
        if v is not None:
            if nxt is not None:
                v *= nxt
                i += 1
            vals.add(_fmt(v))
        i += 1
    return vals


def dates(text):
    """'15 अगस्त' → 08-15; 'अगस्त 2019' → 2019-08; '15 August 2019' → 2019-08-15."""
    toks, out = _tokens(text), set()
    for i, w in enumerate(toks):
        mo = _lookup(MONTHS, w)
        if mo is None:
            continue
        day = year = None
        if i > 0 and re.fullmatch(r"\d{1,2}", toks[i - 1]) and 1 <= int(toks[i - 1]) <= 31:
            day = int(toks[i - 1])
        if i + 1 < len(toks):
            m = re.match(r"(\d{4})", toks[i + 1])
            if m and 1900 <= int(m.group(1)) <= 2100:
                year = int(m.group(1))
        if day and year:
            out.add(f"{year:04d}-{mo:02d}-{day:02d}")
        elif day:
            out.add(f"{mo:02d}-{day:02d}")
        elif year:
            out.add(f"{year:04d}-{mo:02d}")
    return out


def ent_key(word, lang):
    """1-to-1 cross-script key: trim Gujarati enclitics, map Gujarati→Devanagari by the 0x180 block
    offset, romanize, lowercase ASCII, collapse vowels (keep the first character)."""
    from .translit import romanize
    w = strip_punct(word)[1]
    if not w:
        return ""
    if INDIC_RE.search(w):
        if re.search(r"[઀-૿]", w):
            w = gu_to_deva(trim_gu(w))
        r = romanize(w, "hi")
    else:
        r = w.lower()
    r = re.sub(r"[^a-z0-9]", "", r)
    return (r[0] + re.sub(r"[aeiouy]", "", r[1:])) if r else ""


def entity_keys(entity_text, lang):
    return [k for k in (ent_key(w, lang) for w in entity_text.split()) if k]


def _close(k, keys):
    if k in keys:
        return True
    if len(k) <= 2:
        return False
    from rapidfuzz.distance import Levenshtein
    return any(abs(len(k) - len(s)) <= 1 and Levenshtein.distance(k, s, score_cutoff=1) <= 1 for s in keys)


def anchors(text, lang, entities=None, english_terms=()):
    """entities: list of entity surface strings (from ner.py). Returns a dict for JSONL storage."""
    ents = []
    for e in entities or []:
        ks = entity_keys(e, lang)
        if ks:
            ents.append(dict(text=e, key=" ".join(ks)))
    return dict(numbers=sorted(numbers(text)), dates=sorted(dates(text)), entities=ents,
                english=sorted(set(english_terms)))


def anchor_set(a):
    s = {f"num:{n}" for n in a.get("numbers", [])} | {f"date:{d}" for d in a.get("dates", [])}
    s |= {f"ent:{e['key']}" for e in a.get("entities", [])}
    return s


def match_anchors(a_s, a_d, d_text=None, lang=None):
    """Which anchors of summary S are supported by D. Entities match if every word key is within
    edit distance ≤ 1 of an entity word key of D or of any word key of D's text (NER misses)."""
    d_nums, d_dates = set(a_d.get("numbers", [])), set(a_d.get("dates", []))
    d_keys = {k for e in a_d.get("entities", []) for k in e["key"].split()}
    if d_text is not None:
        d_keys |= {k for k in (ent_key(w, lang) for w in d_text.split()) if k}
    out = []
    for n in a_s.get("numbers", []):
        out.append(("num:" + n, n in d_nums))
    for d in a_s.get("dates", []):
        out.append(("date:" + d, d in d_dates or any(x.endswith(d) or d.endswith(x) for x in d_dates)))
    for e in a_s.get("entities", []):
        out.append(("ent:" + e["key"], all(_close(k, d_keys) for k in e["key"].split())))
    return out


def anchor_prf(a_s, a_d, a_r, d_text=None, lang=None):
    """P = |A(S)∩A(D)|/|A(S)|; R = |A(S)∩A*(R)|/|A*(R)| with A*(R) = A(R)∩A(D)."""
    sup = match_anchors(a_s, a_d, d_text, lang)
    p = sum(ok for _, ok in sup) / len(sup) if sup else float("nan")
    r_in_d = [k for k, ok in match_anchors(a_r, a_d, d_text, lang) if ok]
    if not r_in_d:
        r = float("nan")
    else:
        s_keys = [k for k, ok in sup if ok]
        def hit(rk):
            if rk in s_keys:
                return True
            if rk.startswith("ent:"):
                ks = {x for sk in s_keys if sk.startswith("ent:") for x in sk[4:].split()}
                return all(_close(x, ks) for x in rk[4:].split())
            return False
        r = sum(hit(rk) for rk in r_in_d) / len(r_in_d)
    f = 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan")
    return dict(P_anchor=p, R_anchor=r, F1_anchor=f, n_anchor_S=len(sup), n_anchor_R=len(r_in_d),
                n_unsup_num=sum(1 for k, ok in sup if k.startswith("num:") and not ok),
                n_unsup_ent=sum(1 for k, ok in sup if k.startswith("ent:") and not ok))


def salient_anchors(a_d, d_text, lang, top=6):
    """Most frequent source anchors (ties → earlier first mention), used by the Stage-4 J score."""
    keys = sorted(anchor_set(a_d))           # a fixed order for ties, whatever the hash seed
    if not keys:
        return []
    toks = d_text.split()
    def first_pos_and_count(k):
        kind, val = k.split(":", 1)
        if kind == "num":
            pos = [i for i, t in enumerate(toks) if val in numbers(t)]
        elif kind == "ent":
            first = val.split()[0]
            pos = [i for i, t in enumerate(toks) if ent_key(t, lang) == first]
        else:
            pos = []
        return (-len(pos), pos[0] if pos else 10 ** 6)
    return sorted(keys, key=first_pos_and_count)[:top]


def anchor_recall_salient(a_s, a_d, salient, d_text, lang):
    if not salient:
        return 1.0
    sup = {k for k, ok in match_anchors(a_s, a_d, d_text, lang) if ok}
    s_ent = {x for k in sup if k.startswith("ent:") for x in k[4:].split()}
    hit = 0
    for k in salient:
        if k in sup:
            hit += 1
        elif k.startswith("ent:") and all(_close(x, s_ent) for x in k[4:].split()):
            hit += 1
    return hit / len(salient)


def anchor_density(text):
    return len(numbers(text)) + len(dates(text))
