"""Unicode-safe text helpers. Python's \\W matches Indic vowel signs, so punctuation is explicit."""
from __future__ import annotations

import html
import re
import unicodedata

from .config import LANGS

PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") | set("।॥“”‘’…–—₹«»·•")
INDIC_RE = re.compile(r"[ऀ-ॿ઀-૿]")
LATIN_RE = re.compile(r"[A-Za-z]")


def strip_punct(tok):
    i, j = 0, len(tok)
    while i < j and tok[i] in PUNCT:
        i += 1
    while j > i and tok[j - 1] in PUNCT:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def words(text):
    return [w for w in text.split() if strip_punct(w)[1]]


def W(text):
    return len(words(text))


_NORMALIZERS = {}


def clean(text, lang):
    """NFC + indic_nlp normalization (nukta/vowel canonicalization) + whitespace collapse."""
    if lang not in _NORMALIZERS:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        _NORMALIZERS[lang] = IndicNormalizerFactory().get_normalizer(lang)
    text = unicodedata.normalize("NFC", text or "")
    text = _NORMALIZERS[lang].normalize(text)
    return re.sub(r"\s+", " ", text).strip()


_BBC_FOOTER = re.compile(
    r"(\(बीबीसी (हिन्दी|हिंदी) के एंड्रॉएड ऐप.*$)|(\(બીબીસી ગુજરાતીની.*$)|"
    r"(आप हमें फ़ेसबुक.*$)|(તમે અમને ફેસબુક.*$)|(બીબીસી ગુજરાતીને.*$)", re.S)


def strip_boilerplate(text):
    return _BBC_FOOTER.sub("", text).strip()


_CODE = re.compile(r"/\*\s*\.jw-|\.jw-reset|comscore_jw|jwplayer\(|\bvar [A-Za-z_]\w*\s*=|function\s*\w*\s*\(|"
                   r"document\.(?:get|create|write)|googletag|\$\([\"'#.]")
_MARKUP = re.compile(r"<!--.*?-->|</?[A-Za-z][^<>]{0,300}>", re.S)
_LINK = re.compile(r"(?:https?://|www\.|pic\.twitter\.com/|t\.co/)[^\s\u0900-\u0AFF]*")


def strip_web(text):
    """Remove what ILSUM's crawler kept of the web page: player and gallery scripts, tags, HTML
    entities, links, and image credits. Scripts run from the first code word to the last '}' or ');'."""
    t = html.unescape(html.unescape(text or ""))
    m = _CODE.search(t)
    if m:
        end = max(t.rfind("}") + 1, t.rfind(");") + 2, m.end())
        t = t[:m.start()] + " " + t[end:]
    t = _LINK.sub(" ", _MARKUP.sub(" ", t))
    t = re.sub(r"Image\s*Source\s*:?", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def truncate_words(text, n):
    return " ".join(text.split()[:n])


# Letter names that spell initials (एस.पी., એલ.સી.બી) and abbreviations whose full stop ends no sentence.
INITIALS = set("""ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई जेड
એ બી સી ડી ઈ એફ જી એચ આઈ આઇ જે કે એલ એમ એન ઓ પી ક્યુ ક્યૂ આર એસ ટી યુ યૂ વી ડબલ્યુ ડબલ્યૂ એક્સ વાય ઝેડ""".split())
_WORDLIKE = set("ए ई ओ जी आई એ ઈ ઓ જી".split())          # also ordinary words, so they may end a sentence
_ABBR = set("डॉ डा रु रू प्रो पं मो ડૉ ડો ડા રૂ રુ પ્રો મો Dr Mr Mrs Ms Prof Rs St vs".split())
_INDIC = "\u0900-\u0963\u0971-\u0A7F\u0A80-\u0AFF"          # no dandas
_GLUED = re.compile(rf"(?<=[{_INDIC}][.।!?])(?=[{_INDIC}A-Za-z\"'“‘(])")


def _abbrev_end(part):
    """True if the final '.' of `part` belongs to an abbreviation, initials or a list number."""
    if not part.endswith("."):
        return False
    w = part.split()[-1].lstrip("\"'“‘(")[:-1]
    if w in _ABBR or re.fullmatch(r"\d{1,2}", part[:-1].strip()):
        return True
    segs = w.split(".")
    return segs[-1] in INITIALS and (len(segs) > 1 or w not in _WORDLIKE)


def split_sentences(text):
    """Splits after . । ! ? — also when the next word is glued on (છે.પાણી), and never inside
    abbreviations. Only chooses boundaries; the characters are not changed."""
    out, join = [], ""
    for part in re.split(r"(?<=[।.!?])\s+", (text or "").strip()):
        pieces, start = [], 0
        for m in _GLUED.finditer(part):
            if not _abbrev_end(part[start:m.start()]):
                pieces.append(part[start:m.start()])
                start = m.start()
        pieces.append(part[start:])
        for i, p in enumerate(pieces):
            p = f"{join} {p}".strip() if join and i == 0 else p.strip()
            join = ""
            if i == len(pieces) - 1 and p and _abbrev_end(p):
                join = p
            elif p:
                out.append(p)
    if join:
        out.append(join)
    return out


def script_fraction(text, lang):
    lo, hi = LANGS[lang]["block"]
    letters = [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def chunk_words(text, size, stride):
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)]
    out = [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + stride), stride)]
    return [c for c in out if c]


def chunk_sentences(sents, max_words):
    """Greedy sentence packing (FactCG / MiniCheck style chunking)."""
    chunks, cur, n = [], [], 0
    for s in sents:
        k = W(s)
        if cur and n + k > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += k
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [""]
