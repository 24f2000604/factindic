"""Stage 2 prompt conditions (spec §8.2): A standard; B1 +anchor checklist; B2 +2-step Chain of
Density; B3 +matrix-verb hint. All conditions output the matrix language in native script."""
from __future__ import annotations

from .config import LANGS

VARIANT_NOTE = {
    "native": "",
    "script_mixed": "The article mixes {L} with English words in Latin script.",
    "roman": "The article is {L} written in Latin script (Romanized).",
    "natural_cm": "The article may mix {L} with English words.",
}


def _head(lang, variant, article):
    L = LANGS[lang]["name"]
    note = VARIANT_NOTE.get(variant, "").format(L=L)
    return f"Language: {L}\n" + (f"Note: {note}\n" if note else "") + f"Article: {article}\n"


def cond_A(lang, variant, article, n_words, **_):
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    return (_head(lang, variant, article) +
            f"\nWrite a summary of the article in {L} ({S} script) in about {n_words} words. "
            f"Output only the summary.\nSummary:")


def _rules(lang):
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    return (f"Rules: (1) use only information stated in the article and add nothing else; "
            f"(2) keep names, numbers and dates exactly as the article gives them; "
            f"(3) do not change who did what, causes, or negations; "
            f"(4) the article may mix {L} and English or be in Latin script — understand it, but write "
            f"the summary in {L} using {S} script.")


def checklist(rec, max_items=12):
    a = rec["anchors"]
    ents = [e["text"] for e in a.get("entities", [])][:8]
    nums = a.get("numbers", [])[:6]
    dts = a.get("dates", [])[:3]
    parts = []
    if ents:
        parts.append("entities: " + "; ".join(ents))
    if nums:
        parts.append("numbers: " + "; ".join(nums))
    if dts:
        parts.append("dates: " + "; ".join(dts))
    return " | ".join(parts)[:600] or "(none detected)"


def cond_B1(lang, variant, article, n_words, rec=None, **_):
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    return (_head(lang, variant, article) + "\n" + _rules(lang) + "\n" +
            f"Source entities, numbers & dates (keep the most important ones; any quantity you derive "
            f"must follow from the article): {checklist(rec)}\n"
            f"Write a faithful summary in {L} ({S} script) in about {n_words} words. "
            f"Output only the summary.\nSummary:")


def cond_B2_step2(lang, variant, article, n_words, rec, draft, missing):
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    miss = "; ".join(missing) if missing else "(none)"
    return (_head(lang, variant, article) + "\n" + _rules(lang) + "\n" +
            f"Draft summary: {draft}\n"
            f"Important source anchors missing from the draft: {miss}\n"
            f"Rewrite the draft as a denser summary of about {n_words} words in {L} ({S} script) that "
            f"also covers the missing anchors where they matter, without adding anything not in the "
            f"article. Output only the summary.\nSummary:")


def cond_B3(lang, variant, article, n_words, rec=None, **_):
    L = LANGS[lang]["name"]
    ex = ("'increase ho raha hai' = 'बढ़ रहा है'" if lang == "hi" else "'increase thai rahyu chhe' = 'વધી રહ્યું છે'")
    hint = (f"Grammar hint: the matrix language is {L} (Subject–Object–Verb, postpositions"
            f"{', attached case endings -માં -થી -ને -નો/-ની/-નું/-ના' if lang == 'gu' else ''}). English words "
            f"inside {L} clauses are content words of the {L} clause ({ex}); the {L} verb carries tense "
            f"and negation — keep them as in the article.\n")
    base = cond_B1(lang, variant, article, n_words, rec)
    return base.replace("Write a faithful summary", hint + "Write a faithful summary")


CONDITIONS = {"A": cond_A, "B1": cond_B1, "B3": cond_B3}   # B2 = B1 draft + step 2


def bare_prior_prompt(lang, n_words):
    """Uncontaminated prior for CAD: no document, no anchors (spec §8.3 step 1)."""
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    return (f"Language: {L}\n\nWrite a summary of the article in {L} ({S} script) in about {n_words} "
            f"words. Output only the summary.\nSummary:")
