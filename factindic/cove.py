"""Chain-of-Verification (factored) self-rewrite baseline (Exp 5b) — the 'self-verification
fallacy' comparator: the same 2B generator plans, answers, and rewrites its own summary."""
from __future__ import annotations

import re

from .config import LANGS
from .llm import chat_text, generate


def _plan(lang, summary):
    return (f"Here is a {LANGS[lang]['name']} news summary:\n{summary}\n\n"
            "Write up to 4 short verification questions (in English) that check its facts: names, "
            "numbers, dates, and who did what. One question per line, no numbering.")


def _answer(lang, variant_text, q):
    return (f"Article:\n{variant_text}\n\nAnswer the question using only the article, in one short "
            f"sentence. If the article does not say, answer 'not stated'.\nQuestion: {q}\nAnswer:")


def _revise(lang, summary, qa, n_words):
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    qa_txt = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa)
    return (f"Original summary ({L}):\n{summary}\n\nVerification results:\n{qa_txt}\n\n"
            f"Rewrite the summary in {L} ({S} script), about {n_words} words, correcting anything the "
            f"verification contradicts and removing anything marked 'not stated'. Output only the summary.")


def cove_batch(model, tok, recs_variants, drafts, max_new_tokens=160):
    """recs_variants: list of (rec, variant). Returns revised summaries and the Q/A traces."""
    plans, _, _ = generate(model, tok, [chat_text(tok, _plan(r["lang"], d)) for (r, _), d in zip(recs_variants, drafts)],
                           max_new_tokens=120)
    qs = []
    for p in plans:
        lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in p[0].split("?")]
        qs.append([l + "?" for l in lines if len(l) > 5][:4])
    flat = [(i, q) for i, ql in enumerate(qs) for q in ql]
    answers = {}
    for k in range(0, len(flat), 8):
        batch = flat[k:k + 8]
        texts = [chat_text(tok, _answer(recs_variants[i][0]["lang"], recs_variants[i][0]["variants"][recs_variants[i][1]], q))
                 for i, q in batch]
        out, _, _ = generate(model, tok, texts, max_new_tokens=48)
        for (i, q), o in zip(batch, out):
            answers.setdefault(i, []).append((q, o[0]))
    rev_prompts = [chat_text(tok, _revise(r["lang"], d, answers.get(i, []), r["L_target"]))
                   for i, ((r, _), d) in enumerate(zip(recs_variants, drafts))]
    revised, _, _ = generate(model, tok, rev_prompts, max_new_tokens=max_new_tokens)
    return [x[0] for x in revised], [answers.get(i, []) for i in range(len(drafts))]
