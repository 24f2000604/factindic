"""The FactIndic pipeline for one document, on one 8 GB GPU.

Stage 1 normalise + anchors → Stage 2 B1 prompt → Stage 3 Anchor-CAD+DoLa candidate + 4 nucleus samples →
Stage 4 round-trip router (IndicTrans2 → MiniCheck, else native mDeBERTa), J reranking, one prune pass.
Same functions, prompts and tuned settings as the experiments. Not in the experiments: English input, texts over
350 words, and anchors of Romanized input from back-transliterated text (the testbed had the native article)."""
from __future__ import annotations

import gc
import threading
import time
import unicodedata

from factindic.config import CFG, DIRS, LANGS, MODELS, safe_attention
from factindic.io import json_load

from .language import resolve

LANGS["en"] = dict(name="English", script="Latin", it2=None, block=(0x41, 0x7A))     # this process only
EVALUATED_WORDS, MAX_WORDS, MIN_WORDS = 350, 1000, 20
DEFAULT_PARAMS = dict(alpha_base=0.2, alpha_anchor=0.6, beta=0.2)        # dev-set choice of the final run


class InputError(ValueError):
    pass


class Translation:
    """IndicTrans2 without the on-disk cache (users' documents are not stored); English passes through."""

    def __init__(self):
        from factindic.mt import Translator

        class Memory:
            def add(self, row): pass
            def close(self): pass

        self.t = Translator.__new__(Translator)
        self.t.backend, self.t._engine, self.t.cache, self.t.trimmed, self.t.cache_file = "it2", None, {}, set(), Memory()
        self.backend = "it2"

    def translate(self, sents, lang):
        return list(sents) if lang == "en" else self.t.translate(sents, lang)

    def warm(self):
        self.t._run(["नमस्ते।"], "hi")

    def forget(self):
        self.t.cache.clear()

    def release(self):
        """Stop the helper process: its GPU memory grows with long sentences. The next translation starts it again (~8 s)."""
        if self.t._engine is not None:
            self.t._engine.close()
            self.t._engine = None


class OnDemand:
    """A verifier that lives in RAM and visits the GPU only to score. If there is too little room, the translator leaves first."""

    def __init__(self, scorer):
        self.s, self.name, self.make_room = scorer, scorer.name, None
        self.s.model.to("cpu")
        # one fixed copy in RAM: moving back and forth with .to() would allocate a new one every time
        self.home = [(t, t.data) for t in (*self.s.model.parameters(), *self.s.model.buffers())]
        self.bytes = sum(d.numel() * d.element_size() for _, d in self.home)

    def _return(self):
        for t, d in self.home:
            t.data = d

    def score_pairs(self, pairs):
        import torch
        torch.cuda.empty_cache()
        if self.make_room and torch.cuda.mem_get_info()[0] < self.bytes + 500_000_000:
            self.make_room()
        try:
            for t, d in self.home:               # inside try: a half-moved model must come back too
                t.data = d.to("cuda")
            return self.s.score_pairs(pairs)
        finally:
            self._return()
            torch.cuda.empty_cache()


def trim():
    """Give freed heap memory back to the system (glibc keeps it otherwise)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Engine:
    def __init__(self):
        self.lock, self.ready, self.error, self.stage = threading.Lock(), False, None, "starting"

    def load(self):
        try:
            self._load()
            self.ready, self.stage = True, "ready"
        except Exception as e:
            self.error = f"{type(e).__name__}: {str(e)[:500]}"
            self.stage = "failed"
            raise

    def _load(self):
        import torch
        from factindic.llm import load_causal
        from factindic.ner import Tagger
        from factindic.verify import MiniCheck, NativeNLI, Router
        safe_attention()
        self.stage = "loading the verifiers"
        mc, nli = OnDemand(MiniCheck()), OnDemand(NativeNLI())
        self.stage = "loading the entity taggers"
        self.ner = Tagger(device=-1)
        trim(); torch.cuda.empty_cache()
        self.stage = "loading Gemma-4"
        self.model, self.tok, self.precision = load_causal(MODELS["gemma"])
        self.stage = "starting IndicTrans2"
        self.mt = Translation()
        self.mt.warm()
        trim()
        mc.make_room = nli.make_room = self.mt.release
        self.router = Router(self.mt, mc, nli)
        p = json_load(DIRS["outputs"] / "stage3_params.json", {}) or {}
        self.params = {k: p.get(k, v) for k, v in DEFAULT_PARAMS.items()}

    # ---------------------------------------------------------------- stage 1
    def record(self, text, kind, n_words):
        from factindic.lid import tag_tokens
        from factindic.normalizer import anchors
        from factindic.text import W, clean, truncate_words
        from factindic.translit import back_transliterate
        info = resolve(text, kind)
        lang, variant = info["lang"], info["variant"]
        text = clean(text, lang) if lang != "en" else " ".join(unicodedata.normalize("NFC", text).split())
        total = W(text)
        if total < MIN_WORDS:
            raise InputError(f"The document has {total} words; at least {MIN_WORDS} are needed.")
        notes = []
        if total > MAX_WORDS:
            text = truncate_words(text, MAX_WORDS)
            notes.append(f"The document has {total} words. Only the first {MAX_WORDS} were read; the rest is not in the summary.")
        if total > EVALUATED_WORDS:
            notes.append(f"The model was evaluated on articles cut to {EVALUATED_WORDS} words; longer input is untested.")
        native, english = text, []
        if lang != "en" and variant != "native":
            tags = tag_tokens(text, lang)
            english = sorted({w.strip(".,!?;:'\"()").lower() for w, t in zip(text.split(), tags) if t == "eng_Latn"} - {""})
            if variant == "roman":
                native = back_transliterate(text, lang, tags)
                notes.append("Romanized input is the model's weakest case (claim precision 0.68 for Hindi and 0.40 for Gujarati, "
                             "against 0.88 and 0.78 in native script). Check names and numbers.")
        if lang == "en":
            notes.append("English input was not part of the experiments; its quality has not been measured.")
        ents = self.ner.entities(native)
        rec = dict(uid="app", lang=lang, text=native, reference="", L_target=int(n_words), variants={variant: text},
                   english_terms=english, anchors=anchors(native, lang, ents, english))
        return rec, dict(info, words=total, words_read=min(total, MAX_WORDS)), notes

    # ---------------------------------------------------------------- stages 2-4
    def summarize(self, text, kind=None, mode="full", n_words=40):
        import torch
        with self.lock:
            try:
                try:
                    return self._summarize(text, kind, mode, n_words)
                except torch.cuda.OutOfMemoryError:
                    pass                         # retried outside the handler, whose traceback would keep the tensors alive
                self._clean()
                self.mt.release()
                return self._summarize(text, kind, mode, n_words)
            finally:
                self._clean()
                if torch.cuda.mem_get_info()[0] < 2_000_000_000:      # leave room for the next request
                    self.mt.release()

    def _clean(self):
        import torch
        self.router.src_cache.clear(); self.router.score_cache.clear(); self.mt.forget()
        trim()
        torch.cuda.empty_cache()

    def _summarize(self, text, kind, mode, n_words):
        from factindic.claims import sentence_claims
        from factindic.decoding import anchor_token_ids, contrastive_generate
        from factindic.llm import chat_text, generate
        from factindic.pipeline.stage2 import prompt_for
        from factindic.prompts import bare_prior_prompt
        from factindic.rerank import J_score, prune
        import torch
        t, clock = {}, time.perf_counter
        t0 = clock()
        rec, info, notes = self.record(text, kind, n_words)
        variant = info["variant"]
        t["anchors"] = clock() - t0
        prompt = chat_text(self.tok, prompt_for("B1", rec, variant))
        cap = CFG["gen"]["max_new_tokens"] + 4 * max(0, n_words - 45)
        t1 = clock()
        if mode == "fast":                                       # Stage 2 only
            s = generate(self.model, self.tok, [prompt], max_new_tokens=cap)[0][0][0]
            t["generate"] = clock() - t1
            return self._result(s, rec, info, notes, t, t0, mode)
        bare = chat_text(self.tok, bare_prior_prompt(rec["lang"], rec["L_target"]))
        cad, _ = contrastive_generate(self.model, self.tok, [prompt], [bare], [anchor_token_ids(self.tok, rec)], "anchor_cad_dola",
                                      eta=CFG["cad"]["eta"], max_new_tokens=cap, **self.params)
        kw = dict(max_new_tokens=cap, do_sample=True, temperature=CFG["sample_T"], top_p=CFG["sample_top_p"])
        try:
            samples = generate(self.model, self.tok, [prompt], num_return_sequences=CFG["K"] - 1, seed=CFG["seed"], **kw)[0][0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            samples = [generate(self.model, self.tok, [prompt], seed=CFG["seed"] + i, **kw)[0][0][0] for i in range(CFG["K"] - 1)]
        torch.cuda.empty_cache()
        t["generate"] = clock() - t1
        t2 = clock()
        cands = [dict(k=0, source="anchor-CAD + DoLa", summary=cad[0])]
        cands += [dict(k=i + 1, source="sample", summary=s) for i, s in enumerate(samples)]
        cands = [c for c in cands if c["summary"].strip()]
        if not cands:
            raise RuntimeError("the model wrote an empty summary")
        for c in cands:
            c["claims"] = sentence_claims(c["summary"])
        every = list(dict.fromkeys(x for c in cands for x in c["claims"]))     # each claim is scored on its own, so once is enough
        by_claim = {v["claim"]: v for v in self.router.verify(rec, variant, every)} if every else {}
        tau, lam = CFG["tau_ver"], CFG["lambda_len"]
        for c in cands:
            c["verdicts"] = [by_claim[x] for x in c["claims"]]
            c.update(J_score(c["summary"], c["verdicts"], rec, tau, lam, self.ner))
        best = max(cands, key=lambda c: (c["J"], -c["k"]))
        final, pruned = prune(best["verdicts"], tau) if best["verdicts"] else (best["summary"], False)
        t["verify"] = clock() - t2
        out = self._result(final, rec, info, notes, t, t0, mode)
        kept = set(sentence_claims(final))
        out["sentences"] = [dict(text=v["claim"], english=v["claim_en"], supported=v["p"] >= tau, score=round(v["p"], 3),
                                 checked_by="MiniCheck (translated)" if v["route"] == "minicheck" else "mDeBERTa (native)",
                                 removed=v["claim"] not in kept) for v in best["verdicts"]]
        out.update(pruned=pruned, chosen=best["k"], candidates=[
            dict(k=c["k"], source=c["source"], summary=c["summary"], J=round(c["J"], 3), supported=round(c["frac_supported"], 3),
                 anchor_recall=round(c["R_anchor"], 3)) for c in cands])
        if any(not s["supported"] and not s["removed"] for s in out["sentences"]):
            notes.append("Some sentences could not be verified against the document; they are marked below.")
        return out

    def _result(self, summary, rec, info, notes, t, t0, mode):
        from factindic.text import W
        a = rec["anchors"]
        return dict(summary=summary, words=W(summary), mode=mode, language=info["kind"], variant=info["variant"],
                    output_language=LANGS[rec["lang"]]["name"], document_words=info["words"], words_read=info["words_read"],
                    anchors=dict(entities=list(dict.fromkeys(e["text"] for e in a["entities"])), numbers=a["numbers"], dates=a["dates"]),
                    notes=notes, sentences=[], candidates=[], pruned=False, chosen=None,
                    seconds=dict({k: round(v, 2) for k, v in t.items()}, total=round(time.perf_counter() - t0, 2)))


ENGINE = Engine()
