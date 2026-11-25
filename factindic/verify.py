"""Grounded verifiers and the Stage-4 Round-Trip Anchor Verification Router (spec §8.4).

MiniCheck-FT5   English, used inside Stage 4 (router route 1) and for mFACT weights.
mDeBERTa-XNLI   native multilingual NLI, router route 2.
FactCG-DeBERTa  English, HELD OUT: only for final evaluation (never weights or reranking).
"""
from __future__ import annotations

import torch

from .config import IS_LOCAL, MODELS
from .normalizer import ent_key, numbers, _close
from .text import chunk_sentences, split_sentences


def _split_on_oom(f):
    """score_pairs with the batch size halved until it fits."""
    def run(self, pairs, bs=None):
        bs = bs or (4 if IS_LOCAL else 8)
        while True:
            try:
                return f(self, pairs, bs)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
            torch.cuda.empty_cache()
            bs //= 2
    return run


class MiniCheck:
    """First decoder step over token 3 ('▁', which begins "0") vs 209 ('1'); sanity-checked."""
    name = "minicheck"

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODELS["minicheck"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODELS["minicheck"], dtype=torch.bfloat16 if IS_LOCAL else torch.float32).cuda().eval()
        self.ids = [3, 209]
        hi = self.score_pairs([("The district hospital in Surat admitted 42 patients on Monday.",
                                "42 patients were admitted to a hospital in Surat.")])[0]
        lo = self.score_pairs([("The district hospital in Surat admitted 42 patients on Monday.",
                                "The gas leak killed 300 people in Delhi.")])[0]
        assert hi > 0.5 > lo, f"MiniCheck sanity failed ({hi:.2f}, {lo:.2f})"

    @_split_on_oom
    @torch.no_grad()
    def score_pairs(self, pairs, bs):
        out = []
        for i in range(0, len(pairs), bs):
            texts = [f"premise: {d} hypothesis: {c}" for d, c in pairs[i:i + bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            dec = torch.full((enc["input_ids"].shape[0], 1), self.model.config.decoder_start_token_id, device="cuda")
            lg = self.model(**enc, decoder_input_ids=dec).logits[:, 0, self.ids]
            out += torch.softmax(lg.float(), -1)[:, 1].tolist()
        return out


class NativeNLI:
    name = "mdeberta"

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODELS["mdeberta"])
        self.model = AutoModelForSequenceClassification.from_pretrained(MODELS["mdeberta"], dtype=torch.float32).cuda().eval()
        self.ent = [i for i, l in self.model.config.id2label.items() if l.lower().startswith("entail")][0]
        src = "सूरत के ज़िला अस्पताल में सोमवार को 42 मरीज़ भर्ती हुए।"
        hi, lo = self.score_pairs([(src, "सूरत के अस्पताल में 42 मरीज़ भर्ती हुए।"), (src, "दिल्ली में गैस रिसाव से 300 लोगों की मौत हुई।")])
        assert hi > 0.5 > lo, f"mDeBERTa sanity failed ({hi:.2f}, {lo:.2f})"

    @_split_on_oom
    @torch.no_grad()
    def score_pairs(self, pairs, bs):
        out = []
        for i in range(0, len(pairs), bs):
            p = pairs[i:i + bs]
            enc = self.tok([d for d, _ in p], [c for _, c in p], return_tensors="pt", padding=True,
                           truncation="only_first", max_length=512).to("cuda")
            out += torch.softmax(self.model(**enc).logits.float(), -1)[:, self.ent].tolist()
        return out


FACTCG_TEMPLATE = ("{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "
                   "\"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is ")


class FactCG:
    """Template, label index 1 = supported, and max over sentence-packed chunks follow
    github.com/derenlei/FactCG (factcg/inference.py)."""
    name = "factcg"

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODELS["factcg"], use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODELS["factcg"], num_labels=2, dtype=torch.float32).cuda().eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        src = "The district hospital in Surat admitted 42 patients on Monday."
        hi, lo = self.score_pairs([(src, "42 patients were admitted to a hospital in Surat."),
                                   (src, "The gas leak killed 300 people in Delhi.")])
        assert hi > 0.5 > lo, f"FactCG sanity failed ({hi:.2f}, {lo:.2f})"

    @_split_on_oom
    @torch.no_grad()
    def score_pairs(self, pairs, bs):
        out = []
        for i in range(0, len(pairs), bs):
            texts = [FACTCG_TEMPLATE.format(text_a=d, text_b=c) for d, c in pairs[i:i + bs]]
            enc = self.tok(texts, return_tensors="pt", padding="longest", truncation=True, max_length=2048).to("cuda")
            out += torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].tolist()
        return out


def max_over_chunks(scorer, chunks, claims, cache=None):
    """p(D, c) = max_j P(Supported | D_j, c)."""
    pairs, owners = [], []
    res = [0.0] * len(claims)
    for ci, c in enumerate(claims):
        if not c:
            continue
        for ch in chunks:
            key = (scorer.name, ch, c)
            if cache is not None and key in cache:
                res[ci] = max(res[ci], cache[key])
            else:
                pairs.append((ch, c)); owners.append((ci, key))
    if pairs:
        for (ci, key), p in zip(owners, scorer.score_pairs(pairs)):
            if cache is not None:
                cache[key] = p
            res[ci] = max(res[ci], p)
    return res


def source_for_variant(rec, variant, mode="deployment"):
    """Text the verifier treats as D. Deployment: the input variant (Romanized → back-transliterated
    to native). Oracle: the aligned native article (analysis only, H3 proxy)."""
    from .translit import back_transliterate
    if mode == "oracle" or variant in ("native", "natural_cm"):
        return rec["variants"].get("native", rec["text"])
    if variant == "roman":
        return back_transliterate(rec["variants"]["roman"], rec["lang"])
    return rec["variants"][variant]


class Router:
    """Route 1 (translate-then-verify, MiniCheck) when every number and entity key of the native
    claim survives MT; otherwise Route 2 (native mDeBERTa against native-script source chunks)."""

    def __init__(self, translator, minicheck, nli):
        self.mt, self.mc, self.nli = translator, minicheck, nli
        self.src_cache, self.score_cache = {}, {}

    def prepare_source(self, rec, variant, mode="deployment"):
        key = (rec["uid"], variant, mode)
        if key not in self.src_cache:
            D = source_for_variant(rec, variant, mode)
            sents = split_sentences(D)
            en = self.mt.translate(sents, rec["lang"])
            self.src_cache[key] = dict(native_chunks=chunk_sentences(sents, 180),
                                       en_chunks=chunk_sentences(en, 250), text=D)
        return self.src_cache[key]

    @staticmethod
    def survived(claim, claim_en, rec):
        if not numbers(claim) <= numbers(claim_en):
            return False
        src_keys = {k for e in rec["anchors"].get("entities", []) for k in e["key"].split()}
        claim_ent = {ent_key(w, rec["lang"]) for w in claim.split()}
        claim_ent = {k for k in claim_ent if k and _close(k, src_keys)}
        en_keys = {ent_key(w, "en") for w in claim_en.split()} - {""}
        return all(_close(k, en_keys) for k in claim_ent)

    def verify(self, rec, variant, claims, mode="deployment", routes="router"):
        src = self.prepare_source(rec, variant, mode)
        claims_en = self.mt.translate(claims, rec["lang"]) if claims else []
        p_mc = max_over_chunks(self.mc, src["en_chunks"], claims_en, self.score_cache)
        p_nli = max_over_chunks(self.nli, src["native_chunks"], claims, self.score_cache)
        out = []
        for c, ce, a, b in zip(claims, claims_en, p_mc, p_nli):
            ok = self.survived(c, ce, rec)
            out.append(dict(claim=c, claim_en=ce, p_minicheck=a, p_nli=b, survived=ok,
                            route="minicheck" if ok else "nli", p=a if ok else b))
        return out
