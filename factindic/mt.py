"""Indic→English projection with IndicTrans2-dist-200M, which runs as a helper process in its own
environment (it2_env.py). Native digits are mapped to ASCII before translation so numbers are copied
rather than re-generated; all translations are cached on disk.

There is no silent fallback: mixing translators inside one experiment would confound the comparison of verifiers. To run the
whole project on NLLB-200 instead, set CFG["mt_backend"]="nllb" or the env var FACTINDIC_MT=nllb.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import subprocess

from .config import CFG, DIRS, LANGS, MODELS
from .io import JsonlAppender, chunked, jl_read_glob, md5
from .normalizer import DIGITS
from .text import INITIALS

_INIT = "|".join(sorted(INITIALS, key=len, reverse=True))
_CHAIN = re.compile(rf"(?<![\u0900-\u0AFF])(?:(?:{_INIT})\.)+(?:{_INIT})(?![\u0900-\u0AFF])\.?")
_GLUE = re.compile(r"([\u0900-\u0AFF][.।!?])(?=[^\s.,।!?\"'”’)\]])")
_LOOP = re.compile(r"(?:^|\s)(\S(?:.{0,40}?\S)?)(?:\s+\1){3,}(?=\s|$)")


def prepare(s):
    """What the translator sees: ASCII digits; initials without dots and a space after every full stop,
    because IndicTrans2's preprocessor reads 'છે.પાણી' and 'એલ.સી.બી' as web addresses and copies them."""
    s = s.translate(DIGITS)
    s = _CHAIN.sub(lambda m: m.group(0).replace(".", ""), s)
    return _GLUE.sub(r"\1 ", s)


_CHARS = re.compile(r"([^\w\s])\1{7,}")


def runaway(en):
    """Decoder loop: the same phrase four or more times in a row ('"" "" "" ""') or a run of one mark."""
    return _LOOP.search(en) is not None or _CHARS.search(en) is not None


def trim(en):
    """The translation before the loop started."""
    en = _CHARS.sub(r"\1", en)
    m = _LOOP.search(en)
    if m:
        en = en[:m.start()].strip() or m.group(1)
    return en


def lost(en):
    """A loop that left almost no translation before it: worth a second attempt."""
    return runaway(en) and len(trim(en).split()) < 4


class _IT2:
    """Client side of it2_server.py."""

    def __init__(self):
        from . import it2_env
        py = it2_env.ensure()
        DIRS["logs"].mkdir(parents=True, exist_ok=True)
        self.log = DIRS["logs"] / f"it2_{os.getpid()}.log"
        with open(self.log, "w") as err:
            self.p = subprocess.Popen([str(py), "-u", str(it2_env.SERVER), MODELS["it2"]], env=it2_env.clean_env(),
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                      text=True, encoding="utf-8", bufsize=1)
        atexit.register(self.close)
        self._read()

    def _read(self):
        line = self.p.stdout.readline()
        msg = json.loads(line) if line.strip() else {"fatal": "the process exited"}
        if "fatal" in msg or "error" in msg:
            tail = "\n".join(self.log.read_text(errors="ignore").splitlines()[-15:])
            hint = ""
            if any(s in (tail + str(msg)).lower() for s in ("gated", "401", "403")):
                hint = f"\n→ Accept the terms at https://huggingface.co/{MODELS['it2']} with the account that owns HF_TOKEN."
            if "fatal" in msg:
                self.close()
            raise RuntimeError(f"IndicTrans2: {msg.get('fatal') or msg.get('error')}{hint}\n--- {self.log} ---\n{tail}")
        return msg

    def run(self, sents, lang, retry=False):
        self.p.stdin.write(json.dumps(dict(src=LANGS[lang]["it2"], sents=sents, retry=retry), ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        out = self._read()["out"]
        assert len(out) == len(sents), "IndicTrans2 returned a different number of sentences"
        return out

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=20)
            except Exception:
                self.p.kill()


class _NLLB:
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODELS["nllb"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODELS["nllb"], dtype=torch.float16).cuda().eval()

    def run(self, sents, lang, retry=False):
        import torch
        self.tok.src_lang = LANGS[lang]["it2"]
        enc = self.tok(sents, truncation=True, padding="longest", return_tensors="pt", max_length=256).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**enc, num_beams=4, max_length=256, **(dict(no_repeat_ngram_size=4) if retry else {}),
                                      forced_bos_token_id=self.tok.convert_tokens_to_ids("eng_Latn"))
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def close(self):
        self.model = None


class Translator:
    def __init__(self, backend=None, cache_name="mt_cache.jsonl"):
        self.backend = backend or os.environ.get("FACTINDIC_MT") or CFG["mt_backend"]
        if self.backend not in ("it2", "nllb"):
            raise ValueError(f"unknown MT backend {self.backend!r}")
        self._engine = None                          # started on the first uncached sentence
        self.cache_file = JsonlAppender(DIRS["cache"] / cache_name, key_fn=lambda r: r["k"])
        self.cache, self.trimmed = {}, set()
        for r in jl_read_glob(DIRS["cache"] / cache_name):
            if r.get("b") == self.backend and (r.get("r") or not lost(r["en"])):
                self.cache[r["k"]] = trim(r["en"])
                if self.cache[r["k"]] != r["en"]:
                    self.trimmed.add(r["k"])

    def _key(self, lang, s):
        return md5(self.backend, lang, s)

    def has(self, sents, lang):
        """True if every sentence has a cached translation that needed no repair."""
        ks = [self._key(lang, p) for p in map(prepare, sents) if p.strip()]
        return all(k in self.cache and k not in self.trimmed for k in ks)

    def _run(self, sents, lang, retry=False):
        if self._engine is None:
            self._engine = _IT2() if self.backend == "it2" else _NLLB()
        return self._engine.run(sents, lang, retry)

    def translate(self, sents, lang, bs=32):
        sents = [prepare(s) for s in sents]
        todo = sorted({s for s in sents if s.strip() and self._key(lang, s) not in self.cache}, key=len)
        for batch in chunked(todo, bs):
            res = list(zip(batch, self._run(batch, lang), [False] * len(batch)))
            loops = [s for s, e, _ in res if lost(e)]
            if loops:
                again = dict(zip(loops, self._run(loops, lang, retry=True)))
                res = [(s, again[s], True) if s in again else (s, e, r) for s, e, r in res]
            for s, e, r in res:
                e = trim(e)
                k = self._key(lang, s)
                self.cache[k] = e
                self.cache_file.add(dict(k=k, en=e, b=self.backend, **(dict(r=1) if r else {})))
        return [self.cache.get(self._key(lang, s), "") if s.strip() else "" for s in sents]

    def close(self):
        self.cache_file.close()
        if self._engine is not None:
            self._engine.close()
            self._engine = None
