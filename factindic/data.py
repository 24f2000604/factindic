"""Datasets: XL-Sum, ILSUM (+ 5-rule sanitation), the controlled parallel testbed, dev split,
natural code-mixed subset, and the training pool for faithfulness weighting."""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .config import CFG, DATA_VERSION, DIRS, LANGS, VERSION_FILE, data_version, h1_pilot_ids
from .io import jl_read, jl_write, json_dump
from .lid import cmi, mixed_cmi, tag_word
from .normalizer import anchor_density, anchors
from .text import LATIN_RE, W, clean, strip_boilerplate, strip_punct, strip_web, truncate_words
from .translit import romanize, script_mix


# --------------------------------------------------------------------------- loaders
def load_xlsum(lang, split):
    """Parquet conversion branch first (works without dataset scripts), then main branch."""
    from datasets import Dataset, concatenate_datasets, load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    name = LANGS[lang]["xlsum"]
    split = "validation" if split in ("val", "valid", "dev") else split
    for rev in ("refs/convert/parquet", "main"):
        try:
            files = list_repo_files("csebuetnlp/xlsum", repo_type="dataset", revision=rev)
            shards = sorted(f for f in files if f.endswith(".parquet") and f.startswith(f"{name}/")
                            and (f"/{split}/" in f or f"-{split}" in f))
            if not shards:
                continue
            paths = [hf_hub_download("csebuetnlp/xlsum", f, repo_type="dataset", revision=rev)
                     for f in shards]
            return concatenate_datasets([Dataset.from_parquet(p) for p in paths])
        except Exception as e:
            print(f"xlsum {lang}/{split} @ {rev}: {repr(e)[:150]}")
    return load_dataset("csebuetnlp/xlsum", name, split=split)


def load_ilsum(lang, split, version="ILSUM/ILSUM-1.0"):
    """Returns list of dicts (id, article, summary) or [] if unavailable."""
    from datasets import load_dataset
    cfgname = LANGS[lang]["ilsum"]
    try:
        ds = load_dataset(version, cfgname, split=split)
    except Exception:
        try:
            from huggingface_hub import hf_hub_download
            fn = {"validation": "val"}.get(split, split)
            p = hf_hub_download(version, f"{cfgname}/{fn}.csv", repo_type="dataset")
            ds = load_dataset("csv", data_files=p, split="train")
        except Exception as e:
            print(f"ILSUM {lang}/{split} unavailable: {repr(e)[:150]}")
            return []
    cols = {c.lower(): c for c in ds.column_names}
    a, s = cols.get("article"), cols.get("summary")
    i = cols.get("id")
    return [dict(id=str(r[i]) if i else str(k), article=strip_web(r[a]), summary=strip_web(r[s]) if s else "")
            for k, r in enumerate(ds)]


# --------------------------------------------------------------------------- 5-rule sanitation
def prefix_copy(article, summary, max_offset=6):
    """Summary = the article's opening words, allowing a dateline or byline before it
    ("नई दिल्लीः …"). Compared word by word without punctuation; 90% of the words must match."""
    core = lambda t: [c for c in (strip_punct(w)[1] for w in t.split()) if c]
    a, s = core(article), core(summary)
    k = len(s)
    for o in range(max_offset + 1):
        head = a[o:o + k]
        if len(head) < k:
            return False
        if head == s or (k >= 8 and sum(x == y for x, y in zip(head, s)) / k >= 0.9):
            return True
    return False


def sanitize(rows, lang):
    """Rules: (1) empty / <5-word article or summary, (2) duplicate pairs, (3) duplicate summaries,
    (4) article-prefix copies (also after a dateline), (5) summary > 50% of article."""
    counts, kept = Counter(), []
    seen_pairs, seen_summ = set(), set()
    for r in rows:
        a, s = clean(r["article"], lang), clean(r["summary"], lang)
        wa, ws = a.split(), s.split()
        if len(wa) < 5 or len(ws) < 5:
            counts["empty_or_short"] += 1
            continue
        if (a, s) in seen_pairs:
            counts["duplicate_pair"] += 1
            continue
        seen_pairs.add((a, s))
        if s in seen_summ:
            counts["duplicate_summary"] += 1
            continue
        seen_summ.add(s)
        if prefix_copy(a, s):
            counts["article_prefix_copy"] += 1
            continue
        if len(ws) > 0.5 * len(wa):
            counts["summary_gt_50pct"] += 1
            continue
        kept.append(dict(id=r["id"], lang=lang, text=a, summary=s))
    report = dict(lang=lang, raw=len(rows), kept=len(kept), removed=dict(counts),
                  invalid_pct=round(100 * (len(rows) - len(kept)) / max(1, len(rows)), 2))
    return kept, report


# --------------------------------------------------------------------------- article records
def make_record(lang, source, split, doc_id, text, reference, tagger=None, tertile=None):
    D = truncate_words(strip_boilerplate(clean(text, lang)), CFG["max_article_words"])
    R = clean(reference, lang)
    ents = tagger.entities(D) if tagger else []
    mixed, emb = script_mix(D, lang, ents)
    variants = {"native": D, "script_mixed": mixed, "roman": romanize(D, lang)}
    ref_ents = tagger.entities(R) if tagger else []
    rec = dict(uid=f"{lang}:{source}:{doc_id}", lang=lang, source=source, split=split, doc_id=doc_id,
               text=D, reference=R, L_target=int(np.clip(W(R), 25, 45)), tertile=tertile,
               variants=variants, english_terms=emb,
               cmi={v: round(cmi(t, lang), 2) for v, t in variants.items()},   # Romanization must not change it
               anchors=anchors(D, lang, ents, emb), ref_anchors=anchors(R, lang, ref_ents))
    return rec


def _pool(lang, split, exclude):
    ds = load_xlsum(lang, split)
    rows = []
    for ex in ds:
        if ex["id"] in exclude:
            continue
        t, s = strip_boilerplate(clean(ex["text"], lang)), clean(ex["summary"], lang)
        if 150 <= W(t) <= 1500 and 10 <= W(s) <= 60:
            rows.append(dict(id=ex["id"], text=t, summary=s))
    return rows


def _stratified(rows, n, tagger, lang, rng, probe=600):
    """Verifier-independent stratified sampling over entity-density tertiles (seeded)."""
    cand = rng.sample(rows, min(probe, len(rows)))
    dens = []
    for r in cand:
        D = truncate_words(r["text"], CFG["max_article_words"])
        n_ent = len(tagger.entities(D)) if tagger else 0
        dens.append((n_ent + anchor_density(D)) / max(1, W(D)))
    dens = np.array(dens)
    cuts = np.quantile(dens, [1 / 3, 2 / 3])
    tert = np.digitize(dens, cuts)
    per = [n - 2 * (n // 3), n // 3, n // 3]
    out = []
    for t, k in enumerate(per):
        idx = [i for i in range(len(cand)) if tert[i] == t]
        out += [(cand[i], t) for i in rng.sample(idx, min(k, len(idx)))]
    return out


def upgrade():
    """Move files built by an older data version aside so they are rebuilt. The training pool is kept
    and repaired (build_train_pool), so its translations stay cached. From version 2 on, the testbed and
    dev articles are right: only their derived fields are computed again."""
    old = data_version()
    if old == DATA_VERSION:
        return
    if not any((DIRS["data"] / f).exists() for f in ("train_pool.jsonl", "testbed.jsonl", "natural_cm.jsonl")):
        VERSION_FILE.write_text(str(DATA_VERSION))       # fresh start: a retry must continue, not start over
        return
    redo = ["natural_cm.jsonl", "sanitation_report.json", "score_subset.json"]
    if old >= 2:
        for name in ("testbed", "dev"):
            path = DIRS["data"] / f"{name}.jsonl"
            if path.exists():
                jl_write(path, [_refresh(r) for r in jl_read(path)])
    else:
        redo += ["testbed.jsonl", "dev.jsonl", "weights.jsonl", "weights.shard0.jsonl", "weights.shard1.jsonl"]
    for f in redo:
        p = DIRS["data"] / f
        if p.exists():
            p.rename(p.with_name(f"old_v{old}_{f}"))
    pool = DIRS["data"] / "train_pool.jsonl"
    if pool.exists():
        pool.rename(pool.with_name("old_train_pool.jsonl"))


def _refresh(r):
    """Romanized variant, its CMI, and anchor keys of a stored record, without running NER again."""
    lang, names = r["lang"], lambda a: [e["text"] for e in a["entities"]]
    r["variants"]["roman"] = romanize(r["text"], lang)
    r["cmi"]["roman"] = round(cmi(r["variants"]["roman"], lang), 2)
    r["anchors"] = anchors(r["text"], lang, names(r["anchors"]), r["english_terms"])
    r["ref_anchors"] = anchors(r["reference"], lang, names(r["ref_anchors"]))
    return r


def stamp():
    for p in DIRS["data"].glob("old_*"):
        p.unlink()
    VERSION_FILE.write_text(str(DATA_VERSION))


def build_roman_vocab():
    """Roman form → most frequent native spelling, from the training pool (never the test articles)."""
    path = DIRS["data"] / "roman_vocab.json"
    if path.exists():
        return
    out = {}
    for lang in LANGS:
        lo, hi = LANGS[lang]["block"]
        native = Counter(w for r in jl_read(DIRS["data"] / "train_pool.jsonl") if r["lang"] == lang
                         for w in (strip_punct(t)[1] for t in r["text"].split())
                         if w and all(lo <= ord(c) <= hi for c in w))
        total, forms = sum(native.values()), {}
        for w, n in native.most_common():
            if n < 2:
                break
            f = forms.setdefault(romanize(w, lang), [w, 0])
            f[1] += n
        out[lang] = {k: [w, round(math.log10(n / total * 1e9), 2)] for k, (w, n) in forms.items() if k}
        print(f"roman_vocab/{lang}: {len(out[lang])} forms")
    json_dump(out, path)


def build_eval_sets(tagger):
    """testbed.jsonl (n_test_per_lang per language, XL-Sum test), dev.jsonl (50+50 XL-Sum validation)."""
    h1_ids = h1_pilot_ids()
    sets = {"testbed": ("test", CFG["n_test_per_lang"]), "dev": ("validation", CFG["n_dev_per_lang"])}
    for name, (split, n) in sets.items():
        path = DIRS["data"] / f"{name}.jsonl"
        if path.exists():
            continue
        rng = random.Random(f"{CFG['seed']}:{name}")
        recs = []
        for lang in LANGS:
            rows = _pool(lang, split, exclude=h1_ids)
            for r, t in _stratified(rows, n, tagger, lang, rng):
                recs.append(make_record(lang, "xlsum", split, r["id"], r["text"], r["summary"], tagger, int(t)))
            print(f"{name}/{lang}: {sum(x['lang'] == lang for x in recs)}")
        jl_write(path, recs)


def build_natural_cm(tagger):
    """Code-mixed ILSUM articles that pass the sanitation rules. Almost every Hindi article with CMI ≥ 8
    (the spec's rule) had page code in it, so whole English passages are not counted and the threshold
    steps down until there are enough articles. A random sample, since the top ranks are mostly job ads."""
    path = DIRS["data"] / "natural_cm.jsonl"
    if path.exists():
        return
    rng, steps = random.Random(CFG["seed"]), CFG["cmi_natural"]
    pool = jl_read(DIRS["data"] / "train_pool.jsonl")
    recs = []
    for lang in LANGS:
        used = {r["id"] for r in pool if r["lang"] == lang}
        seen = {r["text"][:200] for r in pool if r["lang"] == lang}
        scored = []
        for sp in ("test", "validation", "train"):
            rows = [r for r in load_ilsum(lang, sp) if r["id"] not in used]
            for r in sanitize(rows, lang)[0]:
                D = truncate_words(r["text"], CFG["max_article_words"])
                if W(D) < 100 or D[:200] in seen:
                    continue
                seen.add(D[:200])
                c = mixed_cmi(D, lang, CFG["cmi_max_run"])
                if c >= steps[-1]:
                    scored.append((c, r, sp))
        n = CFG["n_natural_cm_per_lang"]
        cut = next((s for s in steps if sum(x[0] >= s for x in scored) >= n), steps[-1])
        cand = [x for x in scored if x[0] >= cut]
        top = rng.sample(cand, min(n, len(cand)))
        for c, r, sp in top:
            rec = make_record(lang, "ilsum", sp, r["id"], r["text"], r["summary"], tagger)
            toks = rec["text"].split()
            rec["english_terms"] = sorted({strip_punct(t)[1].lower() for t in toks
                                           if LATIN_RE.search(t) and tag_word(t, lang) == "eng_Latn"})
            rec["anchors"]["english"] = rec["english_terms"]
            rec["variants"] = {"natural_cm": rec["text"]}       # already naturally mixed
            rec["cmi"] = {"natural_cm": round(c, 2)}
            recs.append(rec)
        cs = [c for c, _, _ in top]
        print(f"natural_cm/{lang}: {len(top)} of {len(cand)} with CMI≥{cut:g}; mean {np.mean(cs):.1f}, "
              f"{sum(c >= steps[0] for c in cs)} with CMI≥{steps[0]:g}"
              if cs else f"natural_cm/{lang}: none")
    jl_write(path, recs)


def build_train_pool():
    """Sanitized ILSUM-1.0 train (+ XL-Sum train to fill) → pool for weighting and SFT."""
    path = DIRS["data"] / "train_pool.jsonl"
    rep_path = DIRS["data"] / "sanitation_report.json"
    if path.exists():
        return
    rng = random.Random(CFG["seed"])
    old = jl_read(DIRS["data"] / "old_train_pool.jsonl")      # keep the valid pairs of an earlier pool
    seed = [] if old else pool_seed()
    pool, reports = [], []
    for lang in LANGS:
        raw = load_ilsum(lang, "train")
        clean_rows, rep = sanitize(raw, lang)
        reports.append(rep)
        print("ILSUM sanitation:", rep)
        n_target = CFG["n_weighted_per_lang"]
        had = {r["id"] for r in old or seed if r["lang"] == lang and r["source"] == "ilsum"}
        il = [r for r in clean_rows if r["id"] in had][: n_target // 2]
        rest = [r for r in clean_rows if r["id"] not in had]
        il += rng.sample(rest, min(n_target // 2 - len(il), len(rest)))
        for r in il:
            pool.append(dict(r, source="ilsum", text=truncate_words(r["text"], CFG["max_article_words"])))
        kept_xl = [r for r in old if r["lang"] == lang and r["source"] == "xlsum"]
        if kept_xl:
            pool += kept_xl
            continue
        xs = load_xlsum(lang, "train")
        idx = rng.sample(range(len(xs)), min(len(xs), 3 * n_target))
        want = {r["id"] for r in seed if r["lang"] == lang and r["source"] == "xlsum"}
        if want:
            idx = [j for j, i in enumerate(xs["id"]) if i in want] + idx
        xl, got = [], set()
        for j in idx:
            ex = xs[j]
            if ex["id"] in got:
                continue
            got.add(ex["id"])
            t, s = strip_boilerplate(clean(ex["text"], lang)), clean(ex["summary"], lang)
            if W(t) >= 100 and 10 <= W(s) <= 60:
                xl.append(dict(id=ex["id"], lang=lang, text=t, summary=s))
            if len(xl) >= n_target - len(il):
                break
        for r in xl:
            pool.append(dict(r, source="xlsum", text=truncate_words(r["text"], CFG["max_article_words"])))
    if old:
        _drop_stale_scores(old, pool)
    else:
        _seed_scores(seed, pool)
    jl_write(path, pool)
    json_dump(reports, rep_path)


def pool_seed():
    """Ids of the training pool of the finished notebook-01 run and its faithfulness scores (pool_seed.txt),
    so the pool is rebuilt identically and unchanged pairs are not scored again."""
    path = Path(__file__).with_name("pool_seed.txt")
    if not path.exists():
        return []
    cols = ("lang", "source", "id", "sig", "s_mc")
    return [dict(zip(cols, line.split("\t"))) for line in path.read_text(encoding="utf-8").splitlines()]


def _signature(r):
    return hashlib.md5("".join((r["text"] + r["summary"]).split()).encode()).hexdigest()[:8]


def _seed_scores(seed, pool):
    path = DIRS["data"] / "weights.jsonl"
    if path.exists() or list(DIRS["data"].glob("weights.shard*.jsonl")):
        return
    known = {(r["lang"], r["id"]): r for r in seed if r["s_mc"]}
    rows = []
    for r in pool:
        k = known.get((r["lang"], r["id"]))
        if k and k["sig"] == _signature(r):
            s = float(k["s_mc"])
            rows.append(dict(key=f"{r['lang']}:{r['id']}", lang=r["lang"], source=r["source"], s_mc=s,
                             w=float(np.clip(s, CFG["w_min"], 1.0)), keep=s >= CFG["w_min"], mt="it2"))
    jl_write(path, rows)
    print(f"{len(rows)} faithfulness scores reused from the finished run")


def _drop_stale_scores(old, pool):
    """Remove faithfulness scores of pairs whose text or summary is no longer the same."""
    sig = lambda r: (f"{r['lang']}:{r['id']}", r["text"], r["summary"])
    same = {s[0] for s in {*map(sig, old)} & {*map(sig, pool)}}
    for p in DIRS["data"].glob("weights*.jsonl"):
        rows = jl_read(p)
        keep = [w for w in rows if w["key"] in same]
        if len(keep) < len(rows):
            jl_write(p, keep)


def load_set(name):
    """The articles in use: the first n_use per language, taken evenly from the tertiles."""
    rows = jl_read(DIRS["data"] / f"{name}.jsonl")
    n = (CFG.get("n_use") or {}).get(name)
    if not n:
        return rows
    out = []
    for lang in LANGS:
        groups = {}
        for r in rows:
            if r["lang"] == lang:
                groups.setdefault(r["tertile"], []).append(r)
        longest = max(map(len, groups.values()), default=0)
        order = [g[i] for i in range(longest) for g in groups.values() if i < len(g)]
        out += order[:n]
    return out


def items(records, variants=None):
    """Expand article records into (item_id, record, variant) generation inputs."""
    out = []
    for r in records:
        for v, _ in r["variants"].items():
            if variants is None or v in variants:
                out.append((f"{r['uid']}|{v}", r, v))
    return out
