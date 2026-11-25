"""Human evaluation: claim-level annotation sheets (300 summaries, 2 annotators per language),
variant-naturalness sheets (50 items per language), import + Cohen's κ.

Labels: 1 = Supported, 0 = Unsupported, -1 = Contradicted (scored as 0).
Error types (ILSUM 2024 Task 2): Misrepresentation, Inaccurate Quantities, False Attribution, Fabrication.
"""
from __future__ import annotations

import random

import pandas as pd

from .claims import sentence_claims
from .config import CFG, DIRS
from .stats import cohen_kappa

ERROR_TYPES = ["", "Misrepresentation", "Inaccurate Quantities", "False Attribution", "Fabrication"]


def export_claim_sheets(summaries, records, seed=42):
    """summaries: list of dicts (system, item, uid, lang, variant, summary). Systems are blinded."""
    rng = random.Random(seed)
    by_lang = {"hi": [], "gu": []}
    for s in summaries:
        if s.get("summary"):
            by_lang[s["lang"]].append(s)
    n_lang = CFG["n_annot_summaries"] // 2
    rows, key = [], []
    for lang, pool in by_lang.items():
        # stratify across (variant, system) cells
        cells = {}
        for s in pool:
            cells.setdefault((s["variant"], s["system"]), []).append(s)
        picked, cell_keys = [], sorted(cells)
        while len(picked) < n_lang and any(cells.values()):
            for ck in cell_keys:
                if cells[ck] and len(picked) < n_lang:
                    picked.append(cells[ck].pop(rng.randrange(len(cells[ck]))))
        rng.shuffle(picked)
        n_dev = CFG["n_annot_dev"] // 2
        for i, s in enumerate(picked):
            sid = f"{lang}-{i:03d}"
            split = "dev" if i < n_dev else "test"
            key.append(dict(sid=sid, split=split, system=s["system"], item=s["item"], uid=s["uid"],
                            lang=lang, variant=s["variant"]))
            src = records[s["uid"]]["variants"].get(s["variant"], records[s["uid"]]["text"])
            for ci, c in enumerate(sentence_claims(s["summary"])):
                rows.append(dict(sid=sid, claim_idx=ci, source=src if ci == 0 else "", summary=s["summary"] if ci == 0 else "",
                                 claim=c, label="", error_type="", note=""))
    df, kdf = pd.DataFrame(rows), pd.DataFrame(key)
    for lang in ("hi", "gu"):
        part = df[df.sid.str.startswith(lang)]
        for ann in ("A", "B"):
            part.to_csv(DIRS["annotation"] / f"claims_{lang}_annotator{ann}.csv", index=False, encoding="utf-8-sig")
    kdf.to_csv(DIRS["annotation"] / "claims_key_DO_NOT_SHARE.csv", index=False)
    return len(kdf), len(df)


def export_naturalness(records, seed=42):
    rng = random.Random(seed)
    rows = []
    for lang in ("hi", "gu"):
        pool = [r for r in records if r["lang"] == lang]
        for r in rng.sample(pool, min(CFG["n_naturalness_per_lang"], len(pool))):
            for v in ("script_mixed", "roman"):
                rows.append(dict(uid=r["uid"], lang=lang, variant=v, native=r["text"][:600],
                                 variant_text=r["variants"][v][:600], naturalness_1to5="",
                                 meaning_preserved_yes_no=""))
    df = pd.DataFrame(rows)
    for lang in ("hi", "gu"):
        path = DIRS["annotation"] / f"naturalness_{lang}.csv"
        if path.exists() and pd.read_csv(path).naturalness_1to5.notna().any():
            print(f"kept {path.name}: it already has ratings")
            continue
        df[df.lang == lang].to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


def _lab(x):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return None
    return 1 if v == 1 else 0


def import_claim_labels():
    """Returns (merged claim table with gold = agreement or annotator A on ties, agreement report)."""
    key = pd.read_csv(DIRS["annotation"] / "claims_key_DO_NOT_SHARE.csv")
    frames, report = [], {}
    for lang in ("hi", "gu"):
        pa = DIRS["annotation"] / f"claims_{lang}_annotatorA.csv"
        pb = DIRS["annotation"] / f"claims_{lang}_annotatorB.csv"
        if not (pa.exists() and pb.exists()):
            continue
        a, b = pd.read_csv(pa), pd.read_csv(pb)
        m = a.merge(b, on=["sid", "claim_idx"], suffixes=("_a", "_b"))
        m["la"], m["lb"] = m.label_a.map(_lab), m.label_b.map(_lab)
        m = m.dropna(subset=["la", "lb"])
        if m.empty:
            continue
        report[lang] = dict(n_claims=len(m), kappa=cohen_kappa(m.la, m.lb),
                            raw_agreement=float((m.la == m.lb).mean()))
        m["gold"] = m.la  # adjudicate disagreements offline; default to annotator A
        m["error_type"] = m.error_type_a.fillna("")
        m["claim"] = m.claim_a
        frames.append(m[["sid", "claim_idx", "claim", "la", "lb", "gold", "error_type"]])
    if not frames:
        return None, report
    lab = pd.concat(frames).merge(key, on="sid")
    return lab, report
