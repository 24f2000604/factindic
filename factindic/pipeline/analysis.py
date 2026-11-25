"""Aggregation, hypothesis tests, and figures (CPU only). Produces eval/results.json (consumed by
the results dashboard), eval/main_table.csv, and figures/*.png|svg|csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..config import CFG, DIRS, H1_DIR
from ..io import jl_read, jl_read_glob, json_dump, json_load
from ..stats import bootstrap_mean, cohen_kappa, fe_ols, paired_bootstrap

MAIN = ["indicbartss", "mt5", "llama_A", "gemma_A", "gemma_B1", "gemma_B2", "gemma_B3", "gemma_sft",
        "dec_anchor_cad_dola", "drop_only", "cove", "factindic"]
DECODING = ["dec_greedy", "dec_static_cad", "dec_adacad", "dec_dola", "dec_anchor_cad_dola"]
LABEL = {"indicbartss": "IndicBARTSS (Exp 1a)", "mt5": "mT5-base (Exp 1b)", "llama_A": "Llama-3.2-3B · A",
         "gemma_A": "Gemma-4 · A standard", "gemma_B1": "Gemma-4 · B1 anchors", "gemma_B2": "Gemma-4 · B2 CoD",
         "gemma_B3": "Gemma-4 · B3 verb hint", "gemma_sft": "Gemma-4 LoRA-SFT",
         "dec_greedy": "Greedy", "dec_static_cad": "Static CAD", "dec_adacad": "AdaCAD", "dec_dola": "DoLa",
         "dec_anchor_cad_dola": "Anchor-CAD + DoLa", "drop_only": "Drop-only repair", "cove": "CoVe self-rewrite",
         "factindic": "FactIndic (full)"}


def load_all():
    sel = json_load(DIRS["outputs"] / "stage2_selected.json", {}) or {}
    frames = []
    names = set(MAIN) | set(DECODING)
    for f in DIRS["eval"].glob("eval_*.jsonl"):
        if ".shard" in f.name:
            continue
        names.add(f.stem.replace("eval_", ""))
    for s in sorted(names):
        rows = jl_read_glob(DIRS["eval"] / f"eval_{s}.jsonl")
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(), sel
    ev = pd.concat(frames, ignore_index=True)
    meta = []
    for s in ev.system.unique():
        for g in jl_read_glob(DIRS["outputs"] / f"{s}.jsonl"):
            meta.append(dict(system=s, item=g["item"], latency_s=g.get("latency_s"), sum_tokens=g.get("sum_tokens"),
                             input_fertility=g.get("input_fertility"), peak_vram_gb=g.get("peak_vram_gb"),
                             mean_alpha=g.get("mean_alpha")))
    ev = ev.merge(pd.DataFrame(meta), on=["system", "item"], how="left")
    bs = pd.DataFrame(jl_read_glob(DIRS["eval"] / "bertscore.jsonl"))
    if not bs.empty:
        ev = ev.merge(bs[["system", "item", "bertscore"]], on=["system", "item"], how="left")
    ppl = pd.DataFrame(jl_read_glob(DIRS["eval"] / "ppl.jsonl"))
    if not ppl.empty:
        ev = ev.merge(ppl, on=["system", "item"], how="left")
    # dec_greedy = the selected Stage-2 system, restricted to the decoding-ablation articles
    if sel.get("system") and sel["system"] in set(ev.system):
        from .stage3 import ablation_uids
        uids = set(ablation_uids())
        g = ev[(ev.system == sel["system"]) & ev.uid.isin(uids)].copy()
        g["system"] = "dec_greedy"
        ev = pd.concat([ev[ev.system != "dec_greedy"], g], ignore_index=True)
    return ev, sel


METRICS = ["P_fact", "R_fact", "F1_fact", "P_anchor", "R_anchor", "F1_anchor", "rouge1", "rouge2", "rougeL",
           "bertscore", "words", "sum_tokens", "CR", "latency_s", "peak_vram_gb", "input_fertility"]


def main_table(ev):
    rows = []
    for (s, lang, v), d in ev.groupby(["system", "lang", "variant"]):
        r = dict(system=s, label=LABEL.get(s, s), lang=lang, variant=v, n=len(d), empty_rate=float(d["empty"].mean()))
        for m in METRICS:
            if m in d:
                r[m] = float(d[m].astype(float).mean())
        for m in ("P_fact", "F1_fact"):
            b = bootstrap_mean(d[m], CFG["bootstrap"])
            r[f"{m}_lo"], r[f"{m}_hi"] = b["lo"], b["hi"]
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(DIRS["eval"] / "main_table.csv", index=False)
    return t


def overall(ev, systems, metric, uids=None, variants=("native", "script_mixed", "roman")):
    rows = []
    for s in systems:
        d = ev[(ev.system == s) & ev.variant.isin(variants)]
        if uids is not None:
            d = d[d.uid.isin(uids)]
        if d.empty:
            continue
        b = bootstrap_mean(d[metric], CFG["bootstrap"])
        rows.append(dict(system=s, label=LABEL.get(s, s), mean=b["mean"], lo=b["lo"], hi=b["hi"], n=b["n"]))
    return pd.DataFrame(rows)


def paired(ev, a, b, metric, uids=None):
    x = ev[ev.system == a].set_index("item")[metric]
    y = ev[ev.system == b].set_index("item")[metric]
    common = x.index.intersection(y.index)
    if uids is not None:
        keep = set(ev[ev.uid.isin(uids)].item)
        common = [c for c in common if c in keep]
    return paired_bootstrap(x[common].values, y[common].values, CFG["bootstrap"])


# ------------------------------------------------------------------------------------------ H1
MODEL_LABEL = {"gemma4_e2b": "Gemma-4", "sarvam1_sft": "Sarvam-1", "llama32_3b": "Llama-3.2"}
MEASURES = {"halluc_mc": "MiniCheck", "halluc_nli": "Native NLI", "anchor_err": "Names & numbers"}


def h1_sensitivity(ps):
    """Does the gate verdict depend on Llama, and does fertility matter once Romanization is held fixed?"""
    ps = ps.assign(model_doc=ps.model + "_" + ps.doc_key)
    specs = {"within model (gate)": "halluc ~ input_fertility + log_len + C(doc_key) + C(model)",
             "across models (gate)": "halluc ~ input_fertility + log_len + C(cell)",
             "within model, script fixed": "halluc ~ input_fertility + log_len + C(model_doc) + C(script)"}
    subsets = {"all": ps, "no_llama": ps[ps.model != "llama32_3b"]}
    tests = {k: {n: fe_ols(d, f, "input_fertility", cluster="doc_key") for n, f in specs.items()}
             for k, d in subsets.items()}
    fig = viz.coef_plot(tests, {"all": "All models", "no_llama": "Without Llama"}, "h1_sensitivity",
                        "H1: the only positive result goes away once script is held fixed")
    return dict(tests=tests, fig=str(fig))


def h1_measures(ps):
    """Paired native → Romanized change per model × language under each error measure."""
    rows = []
    for (m, lang), d in ps.groupby(["model", "lang"]):
        for col, lab in MEASURES.items():
            w = d.pivot_table(index="doc_key", columns="script", values=col).dropna()
            if len(w) >= 10 and {"native", "roman"} <= set(w.columns):
                b = paired_bootstrap(w["roman"].values, w["native"].values, CFG["bootstrap"])
                rows.append(dict(cell=f"{MODEL_LABEL.get(m, m)}\n{viz.LANG_LABEL[lang]}", measure=lab, change=b["diff"],
                                 lo=b["lo"], hi=b["hi"], n=b["n"]))
    t = pd.DataFrame(rows)
    fig = viz.grouped_bars(t, "h1_shift_by_measure", "Effect of Romanized input depends on which error measure is used",
                           "change in error rate (Romanized − native)", "cell", "measure", "change", err=("lo", "hi"),
                           order_g=list(MEASURES.values()),
                           caption="Same articles in both scripts; bars are paired differences with 95% bootstrap CIs.")
    return dict(table=t.to_dict("records"), fig=str(fig))


def h1_verifier_agreement():
    c = pd.DataFrame(jl_read(H1_DIR / "claim_scores.jsonl"))
    if c.empty:
        return None
    tau, out = CFG["tau_ver"], {}
    for name, d in [("all", c), *c.groupby("script")]:
        mc, nli = (d.p_minicheck >= tau).astype(int), (d.p_nli >= tau).astype(int)
        out[name] = dict(n=len(d), kappa=cohen_kappa(mc, nli), agreement=float((mc == nli).mean()),
                         minicheck_supported=float(mc.mean()), nli_supported=float(nli.mean()),
                         routed_to_nli=int((d.route == "nli").sum()))
    return out


def h1_section():
    out = {}
    p_tab = H1_DIR / "fertility_table.csv"
    p_sum = H1_DIR / "h1_per_summary.csv"
    if p_tab.exists():
        out["fertility_fig"] = str(viz.h1_fertility(pd.read_csv(p_tab)))
    if p_sum.exists():
        ps = pd.read_csv(p_sum)
        ps = ps[ps.script_ok & ps.has_claims]
        out["shift_fig"] = str(viz.h1_shift(ps))
        out["sensitivity"] = h1_sensitivity(ps)
        out["by_measure"] = h1_measures(ps)
        out["verifier_agreement"] = h1_verifier_agreement()
    tests = json_load(H1_DIR / "h1_tests.json")
    if tests:
        out["coef_fig"] = str(viz.coef_plot(
            tests, {"halluc": "Router", "halluc_mc": "MiniCheck"}, "h1_coefficients",
            "H1 gate tests: no reliable positive effect of fertility"))
        out["tests"] = {k: tests[k] for k in ("decision", "model_contrasts_halluc") if k in tests}
        out["regressions"] = {o: tests[o] for o in ("halluc", "halluc_mc", "halluc_nli") if o in tests}
    return out


# ------------------------------------------------------------------------------------------ H2
def h2_section(ev, sel):
    systems = [s for s in ["gemma_A", sel.get("system"), "factindic", "indicbartss"] if s and s in set(ev.system)]
    systems = list(dict.fromkeys(systems))
    rows, tests = [], {}
    for s in systems:
        for lang in ("hi", "gu"):
            for v in ("native", "script_mixed", "roman", "natural_cm"):
                d = ev[(ev.system == s) & (ev.lang == lang) & (ev.variant == v)]
                if d.empty:
                    continue
                b = bootstrap_mean(d.P_fact, CFG["bootstrap"])
                rows.append(dict(system=LABEL.get(s, s), lang=lang, variant=v, mean=b["mean"], lo=b["lo"], hi=b["hi"], n=b["n"]))
        d = ev[(ev.system == s) & ev.variant.isin(["native", "script_mixed", "roman"])].copy()
        tests[s] = {}
        for lang in ("hi", "gu"):
            dl = d[d.lang == lang]
            tests[s][lang] = {f"native→{v}": paired_bootstrap(
                dl[dl.variant == v].set_index("uid").P_fact.reindex(dl[dl.variant == "native"].uid).values,
                dl[dl.variant == "native"].set_index("uid").P_fact.values, CFG["bootstrap"])
                for v in ("script_mixed", "roman")}
        d["gu"] = (d.lang == "gu").astype(int)
        tests[s]["regression"] = {
            "script_mixed": fe_ols(d, "P_fact ~ C(variant, Treatment('native')) + C(uid)",
                                   "C(variant, Treatment('native'))[T.script_mixed]"),
            "roman": fe_ols(d, "P_fact ~ C(variant, Treatment('native')) + C(uid)",
                            "C(variant, Treatment('native'))[T.roman]"),
            "gujarati_vs_hindi_native": fe_ols(d[d.variant == "native"], "P_fact ~ gu", "gu"),
        }
    df = pd.DataFrame(rows)
    figs = {}
    if not df.empty:
        figs["precision"] = str(viz.lines_by_variant(
            df, "h2_precision_by_variant", "H2: claim precision by input modality",
            "Claim-level factual precision (FactCG)", series_col="system",
            caption="Same articles and facts across variants; 95% bootstrap CIs."))
    err = []
    base = ev[ev.system.isin(systems) & ev.variant.isin(["native", "script_mixed", "roman"])]
    for (lang, v), d in base.groupby(["lang", "variant"]):
        n_num = d.n_unsup_num.sum(); n_ent = d.n_unsup_ent.sum(); n_anc = d.n_anchor_S.sum()
        err.append(dict(lang=lang, variant=v, kind="numeric", rate=n_num / max(1, n_anc)))
        err.append(dict(lang=lang, variant=v, kind="entity", rate=n_ent / max(1, n_anc)))
    if err:
        e = pd.DataFrame(err)
        e["group"] = e.lang.map(viz.LANG_LABEL) + " · " + e.kind
        figs["errors"] = str(viz.grouped_bars(
            e, "h2_anchor_errors", "H2: unsupported numbers and entities per summary anchor", "share of anchors unsupported",
            "variant", "group", "rate", order_x=["native", "script_mixed", "roman"], pct=True,
            caption="Gujarati/Gujlish vs Hindi/Hinglish baseline error rates (pooled over the H2 systems)."))
    return dict(tests=tests, figs=figs, table=df.to_dict("records"))


# ------------------------------------------------------------------------------------------ H4
def h4_section(ev):
    from .stage3 import ablation_uids
    uids = set(ablation_uids())
    present = [s for s in DECODING if s in set(ev.system)]
    figs, tests = {}, {}
    for metric, lab, name in (("F1_anchor_num", "Numeric/date anchor F1", "h4_numeric_f1"),
                              ("F1_anchor_ent", "Entity anchor F1", "h4_entity_f1"),
                              ("P_fact", "Claim precision (FactCG)", "h4_claim_precision")):
        df = overall(ev, present, metric, uids)
        if not df.empty:
            figs[name] = str(viz.dot_ci(df, name, f"H4: {lab} by decoding method", lab,
                                        highlight={LABEL["dec_anchor_cad_dola"]}))
    if "ppl" in ev:
        df = overall(ev, present, "ppl", uids)
        if not df.empty:
            figs["h4_ppl"] = str(viz.dot_ci(df, "h4_perplexity", "H4: fluency cost — conditional perplexity",
                                            "Perplexity under the plain model (lower = more fluent)",
                                            highlight={LABEL["dec_anchor_cad_dola"]}))
    for ref in ("dec_static_cad", "dec_adacad", "dec_greedy"):
        if ref in present and "dec_anchor_cad_dola" in present:
            tests[f"anchor_cad_dola−{ref}"] = {m: paired(ev, "dec_anchor_cad_dola", ref, m, uids)
                                               for m in ("F1_anchor_num", "F1_anchor_ent", "P_fact", "F1_fact")
                                               + (("ppl",) if "ppl" in ev else ())}
    return dict(tests=tests, figs=figs)


# ------------------------------------------------------------------------------------------ pipeline
def pipeline_section(ev, sel):
    order = [s for s in ["indicbartss", "mt5", "llama_A", "gemma_A", "gemma_B1", "gemma_B2", "gemma_B3",
                         "gemma_sft", "dec_anchor_cad_dola", "drop_only", "factindic"] if s in set(ev.system)]
    figs, tests = {}, {}
    test_uids = set(ev[ev.variant.isin(["native", "script_mixed", "roman"])].uid)
    for metric, lab in (("F1_fact", "Claim-level factual F1"), ("P_fact", "Claim-level factual precision")):
        df = overall(ev, order, metric, test_uids)
        if not df.empty:
            figs[f"ladder_{metric}"] = str(viz.dot_ci(
                df, f"pipeline_ladder_{metric}", f"Each stage's contribution — {lab}", lab,
                highlight={LABEL["factindic"]},
                caption="Testbed, all three input variants, FactCG held-out grader, 95% bootstrap CIs."))
    pr = []
    for s in order + (["cove"] if "cove" in set(ev.system) else []):
        d = ev[(ev.system == s) & ev.variant.isin(["native", "script_mixed", "roman"])]
        if d.empty:
            continue
        bp, br = bootstrap_mean(d.P_fact), bootstrap_mean(d.R_fact)
        kind = "FactIndic" if s == "factindic" else ("repair" if s in ("drop_only", "cove") else "baseline")
        pr.append(dict(system=LABEL.get(s, s), kind=kind, P=bp["mean"], P_lo=bp["lo"], P_hi=bp["hi"],
                       R=br["mean"], R_lo=br["lo"], R_hi=br["hi"]))
    if pr:
        figs["pr"] = str(viz.pr_scatter(pd.DataFrame(pr), "pipeline_precision_recall",
                                        "Claim precision and recall of every system"))
    lat = overall(ev, order, "latency_s", test_uids)
    if not lat.empty:
        figs["latency"] = str(viz.dot_ci(lat, "pipeline_latency", "Inference cost per summary",
                                         "seconds per summary (batched generation + verification)",
                                         highlight={LABEL["factindic"]}))
    base = sel.get("system")
    for ref in ("gemma_A", base, "drop_only", "cove", "indicbartss"):
        if ref and ref in set(ev.system) and "factindic" in set(ev.system):
            tests[f"factindic−{ref}"] = {m: paired(ev, "factindic", ref, m) for m in ("P_fact", "R_fact", "F1_fact", "F1_anchor")}
    if "gemma_A" in set(ev.system) and base in set(ev.system):
        tests[f"{base}−gemma_A (factuality-aware vs standard prompt)"] = {
            m: paired(ev, base, "gemma_A", m) for m in ("P_fact", "R_fact", "F1_fact", "F1_anchor")}
    # stage-4 diagnostics
    s4 = jl_read_glob(DIRS["verify"] / "stage4.jsonl")
    diag = {}
    if s4:
        ks = pd.Series([r["selected_k"] for r in s4]).value_counts(normalize=True).sort_index()
        diag["selected_k_share"] = ks.to_dict()
        diag["pruned_rate"] = float(np.mean([r["pruned"] for r in s4]))
        routes = [v["route"] for r in s4 for c in r["candidates"] for v in c["verdicts"]]
        var = [r["variant"] for r in s4 for c in r["candidates"] for v in c["verdicts"]]
        rt = pd.DataFrame(dict(route=routes, variant=var))
        share = rt.assign(nli=rt.route == "nli").groupby("variant").nli.mean().reset_index()
        share["lang"] = "all"
        diag["nli_route_share"] = share.set_index("variant").nli.to_dict()
        figs["route_share"] = str(viz.grouped_bars(
            share.rename(columns={"nli": "share"}), "stage4_route_share",
            "Round-trip router: share of claims whose anchors failed MT (→ native NLI)", "share of claims",
            "variant", "lang", "share", pct=True,
            order_x=[v for v in ["native", "script_mixed", "roman", "natural_cm"] if v in set(share.variant)]))
    return dict(tests=tests, figs=figs, stage4=diag)


def run_all():
    ev, sel = load_all()
    res = dict(selected=sel)
    parts = dict(h1=h1_section)
    if not ev.empty:
        res["main_table"] = main_table(ev).to_dict("records")
        parts.update(h2=lambda: h2_section(ev, sel), h4=lambda: h4_section(ev), pipeline=lambda: pipeline_section(ev, sel))
    for k, f in parts.items():               # one failing section must not lose the others
        try:
            res[k] = f()
        except Exception:
            import traceback
            traceback.print_exc()
            res[k] = dict(error=traceback.format_exc()[-600:])
    res["stage3_params"] = json_load(DIRS["outputs"] / "stage3_params.json")
    res["stage2_dev"] = json_load(DIRS["outputs"] / "stage2_dev_scores.json")
    res["sanitation"] = json_load(DIRS["data"] / "sanitation_report.json")
    human = json_load(DIRS["eval"] / "human_results.json")
    if human:
        res["h3"] = human
    json_dump(res, DIRS["eval"] / "results.json")
    return res
