"""All numbers of the paper, from the finished run: python paper/compute_numbers.py → paper/numbers.json
CPU only. Adds what the run did not store: ROUGE precision/recall, token counts, and claim recall/F1 restricted to
items with at least one source-supported reference claim (elsewhere recall is undefined)."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np                                           # noqa: E402

from factindic.config import CFG, DIRS, MODELS               # noqa: E402
from factindic.data import load_set                          # noqa: E402
from factindic.io import jl_read_glob, json_load             # noqa: E402
from factindic.metrics import IndicTok                       # noqa: E402
from factindic.pipeline.analysis import load_all, paired     # noqa: E402
from factindic.pipeline.stage3 import ablation_uids          # noqa: E402
from factindic.stats import bootstrap_mean, fe_ols, paired_bootstrap    # noqa: E402

MAIN3 = ["native", "script_mixed", "roman"]
SYS = ["mt5", "indicbartss", "llama_A", "gemma_A", "gemma_B1", "gemma_B2", "gemma_B3", "gemma_sft", "dec_anchor_cad_dola",
       "drop_only", "factindic"]
DEC = ["dec_greedy", "dec_static_cad", "dec_adacad", "dec_dola", "dec_anchor_cad_dola"]


def mean(x):
    b = bootstrap_mean(np.asarray(x, float), CFG["bootstrap"])
    return dict(mean=b["mean"], lo=b["lo"], hi=b["hi"], n=b["n"])


def f1(d):
    p, r = d.P_fact.fillna(0.0), d.R_fact
    return (2 * p * r / (p + r).where(p + r > 0)).fillna(0.0)


ev, sel = load_all()
tb = ev[ev.variant.isin(MAIN3)].copy()
recs = {r["uid"]: r for s in ("testbed", "natural_cm") for r in load_set(s)}
from rouge_score import rouge_scorer                         # noqa: E402
from transformers import AutoTokenizer                       # noqa: E402
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=IndicTok())
gem = AutoTokenizer.from_pretrained(MODELS["gemma"])
out = dict(selected=sel, systems={}, decoding={}, variants={}, tests={})
for s in SYS + ["cove"]:
    d = tb[tb.system == s]
    texts = {g["item"]: g["summary"] for g in jl_read_glob(DIRS["outputs"] / f"{s}.jsonl")}
    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    toks = []
    for it, uid in zip(d.item, d.uid):
        sc = scorer.score(recs[uid]["reference"], texts[it])
        for k in R:
            R[k].append((sc[k].precision, sc[k].recall, sc[k].fmeasure))
        toks.append(len(gem(texts[it], add_special_tokens=False)["input_ids"]))
    ok = d[d.R_fact.notna()]
    ok = ok[ok.P_fact.notna()] if False else ok
    row = dict(n=len(d), P=mean(d.P_fact), n_P=int(d.P_fact.notna().sum()), R=mean(ok.R_fact), F1=mean(f1(ok)), n_R=len(ok),
               P_nonroman=mean(d[d.variant != "roman"].P_fact), empty=float(d["empty"].mean()),
               Pa=mean(d.P_anchor), Ra=mean(d.R_anchor), Fa=mean(d.F1_anchor), bert=mean(d.bertscore),
               words=float(d.words.mean()), tokens_gemma=float(np.mean(toks)), CR=float(d.CR.mean()), latency=float(d.latency_s.mean()))
    for k, v in R.items():
        row[k] = [100 * float(np.mean([x[i] for x in v])) for i in range(3)]
    out["systems"][s] = row
uids = set(ablation_uids())
for s in DEC:
    d = tb[(tb.system == s) & tb.uid.isin(uids)]
    out["decoding"][s] = dict(n=len(d), P=mean(d.P_fact), ent=mean(d.F1_anchor_ent), num=mean(d.F1_anchor_num), ppl=mean(d.ppl),
                              latency=float(d.latency_s.mean()))
res = json_load(DIRS["eval"] / "results.json")
out["h4"], out["h2"], out["pipeline"] = res["h4"]["tests"], res["h2"]["tests"], res["pipeline"]["tests"]
out["stage4"], out["params"], out["dev"] = res["pipeline"]["stage4"], res["stage3_params"], res["stage2_dev"]
out["h1"] = dict(sens=res["h1"]["sensitivity"]["tests"], contrasts=res["h1"]["tests"]["model_contrasts_halluc"],
                 agreement=res["h1"]["verifier_agreement"], shift=res["h1"]["by_measure"]["table"])
out["lookback"] = {k: v for k, v in (res.get("lookback") or res["h3"]["lookback"]).items() if k != "fig"}
for s in ("gemma_A", "gemma_B1", "factindic", "llama_A", "indicbartss", "mt5", "gemma_sft"):
    out["variants"][s] = {f"{l}/{v}": mean(d.P_fact) for (l, v), d in ev[ev.system == s].groupby(["lang", "variant"])}
for a, b in (("factindic", "gemma_A"), ("factindic", "gemma_B1"), ("gemma_B1", "gemma_A"), ("factindic", "drop_only"), ("gemma_B2", "gemma_B1"),
             ("gemma_B3", "gemma_B1"), ("gemma_sft", "gemma_A")):
    A, B = tb[tb.system == a].set_index("item"), tb[tb.system == b].set_index("item")
    c = A.index.intersection(B.index)
    t = {m: paired_bootstrap(A.loc[c, m].values, B.loc[c, m].values, CFG["bootstrap"]) for m in ("P_fact", "F1_anchor", "R_anchor", "rougeL")}
    k = [i for i in c if A.loc[i, "R_fact"] == A.loc[i, "R_fact"] and B.loc[i, "R_fact"] == B.loc[i, "R_fact"]]
    t["F1_fact_defined"] = paired_bootstrap(f1(A.loc[k]).values, f1(B.loc[k]).values, CFG["bootstrap"])
    nr = [i for i in c if A.loc[i, "variant"] != "roman"]
    t["P_fact_nonroman"] = paired_bootstrap(A.loc[nr, "P_fact"].values, B.loc[nr, "P_fact"].values, CFG["bootstrap"])
    out["tests"][f"{a}-{b}"] = t
# H2 second clause: unsupported entities and numbers per summary anchor on native input, Gemma systems of H2
d = tb[tb.system.isin(["gemma_A", "gemma_B1", "factindic"]) & (tb.variant == "native")]
out["h2_errors"] = {l: dict(ent=float(x.n_unsup_ent.sum() / x.n_anchor_S.sum()), num=float(x.n_unsup_num.sum() / x.n_anchor_S.sum()),
                            anchors=int(x.n_anchor_S.sum())) for l, x in d.groupby("lang")}
nc = load_set("natural_cm")
out["cmi_natural"] = {l: float(np.mean([r["cmi"]["natural_cm"] for r in nc if r["lang"] == l])) for l in ("hi", "gu")}
out["cmi_mixed"] = {l: float(np.mean([r["cmi"]["script_mixed"] for r in load_set("testbed") if r["lang"] == l])) for l in ("hi", "gu")}
refs = jl_read_glob(DIRS["eval"] / "refs.jsonl")
g = [x for r in refs for x in r["grounded"]]
out["refs"] = dict(claims=len(g), grounded=float(np.mean(g)))
out["sanitation"] = json_load(DIRS["data"] / "sanitation_report.json")
w = jl_read_glob(DIRS["data"] / "weights.jsonl")
out["weights"] = {l: dict(n=sum(x["lang"] == l for x in w), kept=sum(x["lang"] == l and x["keep"] for x in w)) for l in ("hi", "gu")}
import re                                                     # noqa: E402
lat = lambda t: len(re.findall("[A-Za-z]", t)) > 0.3 * max(1, len(t))
tu = {r["uid"] for r in load_set("testbed")}
out["latin_out"] = {s: sum(lat(x["summary"]) for x in jl_read_glob(DIRS["outputs"] / f"{s}.jsonl") if x["variant"] == "roman" and x["uid"] in tu)
                    for s in SYS}
out["latency_single"] = {f.stem: json_load(f) for f in DIRS["outputs"].glob("latency_*.json")}
out["train"] = json_load(DIRS["models"] / "gemma_sft" / "train_info.json")
(REPO / "paper" / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
print("ok", out["cmi_natural"], out["cmi_mixed"], out["refs"])
