"""Build self-contained Kaggle/Colab notebooks: each embeds the current `factindic` package as a
base64 tarball (extracted to /tmp at run time), so no GitHub clone or dataset upload is needed.

    python tools/build_notebooks.py        # rewrites notebooks/0[1-7]_*.ipynb
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(q for q in (ROOT / "factindic").rglob("*") if q.suffix in (".py", ".txt", ".csv", ".json", ".jsonl")):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


REQ = (ROOT / "factindic" / "requirements.txt").read_text(encoding="utf-8")
ENV = '''# Environment: one pinned stack for every notebook (tools/build_notebooks.py writes this cell from
# factindic/requirements.txt). IndicTrans2 gets its own small environment in the preflight cell below.
REQUIREMENTS = """
''' + REQ + '''"""
open("/tmp/factindic_requirements.txt", "w").write(REQUIREMENTS)
!pip install -q -r /tmp/factindic_requirements.txt'''

PREFLIGHT = '''# Preflight: versions, Hugging Face access, translator. Stops here if anything is missing.
from factindic import preflight
preflight.check({args})'''

BOOT = '''# Unpack the factindic package embedded in this notebook (built by tools/build_notebooks.py)
import base64, io, os, sys, tarfile
PKG_B64 = "{b64}"
SRC = "/tmp/factindic_src"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG_B64))).extractall(SRC, filter="data")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")
from factindic.config import setup, CFG, DIRS
info = setup()'''

SHOW = '''from IPython.display import Image, Markdown, display
for p in sorted(DIRS["figures"].glob("{pattern}*.png")):
    display(Markdown(f"**{{p.stem}}**")); display(Image(str(p), width=760))'''

ZIP = '''# Package this notebook's outputs; attach this notebook's output as an input to the next notebook
!cd /kaggle/working 2>/dev/null && zip -qr factindic_outputs.zip factindic factindic_h1 -x "*/models/*/ckpt/*" || true
print("done:", DIRS["outputs"].parent)'''

HEADER = '''# FactIndic — {title}

**Run on:** Kaggle (Settings → Accelerator **GPU T4 x2**, Internet **On**, Secret `HF_TOKEN`) or Colab (T4).
**Inputs:** {inputs} · **Rough T4 time:** {time}

{body}

Every step writes JSONL incrementally and skips finished work, so after a disconnect just *Run All* again
(on Kaggle, attach this notebook's previous output as an input first).'''

NOTEBOOKS = [
    dict(file="01_data_stage1.ipynb", title="01 · Data, sanitation, Stage-1 anchors, faithfulness weights",
         preflight='mt=True, required=("it2", "indicner"), need_data=False', inputs="the output of an earlier run of this notebook, if there is one (its translations are reused)", time="~2–2.5 h from scratch, ~30–40 min with the previous (data version 2) output attached",
         body="Builds the ILSUM 5-rule sanitation, the 300-article parallel testbed (native / script-mixed / "
              "Romanized), the 100-article dev split, the ILSUM natural code-mixed subset, normalized "
              "anchor sets A_core(D) with IndicNER, and mFACT-style weights w_i = clip(s_MC, 0.3, 1).",
         cells=[
             ("md", "## 1. Datasets, sanitation, testbed, anchors"),
             ("code", "from factindic.pipeline import stage1\nstage1.build_all()\n"
                      "from factindic.io import json_load\njson_load(DIRS['data'] / 'sanitation_report.json')"),
             ("md", "## 2. Faithfulness weights (IndicTrans2 → MiniCheck on lead + reference) — both GPUs"),
             ("code", "from factindic import parallel\nstage1.prepare_weights()\nparallel.run('factindic.pipeline.stage1:weights_job')\n"
                      "pool = stage1.weighted_pool(); print(len(pool), 'weighted pairs kept')"),
             ("md", "## 3. Naturalness sheets for 2 native speakers per language"),
             ("code", "from factindic.annotate import export_naturalness\nfrom factindic.data import load_set\n"
                      "export_naturalness(load_set('testbed'))"),
             ("md", "## 4. Figures"),
             ("code", "stage1.figures()\n" + SHOW.format(pattern="data_")),
         ]),
    dict(file="02_seq2seq_baselines.ipynb", title="02 · Seq2Seq baselines (Exp 1a IndicBARTSS, Exp 1b mT5-base)",
         preflight='', inputs="output of 01", time="~2–2.5 h",
         body="Standard (unweighted) fine-tuning on the sanitized pool, beam-4 decoding on every testbed variant "
              "and the natural code-mixed subset.",
         cells=[
             ("code", "from factindic.pipeline import stage2\nfrom factindic import parallel\n"
                      "stage2.train_seq2seq('indicbartss')\n"
                      "parallel.run('factindic.pipeline.stage2:seq2seq_gen_job', name='indicbartss')"),
             ("code", "stage2.train_seq2seq('mt5')\n"
                      "parallel.run('factindic.pipeline.stage2:seq2seq_gen_job', name='mt5')"),
         ]),
    dict(file="03_stage2_prompting_sft.ipynb", title="03 · Stage 2: Standard vs Factuality-Aware prompting, LoRA-SFT, Llama",
         preflight='mt=True, required=("it2", "gemma"), optional=("llama",)', inputs="output of 01 (and 02)", time="~4–8.5 h (split across sessions if needed)",
         body="Dev: Gemma-4 Cond A / B1 / B2 / B3 and faithfulness-weighted LoRA-SFT → the dev grader "
              "(IndicTrans2→MiniCheck router; *not* the evaluation grader) selects the Stage-3 base. Test: every "
              "condition on the testbed + natural code-mixed subset; Llama-3.2-3B Cond A; batch-1 latency probes.",
         cells=[
             ("md", "## 1. Dev generations (zero-shot conditions)"),
             ("code", "from factindic.pipeline import stage2\nfrom factindic import parallel\n"
                      "for cond in ['A', 'B1', 'B2', 'B3']:\n"
                      "    parallel.run('factindic.pipeline.stage2:llm_gen_job', system=f'gemma_{cond}', model_key='gemma',\n"
                      "                 cond=cond, sets=['dev'])"),
             ("md", "## 2. Faithfulness-weighted LoRA-SFT of Gemma-4 (r=16, α=32, q/k/v/o)"),
             ("code", "from factindic.sft import train_lora\nfrom factindic.llm import load_tokenizer\n"
                      "from factindic.config import MODELS\n"
                      "tok = load_tokenizer(MODELS['gemma'], padding_side='left')\n"
                      "ex = stage2.sft_examples(tok); print(len(ex), 'SFT examples')\n"
                      "ADAPTER = str(train_lora(MODELS['gemma'], ex, DIRS['models'] / 'gemma_sft', CFG['sft']))\n"
                      "from factindic.io import free_gpu; free_gpu()"),
             ("code", "parallel.run('factindic.pipeline.stage2:llm_gen_job', system='gemma_sft', model_key='gemma',\n"
                      "             cond='A', sets=['dev'], adapter=ADAPTER)"),
             ("md", "## 3. Dev selection of the Stage-3 base (B2 is two-pass, so it cannot be the single-pass decoding base)"),
             ("code", "from factindic.io import json_dump\n"
                      "scores = {s: stage2.dev_grade(s) for s in ['gemma_A', 'gemma_B1', 'gemma_B2', 'gemma_B3', 'gemma_sft']}\n"
                      "stage2.release_dev_router()\n"
                      "json_dump(scores, DIRS['outputs'] / 'stage2_dev_scores.json')\n"
                      "cands = {'gemma_B1': ('B1', None), 'gemma_B3': ('B3', None), 'gemma_sft': ('A', ADAPTER)}\n"
                      "best = max(cands, key=lambda s: (scores[s]['score'], scores[s]['rougeL']))\n"
                      "json_dump(dict(system=best, model_key='gemma', cond=cands[best][0], adapter=cands[best][1]),\n"
                      "          DIRS['outputs'] / 'stage2_selected.json')\n"
                      "import pandas as pd; display(pd.DataFrame(scores).T); print('Stage-3 base:', best)"),
             ("md", "## 4. Test generations (testbed × 3 variants + natural code-mixed)"),
             ("code", "for cond in ['A', 'B1', 'B3', 'B2']:\n"
                      "    parallel.run('factindic.pipeline.stage2:llm_gen_job', system=f'gemma_{cond}', model_key='gemma', cond=cond)\n"
                      "parallel.run('factindic.pipeline.stage2:llm_gen_job', system='gemma_sft', model_key='gemma', cond='A', adapter=ADAPTER)"),
             ("code", "# Exp 2c: Llama-3.2-3B (gated; up to 320 new tokens because Gujarati costs it ~10 tokens per word)\n"
                      "try:\n"
                      "    parallel.run('factindic.pipeline.stage2:llm_gen_job', system='llama_A', model_key='llama', cond='A')\n"
                      "except Exception as e:\n"
                      "    print('Llama skipped:', e)"),
             ("md", "## 5. Per-summary latency at batch size 1"),
             ("code", "from factindic.io import json_load\nsel = json_load(DIRS['outputs'] / 'stage2_selected.json')\n"
                      "for s, mk, c, ad in [('gemma_A', 'gemma', 'A', None), (sel['system'], 'gemma', sel['cond'], sel['adapter'])]:\n"
                      "    print(stage2.latency_probe(s, mk, c, ad))"),
         ]),
    dict(file="04_stage3_decoding.ipynb", title="04 · Stage 3: Anchor-aware CAD + DoLa, K=5 candidates, CoVe, Lookback",
         preflight='mt=True, required=("it2", "gemma")', inputs="output of 03", time="~3.5–6.5 h",
         body="Tunes (α_base, α_anchor, β) on dev; runs the decoding ablation (Exp 4a–4d) on 150 articles × 3 "
              "variants; runs Anchor-CAD+DoLa on all inputs plus 4 nucleus samples (K=5); CoVe self-rewrite "
              "baseline; conditional perplexity; per-head lookback features.",
         cells=[
             ("md", "## 1. Tune on dev (generate grid → grade with the dev grader)"),
             ("code", "from factindic.pipeline import stage3\nfrom factindic import parallel\n"
                      "parallel.run('factindic.pipeline.stage3:tune_generate_job')\n"
                      "best, grid = stage3.pick_params()\nimport pandas as pd; display(pd.DataFrame(grid)); print('selected', best)"),
             ("md", "## 2. Decoding ablation (Exp 4a–4c) on the ablation articles"),
             ("code", "U = stage3.ablation_uids()\n"
                      "for m in ['static_cad', 'adacad', 'dola']:\n"
                      "    parallel.run('factindic.pipeline.stage3:decode_job', method=m, system=f'dec_{m}', uids=U)"),
             ("md", "## 3. Anchor-CAD + DoLa on every input (Exp 4d; candidate k=0) and 4 samples (k=1..4)"),
             ("code", "parallel.run('factindic.pipeline.stage3:decode_job', method='anchor_cad_dola', system='dec_anchor_cad_dola',\n"
                      "             sets=['testbed', 'natural_cm'])\n"
                      "parallel.run('factindic.pipeline.stage3:samples_job')"),
             ("md", "## 4. CoVe self-rewrite baseline (Exp 5b) · perplexity · lookback features"),
             ("code", "parallel.run('factindic.pipeline.stage3:cove_job')\n"
                      "sel = stage3.selected()\n"
                      "parallel.run('factindic.pipeline.stage3:ppl_job', systems=[sel['system'], 'dec_static_cad', 'dec_adacad', 'dec_dola', 'dec_anchor_cad_dola'])\n"
                      "parallel.run('factindic.pipeline.stage3:lookback_job', systems=[sel['system'], 'dec_anchor_cad_dola'])"),
         ]),
    dict(file="05_stage4_verify_rerank.ipynb", title="05 · Stage 4: Round-trip router, K=5 reranking, pruning, drop-only",
         preflight='mt=True, required=("it2", "indicner")', inputs="output of 04", time="~1.5–2 h",
         body="IndicTrans2 projection (digits normalized) → round-trip anchor check → MiniCheck-FT5 (anchors "
              "survived) or native mDeBERTa-XNLI (anchors lost); J(S_k) = frac_supported · R_anchor − λ·|W−L|/L; "
              "single capped pruning pass. Romanized sources are back-transliterated (deployment mode); oracle "
              "native-source verdicts are logged for the H3 proxy.",
         cells=[
             ("code", "from factindic.pipeline import stage4\nfrom factindic import parallel\n"
                      "parallel.run('factindic.pipeline.stage4:verify_job')\nprint(stage4.materialize(), 'final summaries')"),
             ("code", "import pandas as pd\nfrom factindic.io import jl_read_glob\n"
                      "s4 = pd.DataFrame(jl_read_glob(DIRS['verify'] / 'stage4.jsonl'))\n"
                      "print(s4.groupby('variant').agg(pruned=('pruned', 'mean'), k0=('selected_k', lambda x: (x == 0).mean())))"),
         ]),
    dict(file="06_evaluate.ipynb", title="06 · Non-circular evaluation, hypothesis tests, figures",
         preflight='mt=True, required=("it2", "indicner")', inputs="outputs of 02 and 05 (+ H1 output for the H1 figures)", time="~1.5–2.5 h",
         body="Held-out FactCG-DeBERTa-L (never used for weights or reranking) grades every system against the "
              "aligned native source: claim P/R/F1 with source-grounded reference recall, normalized anchor P/R/F1 "
              "(numbers and entities separately), Indic-tokenized ROUGE, rescaled IndicBERTv2 BERTScore, length, "
              "compression, fertility, latency, VRAM; paired bootstrap tests for H1, H2, H4 and every pipeline stage; "
              "then exports the blinded human-annotation sheets.",
         cells=[
             ("code", "from factindic.pipeline import evaluate, analysis\nfrom factindic import parallel\n"
                      "SYSTEMS = [s for s in ['indicbartss', 'mt5', 'llama_A', 'gemma_A', 'gemma_B1', 'gemma_B2', 'gemma_B3', 'gemma_sft',\n"
                      "           'dec_static_cad', 'dec_adacad', 'dec_dola', 'dec_anchor_cad_dola', 'drop_only', 'cove', 'factindic']\n"
                      "           if any(DIRS['outputs'].glob(f'{s}.*jsonl'))]\nprint(SYSTEMS)\n"
                      "parallel.run('factindic.pipeline.evaluate:ref_job')\n"
                      "parallel.run('factindic.pipeline.evaluate:eval_job', systems=SYSTEMS)\n"
                      "parallel.run('factindic.pipeline.evaluate:bertscore_job', systems=SYSTEMS)"),
             ("code", "res = analysis.run_all()\nimport pandas as pd\n"
                      "t = pd.read_csv(DIRS['eval'] / 'main_table.csv')\n"
                      "display(t.pivot_table(index='label', columns='variant', values='F1_fact').round(3))"),
             ("md", "## Figures — hypotheses and pipeline justification"),
             ("code", SHOW.format(pattern="h")),
             ("code", SHOW.format(pattern="pipeline_") + "\n" + SHOW.format(pattern="stage4_").split("\n", 1)[1]),
             ("md", "## Export blinded annotation sheets (2 annotators × 2 languages)"),
             ("code", "from factindic.pipeline import human\nhuman.export_sheets()"),
         ]),
    dict(file="07_human_eval_results.ipynb", title="07 · Human evaluation, H3, Lookback detector, final results",
         preflight='mt=True, required=("it2",)', inputs="output of 06 + a Kaggle dataset with the filled annotation CSVs", time="~30 min",
         body="Copy the four filled `claims_<lang>_annotator<A|B>.csv` files into `factindic/annotation/`. Computes "
              "inter-annotator κ, FactCG-vs-human κ, the H3 verifier-agreement test by modality (thresholds tuned on "
              "the 100-summary dev split), the label-free H3 proxy, and the Lookback-Lens detector; rebuilds "
              "`eval/results.json` for the results dashboard.",
         cells=[
             ("code", "import glob, shutil\nfor f in glob.glob('/kaggle/input/**/claims_*_annotator*.csv', recursive=True):\n"
                      "    shutil.copy(f, DIRS['annotation']); print('annotation:', f)"),
             ("code", "from factindic.pipeline import human, analysis\nhres = human.run()\nres = analysis.run_all()\n"
                      "import json; print(json.dumps({k: hres[k] for k in ('inter_annotator', 'factcg_vs_human') if k in hres}, indent=2))"),
             ("code", SHOW.format(pattern="h3_") + "\n" + SHOW.format(pattern="lookback").split("\n", 1)[1]),
         ]),
]


RUN_ALL = dict(
    file="FactIndic_run_all.ipynb", title="The whole project in one notebook (01–06)",
    preflight='mt=True, required=("it2", "gemma", "indicner"), optional=("llama", "indicbartss", "indicbert"), need_data=False',
    inputs="the `factindic_inputs` dataset (H1 results and training-pool scores; data only)",
    time="~6.5–7.5 h, stops by itself after 8 h 15 of work",
    body="Use **Save Version → Save & Run All (Commit)**; the page can be closed. Part 1 builds the data (notebook 01), "
         "Part 2 tests every step on 4 articles and prints `SMOKE TEST: …` (about 1 h after the start), Part 3 runs 01's scores and "
         "02–06. Failed jobs are retried and skipped, optional ones are dropped if time runs short, and the analysis always runs.",
    cells=[
        ("md", "## Part 1. Data: sanitation, testbed, anchors, training pool (notebook 01)"),
        ("code", "from factindic.pipeline import auto\nauto.part1_data()"),
        ("md", "## Part 2. Smoke test of every step"),
        ("code", "auto.part2_smoke()"),
        ("md", "## Part 3. Faithfulness weights, baselines, prompting, SFT, decoding, verification, evaluation (01–06)"),
        ("code", "status = auto.part3_full()\nstatus"),
        ("md", "## Part 4. Results and figures"),
        ("code", "import pandas as pd\nf = DIRS['eval'] / 'main_table.csv'\n"
                 "if f.exists():\n    t = pd.read_csv(f)\n"
                 "    display(t.pivot_table(index='label', columns='variant', values=['P_fact', 'F1_fact']).round(3))"),
        ("code", SHOW.format(pattern="")),
    ])


def make(nb, b64):
    cells = [dict(cell_type="markdown", metadata={}, source=HEADER.format(**nb)),
             dict(cell_type="code", metadata={}, source=ENV, outputs=[], execution_count=None),
             dict(cell_type="code", metadata={}, source=BOOT.format(b64=b64), outputs=[], execution_count=None),
             dict(cell_type="code", metadata={}, source=PREFLIGHT.format(args=nb["preflight"]), outputs=[],
                  execution_count=None)]
    for kind, src in nb["cells"]:
        if kind == "md":
            cells.append(dict(cell_type="markdown", metadata={}, source=src))
        else:
            cells.append(dict(cell_type="code", metadata={}, source=src, outputs=[], execution_count=None))
    cells.append(dict(cell_type="code", metadata={}, source=ZIP, outputs=[], execution_count=None))
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                          "accelerator": "GPU",
                          "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}})


# --------------------------------------------------------------------------- readable single notebook
SRC = "/tmp/factindic_src"
GROUPS = [
    ("Configuration and shared helpers", ["__init__", "config", "io", "parallel", "stats"],
     "Paths, model list, sizes, checkpointed JSONL files, multi-GPU workers, bootstrap and regression tests."),
    ("Text processing: Hindi, Gujarati, Romanized and code-mixed", ["text", "translit", "lid", "normalizer", "ner"],
     "Cleaning, sentence splitting, Romanization and script mixing, word-level language ID and CMI, the number/date/entity "
     "normalizer with the anchor set A_core, and named entities."),
    ("Datasets (Stage 0)", ["data"], "XL-Sum and ILSUM, the 5-rule sanitation, the parallel testbed, the dev and natural code-mixed sets, and the training pool."),
    ("Indic to English translation", ["it2_env", "it2_server", "mt"], "IndicTrans2 in its own environment, with a translation cache."),
    ("Verifiers and metrics", ["verify", "claims", "rerank", "metrics"],
     "MiniCheck, mDeBERTa-XNLI, FactCG (held out for evaluation), the round-trip router, reranking, and the P/R/F1 metrics."),
    ("Generation: prompts, LLM loading, LoRA SFT", ["llm", "prompts", "sft"], "Prompt conditions A, B1, B2, B3 and faithfulness-weighted LoRA fine-tuning."),
    ("Decoding: anchor-aware CAD + DoLa, CoVe, Lookback Lens", ["decoding", "cove", "lookback"], "The contrastive decoding methods of Stage 3 and the self-verification baseline."),
    ("Figures, annotation sheets, checks", ["viz", "annotate", "preflight"], ""),
    ("Pipeline stages", ["pipeline/__init__", "pipeline/stage1", "pipeline/stage2", "pipeline/stage3", "pipeline/stage4",
                         "pipeline/evaluate", "pipeline/analysis", "pipeline/human", "pipeline/auto"],
     "Stage 1 to Stage 4, evaluation, hypothesis tests, and the job queue that runs everything on both GPUs."),
]
TEXT_FILES = ["requirements.txt", "requirements-it2.txt", "h1_pilot_ids.txt"]
MKDIR = f"""import os
SRC = "{SRC}"
os.makedirs(f"{{SRC}}/factindic/pipeline", exist_ok=True)"""
PIP = f"""!pip install -q -r {SRC}/factindic/requirements.txt"""
START_CELL = """import glob, os, shutil, sys
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

# factindic_inputs dataset: results of the finished H1 run, and the training-pool ids with their faithfulness scores
for f in glob.glob("/kaggle/input/**/pool_seed.txt", recursive=True)[:1]:
    shutil.copy(f, f"{SRC}/factindic/pool_seed.txt"); print("training-pool scores:", f)
for d in glob.glob("/kaggle/input/**/h1_results", recursive=True)[:1]:
    shutil.copytree(d, f"{SRC}/factindic/h1_results", dirs_exist_ok=True); print("H1 results:", d)
assert os.path.exists(f"{SRC}/factindic/pool_seed.txt"), "Add the factindic_inputs dataset (Add Input), then run again."

from factindic.config import setup, CFG, DIRS
info = setup()"""


def make_readable(nb):
    """Every source file as its own %%writefile cell: the code is visible, and worker processes import it from disk."""
    md = lambda s: dict(cell_type="markdown", metadata={}, source=s)
    code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)
    write = lambda name: code(f"%%writefile {SRC}/factindic/{name}\n" + (ROOT / "factindic" / name).read_text(encoding="utf-8").rstrip("\n"))
    cells = [md(HEADER.format(**nb)), md("## Setup"), code(MKDIR)] + [write(f) for f in TEXT_FILES] + [code(PIP)]
    listed = set()
    for title, files, note in GROUPS:
        cells.append(md(f"## {title}\n\n{note}".rstrip()))
        for f in files:
            listed.add(f + ".py")
            cells += [md(f"### `factindic/{f}.py`"), write(f + ".py")]
    have = {str(p.relative_to(ROOT / "factindic")) for p in (ROOT / "factindic").rglob("*.py")}
    assert have == listed, f"not in GROUPS: {have ^ listed}"
    cells += [md("# Run"), code(START_CELL), code(PREFLIGHT.format(args=nb["preflight"]))]
    cells += [(md if kind == "md" else code)(src) for kind, src in nb["cells"]]
    cells.append(code(ZIP))
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
    out = make(nb, "")
    out["cells"] = cells
    return out


def inputs_zip():
    """The small dataset the single notebook needs (data only, no code)."""
    import zipfile
    path = ROOT / "factindic_inputs.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [ROOT / "factindic" / "pool_seed.txt", *sorted((ROOT / "factindic" / "h1_results").iterdir())]:
            z.write(p, p.relative_to(ROOT / "factindic"))
    return path


if __name__ == "__main__":
    b64 = bundle()
    NB.mkdir(exist_ok=True)
    path = NB / RUN_ALL["file"]
    path.write_text(json.dumps(make_readable(RUN_ALL), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    print(inputs_zip().name)
    for nb in NOTEBOOKS:
        path = NB / nb["file"]
        path.write_text(json.dumps(make(nb, b64), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
