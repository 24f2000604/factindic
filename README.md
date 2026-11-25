# FactIndic

A four-stage, zero-cost pipeline for factuality-aware summarization in Hindi, Gujarati, Hinglish
and Gujlish. The full specification is `Factuality-Aware Indic Summarization Research.pdf`.

It runs on a laptop with one 8 GB GPU (`run_local.py`, see below) or on a free **Kaggle T4 x2**
(`notebooks/FactIndic_run_all.ipynb`).

## Results of the final run (25–26 Nov 2026)

Full report with confidence intervals and limitations: [`results/final_run/REPORT.md`](results/final_run/REPORT.md).

| Hypothesis | Verdict |
|---|---|
| H1 fertility ↔ hallucination | Not supported (see below) |
| H2 precision drop across modalities | Partly supported: Romanized input lowers claim precision strongly, script-mixed input does not |
| H3 router vs human labels | Planned, not evaluated: it needs human annotation, which did not fit the timeline. No claim is made; future work |
| H4 anchor-aware CAD | Not supported |
| Full pipeline | Claim precision 0.756 against 0.728 (standard prompt) and 0.752 (anchor checklist); neither difference is significant |

## Run on a laptop (RTX 40/50, 8 GB)

```bash
bash setup_local.sh && source .venv/bin/activate     # Windows without WSL2: setup_local.ps1
hf auth login
python run_local.py doctor                           # hardware, packages, model access
python run_local.py download                         # ~30 GB of models
python run_local.py all                              # data, smoke test, full run, figures
python run_local.py status                           # in a second terminal
```

On 8 GB, Gemma-4 runs in bf16 with its 4.7 GB per-layer embedding table in RAM, Llama in 4-bit, MiniCheck in bf16, and mT5
trains with Adafactor. `CLAUDE.md` has the run book.

## H1: not supported (final run, 3 models)

The backbone is **`google/gemma-4-E2B-it`**. Everything is in `h1_report/working/factindic_h1/`
(594 gradeable summaries); the executed notebook is `notebooks/00-h1-gate.ipynb` and stays as it is.

| Gate test (β on tokens per word) | All models | Without Llama |
|---|---|---|
| Within model | −0.022, one-sided p = 0.956 | +0.106, p = 0.029 |
| Across models | +0.018, p = 0.072 | +0.066, p = 0.27 |
| Within model, script held fixed | −0.020, p = 0.87 | −0.107, p = 0.82 |

- **Counterexample:** Llama-3.2 on Gujarati drops from 9.8 to 2.5 tokens per word when Romanized, while
  its hallucination rate rises from 0.49 to 0.77. Gemma-4 on Gujarati does the same (2.32 → 2.21, 0.27 → 0.52).
- **Why Gemma-4:** Sarvam-1-SFT hallucinates 18 points more (95% CI 10 to 26). Gemma-4 also wins under
  native NLI (11 points) and on unsupported names and numbers (9 points). Its ROUGE-L is lower.
- **Limits:** the result is a failure to find the effect, on 100 articles. MiniCheck and the native NLI
  verifier agree poorly (κ = 0.15) and there are no human labels, so "Romanization raises hallucination" is not settled.
  Claims were translated with NLLB-200, not IndicTrans2. The Romanized inputs had the spelling defects listed below
  (fixed for 01–07), which also counts against reading Romanized H1 scores as pure script effects. 20 of 100 Llama Gujarati summaries hit the
  192-token cap; leaving long outputs out does not change the verdict.

The first two rows are the gate's own tests. The rest is a sensitivity analysis done afterwards;
notebook 06 recomputes it (`h1_sensitivity`, `h1_shift_by_measure`).

## Pipeline

```
Stage 1  normalize + anchors      indic_nlp normalizer · word-level LID + CMI · Gujarati enclitic trim ·
                                  A-5 normalizer (digits, lakh/crore, dates, cross-script entity keys) ·
                                  IndicNER (offset-safe) → A_core(D) invariant across variants
Stage 2  structured prompting     Gemma-4: Cond A standard · B1 +anchor checklist · B2 +2-step CoD ·
                                  B3 +matrix-verb hint · faithfulness-weighted LoRA-SFT (w_i ∈ [0.3,1])
Stage 3  contrastive decoding     batched dual-KV-cache loop: anchor-aware dynamic CAD + DoLa
                                  (JSD premature layer, η=0.1 plausibility mask) · K=5 candidates
Stage 4  verify + rerank          IndicTrans2 (own environment, digits normalized) → round-trip anchor check →
                                  MiniCheck-FT5 | native mDeBERTa-XNLI → J(S_k) rerank → 1 prune pass
Eval     non-circular             held-out FactCG-DeBERTa-L (no human annotation)
```

## Repository

| Path | Contents |
|---|---|
| `factindic/` | The package: config, data, translit, lid, normalizer, ner, prompts, llm, sft, decoding, lookback, cove, mt, claims, verify, rerank, metrics, stats, viz, parallel, preflight, it2_env + it2_server (IndicTrans2) |
| `factindic/pipeline/` | Stage drivers: `stage1`, `stage2`, `stage3`, `stage4`, `evaluate`, `analysis`, `detector`, `auto` |
| `notebooks/` | `00-h1-gate` (done) and `FactIndic_run_all` (01–06, all source code visible; needs the `factindic_inputs.zip` dataset) |
| `run_local.py`, `setup_local.*`, `requirements-local.txt` | Laptop run |
| `inputs/notebook01_v2.zip` | Output of the finished notebook 01 run |
| `tools/build_notebooks.py` | Rewrites the notebooks after a code change: `python tools/build_notebooks.py` |
| `src/stage0_data_and_h1_gate.py` | Script form of the H1 notebook |
| `h1_report/` | H1 results |
| `results/final_run/` | **The final run**: `REPORT.md`, tables, figures, all summaries, per-item grades (`tools/collect_results.py` builds it; models, data and logs stay local) |
| `results/factindic/` | Unzipped outputs of notebooks 01–07 (not in git). Keep them out of `factindic/`, which is code only |
| `factindic/requirements.txt` | Pinned main environment (notebooks 01–07) |
| `factindic/requirements-it2.txt` | Locked IndicTrans2 environment |

## How to run

Each notebook's output is the next notebook's input. Locally, unzip `factindic_outputs.zip` into `results/`. On Kaggle, use *Add Input → Your Work →
previous notebook*; the setup cell merges the attached folder into `/kaggle/working/factindic`
automatically. On Colab, everything lives in `MyDrive/factindic`.

**`notebooks/FactIndic_run_all.ipynb` is the whole project in one notebook** (01–06) with *Save Version → Save & Run All*. Attach
`factindic_inputs.zip` as a dataset: its `pool_seed.txt` holds the training-pool ids and faithfulness scores of the finished notebook 01 run. It uses 60 test articles per language (3 variants each), 24 dev and
40 natural code-mixed articles, 600 SFT pairs per language, 4,000 seq2seq training pairs and 40 decoding-ablation articles per
language (`CFG` in `config.py`). `factindic/pipeline/auto.py` schedules the jobs on both GPUs, tests everything on 4 articles
first, retries failures, and stops by itself.

| Notebook | What it does | T4 time |
|---|---|---|
| 01 data_stage1 | 5-rule ILSUM sanitation; 300-article testbed in 3 variants; 100 dev articles; natural-CM subset; anchors; mFACT weights | ~2–2.5 h (2 h 15 measured); ~30–40 min (estimate) when the version 2 output is attached |
| 02 seq2seq_baselines | IndicBARTSS (Exp 1a) and mT5-base (Exp 1b), fine-tuned, beam 4 | ~2–2.5 h |
| 03 stage2_prompting_sft | Cond A/B1/B2/B3; weighted LoRA-SFT; dev selection of the Stage-3 base; test runs; Llama (Exp 2c); latency | ~4–8.5 h |
| 04 stage3_decoding | α/β tuning; Exp 4a–4d; K=5 candidates; CoVe (Exp 5b); perplexity; lookback features | ~3.5–6.5 h |
| 05 stage4_verify_rerank | Router, reranking, pruning, drop-only (Exp 5a), FactIndic (Exp 6) | ~1.5–2 h |
| 06 evaluate | FactCG evaluation, all metrics, H2/H4/pipeline tests and figures | ~1.5–2.5 h |
| 07 results | Lookback detector, final `eval/results.json` | ~10 min |

**Environment.** One pinned stack for every notebook: transformers 5.24.0 and huggingface_hub 1.33.0, the
versions the H1 run proved. torch, numpy, pandas and pyarrow come from the Kaggle/Colab image.

**IndicTrans2** can't run on transformers 5: its toolkit and tokenizer import `transformers.tokenization_utils`,
its config imports `transformers.onnx` (both removed), and its model uses the old tuple cache. It therefore
runs as a helper process in its own Python 3.11 venv with transformers 4.53.2 (the version its authors pin).
The venv is built once per session in `/tmp`. If it fails, the notebook stops; it never switches to NLLB
silently. To choose NLLB-200 for the whole project, set `CFG["mt_backend"] = "nllb"`.

**Preflight.** The third code cell of every notebook checks versions, the data version of notebook 01, Hugging Face
access and translation, and stops in the first minutes if something is missing.

**Hugging Face access.** Add the `HF_TOKEN` secret. Accept the terms for `ai4bharat/indictrans2-indic-en-dist-200M`
(gated, required), `ai4bharat/IndicNER` (gated, required) and `meta-llama/Llama-3.2-3B-Instruct` (gated, optional).

**Budget.** The final run took 3 h 25 min on the RTX 5050 laptop. A Kaggle T4 x2 run took 3.6 h without seven optional jobs.
The times in the table are earlier estimates for larger samples. Every job appends
JSONL and skips finished items, so you can split any notebook across sessions.

## Hypotheses → experiments → figures

| | Test | Figures (`factindic/figures/`, PNG + SVG + CSV) |
|---|---|---|
| **H1** fertility ↔ hallucination | Article fixed effects, separated into within-model (script changes) and across-model (family changes) | `h1_fertility`, `h1_shift`, `h1_coefficients`, `h1_sensitivity`, `h1_shift_by_measure` |
| **H2** precision drop across modalities | Same articles and facts in native, script-mixed and Romanized form; paired bootstrap and FE-OLS; Gujarati vs Hindi | `h2_precision_by_variant`, `h2_anchor_errors` |
| **H3** translate-then-verify vs router | **Planned, not evaluated** (needs human labels) | none |
| **H4** anchor-aware CAD | Numeric and entity anchor F1 plus perplexity vs static CAD, AdaCAD, DoLa and greedy (paired bootstrap) | `h4_numeric_f1`, `h4_entity_f1`, `h4_perplexity` |
| **Pipeline** | Stage-by-stage F1 ladder; precision–recall (drop-only collapse vs reranking); latency; router route share | `pipeline_ladder_*`, `pipeline_precision_recall`, `pipeline_latency`, `stage4_route_share` |
| **Data** | Sanitation vs the published audit; faithfulness-score distribution; CMI per modality | `data_sanitation_*`, `data_faithfulness_scores`, `data_cmi` |

The mandatory metrics are all reported per system × language × variant in `eval/main_table.csv`:
- P/R/F1 at claim level, normalized-anchor level and n-gram level (ROUGE)
- Average length in words and tokens
- Compression ratio
- Inference time

## Where the build departs from the spec

The spec (§13) allows these changes.

**Backbone.** Gemma-4 replaces Sarvam-1 because of the H1 gate result.

**Tools that won't install.** IndicLID and IndicXlit need fairseq, which doesn't install on
current Kaggle or Colab Python. The replacements are:
- **Word-level LID:** script detection plus an English frequency lexicon (`wordfreq`).
- **Romanization:** rule-based ITRANS with schwa deletion.
- **Back-transliteration:** IndicXlit if it imports, otherwise the most frequent native spelling of that Roman
  form in the training pool (about 92% of Hindi and 90% of Gujarati words restored on 120 testbed articles), then reverse ITRANS.
  Testbed and vocabulary use the same Romanizer, so real user spellings will score lower.

**Script-mixed variant.** It swaps about 80 common nouns per language, never inside a named entity, for English and keeps
Gujarati case endings as separate words (`pollution નું`). The variants are built by rules and were not
rated by native speakers.

**Claims.** Claim units are summary sentences, because MiniCheck and FactCG are trained at
sentence level. Qwen molecular decomposition is available in `claims.MolecularDecomposer`.

**Candidates.** K=5 means one Anchor-CAD+DoLa candidate plus four nucleus samples
(T = 0.7, top-p = 0.9), as §13 suggests. α_base, α_anchor and β are tuned over a 4-point grid
inside the spec's ranges.

**Stage-2 dev selection.** The dev set is graded with the IndicTrans2 → MiniCheck router. This grade only
*tunes* the pipeline. The *evaluation* grader is always FactCG.

**B2 can't be the Stage-3 base.** Chain of Density takes two passes, so it is evaluated as a
Stage-2 condition but can't feed the single-pass decoding loop.

**Lookback Lens.** It is trained at sentence level, not on 8-token windows. Labels come from
FactCG; the train/test split is by article.

**Reduced sizes (to fit the time and GPU budget; `CFG` in `config.py`).**
- Testbed: 60 articles per language (120 total, × 3 variants = 360 items per system), out of 150 prepared.
- Decoding comparison (H4): 40 articles per language (240 items). CoVe: 60 items.
- Dev set: 24 articles per language, out of 50 prepared. Decoding parameters are tuned on 10 per language.
- Natural code-mixed set: 40 articles per language, out of 77 Hindi and 100 Gujarati prepared.
- Faithfulness scores: computed on 2,000 of the 4,000 pool pairs per language.
- Gemma LoRA-SFT: 600 weighted pairs per language, one epoch.
- Seq2seq baselines: 4,000 pool pairs in total (both languages), two epochs.
- Latency: 15 items, batch size 1.
- Confidence intervals are roughly twice as wide as with the 250 test articles per language first planned.

**Seeds.** Greedy decoding is deterministic. The seeds (42, 108, 2026) apply to sampling and to
the bootstrap.

**Spec erratum.** The Gujarati Unicode block sits **0x180** above Devanagari, not +0x0280.

## Bugs found and fixed after the first run of notebook 01 (data version 2)

- **Romanization.** Letters such as ड़, ढ़ became `.d` (136 of 150 Hindi articles) and ऑ/ૉ stayed in native script
  (230 of 300 articles). Both are fixed, and medial schwas are dropped (`sabase` → `sabse`). H1 used the old Romanizer.
- **Prefix copies.** 43% of the Hindi ILSUM pairs in the pool copied the article's opening after a dateline
  (`नई दिल्लीः …`), which the rule missed. It now allows up to six words before the copy.
- **Translation.** 2.8% of IndicTrans2 outputs ended in a decoder loop (`"" "" ""`); they are cut where the loop starts.
  About 2% kept Indic fragments because `છે.પાણી` was read as a web address; sentences are now split there.
- **Sentence splitting.** No longer breaks after `डॉ.`, `રૂ.` or initials.
- **Testbed.** 8 of 300 articles were H1 pilot articles; the pilot ids now ship with the package.
- **Script mixing** replaced words inside names (`સુરક્ષા દળ`) and used `protest` for પ્રદર્શન (demonstration, show).
- **Language ID** read 18% of Romanized Hindi words as English; it now uses a Roman vocabulary from the training pool.
- **Natural code-mixed set** had 36 + 42 articles, because ~1% qualify; ILSUM train articles outside the pool are added.

## Bugs found and fixed after the second run of notebook 01 (data version 3)

- **Page code in ILSUM.** 90 of 100 Hindi natural code-mixed articles contained video-player or gallery JavaScript, which
  the CMI counted as English; so did 53 of 2,000 Hindi pool articles, and 239 Gujarati ones kept links or HTML entities.
  `strip_web` removes it when ILSUM is loaded.
- **Natural code-mixed set.** English captions, tweets and headlines (more than 5 Latin words in a row) no longer count as
  mixing, the sanitation rules apply (61 Hindi references repeated the article's opening), and the English terms are the
  Latin words in the text. With page code gone, 1 of 2,000 Hindi pool articles reaches the spec's CMI ≥ 8, so each language
  uses the highest of 8, 6, 5, 4, 3 that leaves 100 articles. **The achieved CMI must be reported in the paper.**
- **Romanization** of hyphenated words and initials (`satha-sath` → `sath-sath`, `esa.pi.` → `es.pi.`): 188 of 400 articles.

## Bugs found and fixed after the H1 runs

- **IndicNER entity text.** IndicNER returns entity text with the vowel signs stripped. Entities
  are now cut from the original text by character offsets. In the final run all 946 entities match the text.
- **The H1 gate rule.** It now requires both the within-model and the across-model test to pass.
  The pooled regression confounds script with fertility.
- **Llama token cap.** The pipeline cap of 160 tokens would have cut 44 of 100 Llama Gujarati summaries. Llama now gets 320.
- **Silent half precision.** transformers 5 loads each checkpoint in its saved dtype. Verifiers, BERTScore and the
  seq2seq baselines now load in fp32 explicitly, and each verifier checks itself on a known example.
