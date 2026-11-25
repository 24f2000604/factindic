# FactIndic: results of the full run (H2, H4 and pipeline)

Run 25 Nov 2026 21:51 – 26 Nov 01:15 on one RTX 5050 laptop (8 GB), 3 h 25 min; all 45 jobs + the final analysis ok, none dropped.
**H1 is not part of this folder**: it was finished earlier and not rerun (`h1_report/`).

## Setup
- 60 test articles per language (Hindi, Gujarati) × 3 input variants (native, script-mixed, Romanized) = 360 items, plus 40 natural
  code-mixed articles per language = 440 generations per system. Decoding ablation: 40 articles per language (240 items). CoVe: 60 items.
- Backbone Gemma-4 E2B (bf16). Grader: FactCG-DeBERTa (held out). Claim = summary sentence. CIs: 1,000-sample bootstrap.
- Dev selection picked prompt **B1** and α_base 0.2, α_anchor 0.6, β 0.2. The grid had 4 points and the dev set is small.
- p = 0.000 means < 0.002. No multiple-comparison correction: treat single p-values near 0.05 with care.

## Verdicts
| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H2** precision drop | **Partly supported** | Romanized input: strong drop. Script-mixed: no drop. "Gujarati has higher baseline error": not supported |
| H3 router vs human labels | **Future work** | Planned, but it needs human claim labels, which this project does not have. No claim is made about it |
| **H4** anchor-aware CAD | **Not supported** | No gain in entity F1 or precision; numeric F1 worse on 14 pairs; perplexity not better |
| **Full pipeline** | **Mostly null** | Better than seq2seq baselines and than Cond A on anchors; not better than the simple B1 prompt |

### H2: precision by input variant (paired, same articles)
FE regression on claim precision, native as reference (n = 360): Romanized β = −0.343 (A), −0.308 (B1), −0.294 (FactIndic), all p < 0.001;
script-mixed β = +0.022, +0.005, +0.063 (p = 0.48, 0.89, 0.11).

| Change native → Romanized | Hindi | Gujarati |
|---|---|---|
| Gemma A | −0.186 [−0.319, −0.047] | −0.500 [−0.633, −0.342] |
| Gemma B1 | −0.185 [−0.308, −0.079] | −0.431 [−0.558, −0.283] |
| FactIndic | −0.208 [−0.342, −0.083] | −0.381 [−0.514, −0.233] |

- Romanization hurts Gujarati more in size; the difference between languages was not tested formally.
- Gujarati vs Hindi on native input: not significant (β −0.094, p = 0.12 for B1; −0.100, p = 0.13 for FactIndic), and unsupported-entity
  rates pooled over the H2 systems are Hindi 2.8 % vs Gujarati 0.9 %, the opposite of the hypothesis.
- The script-mixed inputs are mild: mean CMI 3.3 (Hindi) and 3.1 (Gujarati).
- IndicBARTSS, mT5 and Gemma-SFT answer Romanized input in Latin script (120 of 120 each, 0 of 120 for B1 and FactIndic; Llama 10 of 120),
  so their Romanized precision measures a script failure plus a grader that reads Latin-script text. Their H2 rows are not comparable.

### H4: Anchor-CAD + DoLa against the other decoders (paired difference, 240 items)
| vs | Entity F1 | Numeric F1 (n = 14) | Claim precision | Perplexity |
|---|---|---|---|---|
| Static CAD | +0.009 [−0.017, 0.037] | −0.190 [−0.357, −0.048] | +0.014 [−0.031, 0.062] | +0.011 [−0.004, 0.028] |
| AdaCAD | +0.011 [−0.015, 0.040] | −0.262 [−0.476, −0.071] | +0.029 [−0.017, 0.077] | +0.030 [0.016, 0.045] |
| Greedy | +0.013 [−0.017, 0.045] | +0.071 [−0.143, 0.286] | −0.006 [−0.052, 0.042] | +0.096 [0.080, 0.112] |

Perplexity is measured under the plain model, which favours greedy decoding by construction; the AdaCAD difference is small. Entity F1 of all
five decoders lies between 0.465 and 0.476.

## All systems
Testbed items, all 3 variants, FactCG precision. **The decoding rows (240 items) and CoVe (60 items) are different, smaller samples and must not
be ranked against the 360-item rows.** "No Roman" leaves out the Romanized variant.

| System | Precision | No Roman | Anchor P | Anchor R | ROUGE-L | Words | Empty |
|---|---|---|---|---|---|---|---|
| mT5-base | 0.467 | 0.650 | 0.948 | 0.254 | 10.7 | 19 | 0 |
| IndicBARTSS | 0.512 | 0.720 | 0.958 | 0.309 | 11.0 | 27 | 0 |
| Llama-3.2-3B (A) | 0.589 | 0.735 | 0.966 | 0.407 | 15.6 | 20 | 0 |
| Gemma LoRA-SFT | 0.656 | 0.930 | 0.963 | 0.327 | 10.6 | 25 | 0 |
| Gemma B2 (CoD) | 0.704 | 0.805 | 0.985 | 0.471 | 14.7 | 30 | 0 |
| Anchor-CAD + DoLa | 0.725 | 0.830 | 0.991 | 0.370 | 14.9 | 23 | 0 |
| Gemma A | 0.728 | 0.845 | 0.962 | 0.336 | 14.6 | 24 | 0 |
| Gemma B3 | 0.737 | 0.817 | 0.990 | 0.368 | 14.9 | 22 | 0 |
| Gemma B1 | 0.752 | 0.855 | 0.994 | 0.373 | 14.9 | 23 | 0 |
| **FactIndic (full)** | **0.756** | 0.865 | 0.989 | 0.391 | 14.8 | 23 | 0 |
| Drop-only | 0.799 | 0.897 | 0.990 | 0.291 | 12.7 | 18 | 14.2 % |
| *240 items:* static CAD / AdaCAD / DoLa / greedy | 0.741 / 0.726 / 0.766 / 0.761 | | | | | | |
| *60 items:* CoVe | 0.790 | | | | | | |

- **SFT:** lowest of the Gemma systems overall because it copies the input script (all 120 Romanized test summaries are in Latin script). Without Romanized input its
  precision (0.930) is the highest here, but ROUGE-L is the lowest of the Gemma systems: it states less.
  This subset comparison was made after seeing the results.
- The pairwise comparisons below are the pre-specified tests on **all 440 items** (test bed plus natural code-mixed), not only the 360 in the table. On the 360
  test-bed items alone: anchor F1 FactIndic − A = +0.043 [0.014, 0.076], B1 − A = +0.022 [−0.009, 0.055] (not significant), and precision differences stay not significant.
- **FactIndic vs Cond A:** precision +0.024 [−0.010, 0.058], not significant; anchor F1 +0.055 [0.027, 0.084], p < 0.002.
- **FactIndic vs B1:** precision +0.008 [−0.024, 0.042], anchor F1 +0.015 [−0.009, 0.040]: no detectable gain from Stages 3–4.
- **B1 vs A:** anchor F1 +0.034 [0.006, 0.062]; precision +0.015 [−0.022, 0.052].
- **Drop-only vs FactIndic:** FactIndic precision −0.019 [−0.053, 0.012] (not significant), anchor F1 +0.059 [0.035, 0.084]; drop-only leaves 14 % of summaries empty. Empty summaries have no precision, so its 0.799 covers only the non-empty ones and is optimistic.
- **FactIndic vs IndicBARTSS:** precision +0.209 [0.158, 0.264], inflated by the script failure on Romanized input.
- **Cost:** 4.0 s per summary (generation + verification) vs 0.3–0.4 s for prompting, batch mode. Single-item latency: A 1.14 s, B1 1.20 s (15 items).
- **Stage 4:** candidate 0 (Anchor-CAD) chosen in 33 %; 8 % pruned; 66–72 % of claims go to native NLI, so the router rarely trusts the translation.
- **Lookback detector:** AUROC 0.74 (330 test sentences) vs 0.61 for the mean lookback ratio; labels are FactCG verdicts, so it predicts the grader, not human truth.

## Problems that limit the conclusions
1. **Recall and F1 (claim level) are nearly uninformative.** Only 135 of 360 testbed items have a reference claim that the grader supports (61 % of reference claims overall),
   and recall is 0–0.13 (0.17 for CoVe's 60 items). F1 in `main_table.csv` is also **biased downward**: an item with undefined recall is dropped unless its
   precision is 0, then it counts as 0. Recomputed on items with defined recall, e.g. FactIndic − A = +0.008 [−0.017, 0.033],
   FactIndic − B1 = −0.010 [−0.037, 0.019]: the conclusion (no difference) is unchanged. The code was left as it is.
   **Use precision and anchor metrics; do not rank by F1.**
2. Numeric-anchor F1 in H4 rests on 14 paired items.
3. Natural code-mixed CMI averages 4.5 (Hindi) and 6.3 (Gujarati) over the prepared sets, spec ≥ 8; only 77 Hindi articles qualified (CMI ≥ 3). The run used 40 per language.
4. Sizes are reduced from the spec (`n_use`: 60 / 24 / 40 per language), so moderate effects cannot be excluded.
5. Anchor precision is near ceiling (0.95–0.99), so it separates systems poorly.
6. Grading goes through machine translation (IndicTrans2); only FactCG is held out from training and reranking, and it has no human validation here.
7. Precision: Gemma in bf16 here vs 4-bit in H1; H1 used the old Romanizer. H1 and these results are not directly comparable.

## What was changed to make it run
Full diff: `1_reports/code_changes.diff`. Found while running:
- PyTorch's memory-efficient attention kernel gave wrong Gemma-4 logits at sequence lengths 32k + 1 on this GPU (errors of 10–30 in logits). It is now
  disabled; a first run and its SFT adapter were discarded and are not in this folder.
- `dev_grade` scored anchor recall without entities; a fixed order for anchor ties; analysis (`empty` column) and Lookback (shard files) crashes or silent skips;
  IndicTrans2 out-of-memory retry; protobuf 7.36.1 for the mT5/IndicBARTSS tokenizers; the mDeBERTa self-test sentence.
- IndicBARTSS output had `</s>` filler after the summary. It was cut at the first `</s>` before evaluation (raw text kept in `runs/originals`).
- IndicBARTSS and mT5 weights are kept. Models were trained once, seed 42.

## Folder
| Folder | Content |
|---|---|
| `1_reports/` | `results.json` (tests), `main_table.csv`, dev scores, chosen decoding settings, environment and model revisions |
| `2_figures/` | png, svg, csv |
| `3_summaries/` | every system's summaries, one file per system |
| `4_evaluation/` | per-item grades, references, Stage-4 verdicts |
| `5_models/` | Gemma LoRA adapter, IndicBARTSS, mT5 (not in git: 3.2 GB) |
| `6_data/`, `7_logs/` | test, dev and code-mixed sets, faithfulness weights, logs (not in git) |
| `CHECKSUMS.md5` | verify with `md5sum -c CHECKSUMS.md5` |

The base models (Gemma-4, Llama, MiniCheck, FactCG, IndicTrans2, …) stay in the Hugging Face cache; their revisions are in `1_reports/environment.json`.
