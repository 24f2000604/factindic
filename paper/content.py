"""Text of the paper. Every result is taken from numbers.json. Markup: *italic*, **bold**, _{sub}, ^{sup}."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "numbers.json").read_text())
S, T, D, H1 = N["systems"], N["tests"], N["decoding"], N["h1"]["sens"]
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"


def sg(x, d=3):
    return f"{x:+.{d}f}".replace("-", "−")


def ci(t, d=3):
    return f"{sg(t['diff'], d)} [{sg(t['lo'], d)}, {sg(t['hi'], d)}]"


def pv(p):
    return "p < 0.002" if p < 0.002 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def reg(t):
    return f"β = {sg(t['beta'])}, {'p < 0.001' if t['p_two'] < 0.001 else 'p = %.2f' % t['p_two']}"


P = N["pipeline"]                      # pairwise tests of the pre-specified analysis: all 440 items
FA, FB = P["factindic−gemma_A"], P["factindic−gemma_B1"]
BA = P["gemma_B1−gemma_A (factuality-aware vs standard prompt)"]
FD = P["factindic−drop_only"]
H2 = N["h2"]
H4 = N["h4"]
san = {r["lang"]: r for r in N["sanitation"]}
W = N["weights"]
sh = N["stage4"]["selected_k_share"]
nli = N["stage4"]["nli_route_share"]
vr = N["variants"]
lat1 = N["latency_single"]
assert N["latin_out"]["gemma_B1"] == 0 and N["latin_out"]["factindic"] == 0 and N["selected"]["system"] == "gemma_B1"

TITLE = "FactIndic: A Zero-Cost Pipeline for Factuality-Aware Summarization of Hindi, Gujarati and Code-Mixed Text, with Mostly Negative Results"
AUTHORS = ["[Author name(s)]", "[Department, institution]", "[City, country]", "[e-mail]"]
ABSTRACT = (
    "Small open language models can summarize Hindi and Gujarati news fluently, but they also state facts the article does not support. "
    "We build FactIndic, a four-stage pipeline (anchor extraction, anchor-aware prompting, context-contrastive decoding, and verification with "
    "reranking) that runs on one 8 GB laptop GPU with open models only, and test it on native-script, script-mixed and Romanized versions of the "
    f"same {S['gemma_A']['n'] // 3} articles. A held-out checker (FactCG) grades every summary. Romanized input lowers claim precision sharply, by "
    f"{abs(H2['gemma_B1']['hi']['native→roman']['diff']):.2f} for Hindi and {abs(H2['gemma_B1']['gu']['native→roman']['diff']):.2f} for Gujarati with our best prompt, "
    "while replacing nouns with English words has no detectable effect. Tokenizer fertility does not explain hallucination, and anchor-aware contrastive "
    "decoding does not improve on static or adaptive context-aware decoding. The full pipeline reaches a precision of "
    f"{f3(S['factindic']['P']['mean'])} against {f3(S['gemma_A']['P']['mean'])} for a standard prompt and {f3(S['gemma_B1']['P']['mean'])} for a prompt with an anchor checklist; "
    "neither difference is significant, and the pipeline costs about ten times more time. It does improve normalized anchor F1 over the standard prompt. "
    "We report these negative results in full, together with three engineering faults that could have distorted the conclusions had they gone unnoticed.")
INDEX = "abstractive summarization, factual consistency, hallucination, Hindi, Gujarati, code-mixing, contrastive decoding, negative results"

B = []                                  # blocks: ("h1", text) ("h2", text) ("p", text) ("eq", text) ("fig", …) ("table", …) ("li", text)
h1 = lambda t: B.append(("h1", t))
h2 = lambda t: B.append(("h2", t))
p = lambda t: B.append(("p", t))
pc = lambda t: B.append(("pc", t))         # continues after an equation: no indent
eq = lambda t: B.append(("eq", t))
li = lambda t: B.append(("li", t))

h1("Introduction")
p("Automatic summarization is useful for Indian-language news only if the summary is true to the article. Neural summarizers are trained to be fluent, "
  "and hallucinate facts that the source does not contain [1]. The problem is harder for Hindi and Gujarati in practice: many readers write them in Latin "
  "script or mix them with English, the tools that check facts are built for English, and the public training sets contain corrupt references.")
p("This paper describes FactIndic, a four-stage pipeline for Hindi, Gujarati and their code-mixed forms. It uses open models only and costs nothing to run: "
  "all experiments ran on a free cloud GPU or on one laptop GPU with 8 GB of memory. We ask whether each stage earns its cost, under a grader that the "
  "pipeline never sees. Four hypotheses were written down before the experiments. Three are tested here:")
li("**H1:** higher tokenizer fertility (tokens per word) goes with more hallucination.")
li("**H2:** with the facts held constant, claim precision drops from native-script to script-mixed and Romanized input, and Gujarati has a higher baseline error rate than Hindi.")
li("**H4:** modulating context-aware decoding by anchor tokens improves numeric and entity F1 over static and adaptive decoding without raising perplexity.")
p("The fourth, H3, concerns agreement between automatic verifiers and human labels. It needs human annotation, which this study does not have, so it is left to future work (Section VIII) and no claim is made about it. "
  "H1 and H4 are not supported. H2 is supported for Romanized input only. The full pipeline is not measurably more precise than a prompt that lists the "
  "article's names, numbers and dates. Our contributions are (i) a parallel test set in three input forms with the same facts, (ii) a controlled comparison of "
  "prompting, decoding and verification stages with a held-out grader, (iii) negative results with confidence intervals, and (iv) an account of "
  "silent faults, including a GPU kernel error, that such experiments should test for.")

h1("Related Work")
p("*Data and models.* XL-Sum [2] provides BBC articles and summaries in 44 languages, and ILSUM [3] provides Hindi and Gujarati news. IndicBART [4] and mT5 [5] are "
  "the usual sequence-to-sequence baselines; IndicGenBench [6] evaluates large models on Indic generation. RomanSetu [7] shows that Romanized text can reduce token counts "
  "and serve as a bridge for large models; multilingual models may also work through an English-like latent space [8].")
p("*Factual consistency.* Maynez et al. [1] document hallucinations in summarization, including in reference summaries. For many languages, Aharoni et al. [9] filter "
  "training data by entailment, and Qiu et al. [10] weight the loss by a faithfulness score, the approach of our fine-tuned baseline. MiniCheck [11] and FactCG [12] are compact English "
  "checkers that decide whether a document supports a claim. We use the first inside the pipeline and keep the second for evaluation only.")
p("*Interventions.* Chain of Density [13] makes summaries denser in entities. Chain-of-Verification [14] lets a model check its own draft. Context-aware decoding "
  "(CAD) [15] contrasts next-token distributions with and without the document; AdaCAD [16] sets the contrast strength from their divergence. DoLa [17] contrasts late and early layers. "
  "Lookback Lens [18] detects contextual hallucinations from attention maps. FactIndic combines a checklist prompt, an anchor-dependent CAD weight with DoLa, and external verification.")

h1("Method")
p("Fig. 1 shows the pipeline. The input *D* is a Hindi or Gujarati article in native script, in script-mixed form (content nouns in English) or Romanized. The output is always written in the matrix language in its native script.")
B.append(("fig", "pipeline.png", "The FactIndic pipeline. Stage 2 uses the prompt that scored best on the development set.", True))
h2("Anchors")
p("Text is normalized with the Indic NLP library. Words are labelled by script, English word frequency and a Roman-form vocabulary from the training pool. Named entities come from IndicNER [19] in native script and a multilingual tagger in Latin script. "
  "Numbers, dates and entities are mapped to canonical keys: Devanagari and Gujarati digits to ASCII, *lakh* and *crore* to values, and entity names to a vowel-reduced Latin key that is compared with edit distance 1, so that a name matches across scripts. "
  "These keys form the anchor set *A*(*D*). Romanized text is converted back to native script for verification, with the most frequent native spelling of each Roman form in the training pool. "
  "IndicLID and IndicXlit, named in our plan, do not install on current Python; the rules above replace them.")
h2("Prompting and fine-tuning")
p("All conditions ask Gemma-4 E2B [20] for a summary of a given length in native script. *A* is a standard prompt. *B1* adds factuality rules and the anchor list. "
  "*B2* adds a second pass that fuses missing anchors, in the manner of Chain of Density. *B3* adds a hint about the matrix-language verb. We also fine-tune Gemma with LoRA [21] "
  "(rank 16, α = 32, 1,200 pairs, half per language, one epoch), weighting each example by a faithfulness score *w* = clip(*s*, 0.3, 1), where *s* is the mean MiniCheck support of the translated reference; pairs with *s* < 0.3 are discarded.")
h2("Contrastive decoding")
p("Let *P*_{L} be the next-token distribution given the document and anchors, *P*_{0} that of a prompt without them, and *P*_{l*} that of the early layer, among four candidates, with the largest Jensen–Shannon divergence from the last layer. Each plausible token *w*, that is *P*_{L}(*w*) ≥ 0.1 max *P*_{L}, is scored by")
eq("*s*(*w*) = (1 + α_{t}) log *P*_{L}(*w*) − α_{t} log *P*_{0}(*w*) − β log *P*_{l*}(*w*),")
pc(f"where α_{{t}} = α_{{anchor}} if the most likely token belongs to a source anchor, and α_{{base}} otherwise. A four-point grid on development data chose α_{{base}} = {N['params']['alpha_base']}, α_{{anchor}} = {N['params']['alpha_anchor']} and β = {N['params']['beta']}.")
h2("Verification and reranking")
p("Stage 4 scores *K* = 5 candidates: one from Stage 3 and four nucleus samples (T = 0.7, top-p = 0.9). Each summary sentence is a claim. It is translated with IndicTrans2 [22]. If every number and entity of the claim survives translation, MiniCheck scores it against the translated source; otherwise a native mDeBERTa NLI model [23], [24] scores it against the native source. The candidate with the highest")
eq("*J*(*S*) = *F*(*S*) · *R*(*S*, *D*) − λ · | words(*S*) − *L* | / *L*")
pc("is kept, where *F* is the share of claims with score ≥ 0.5, *R* the recall of the six most salient source anchors, *L* the target length in words and λ = 0.2. Unsupported sentences of the winner are removed once, if at least one supported sentence remains.")

h1("Experimental Setup")
h2("Data")
p(f"We remove empty, duplicate and near-extractive pairs from ILSUM-1.0 [3]. This drops {san['hi']['invalid_pct']:.1f}% of {san['hi']['raw']:,} Hindi and {san['gu']['invalid_pct']:.1f}% of {san['gu']['raw']:,} Gujarati training pairs, mostly Hindi summaries that copy the article's opening after a dateline. "
  f"The training pool has 4,000 pairs per language from ILSUM and XL-Sum [2]; 2,000 per language have faithfulness scores, of which {W['hi']['kept']:,} (Hindi) and {W['gu']['kept']:,} (Gujarati) pass the 0.3 cut.")
p("The test bed has 60 XL-Sum test articles per language, stratified by entity density and cut to 350 words. Each exists in three forms with the same facts: native, script-mixed (a list of about 80 nouns per language replaced by English, never inside a name) and Romanized (rule-based). "
  f"This gives {S['gemma_A']['n']} items. A further 40 naturally code-mixed ILSUM articles per language bring the total to 440. The development set has 24 articles per language. Articles used for H1 are excluded.")
p(f"The plan called for larger sets (250 test articles per language) and naturally mixed articles with a code-mixing index (CMI) [25] of at least 8. After web page code was removed, too few articles reached it: the used Hindi and Gujarati articles average {N['cmi_natural']['hi']:.1f} and {N['cmi_natural']['gu']:.1f}. The script-mixed test articles average {N['cmi_mixed']['hi']:.1f} and {N['cmi_mixed']['gu']:.1f}, so they are a mild form of mixing.")
h2("Systems and compute")
p("Baselines are IndicBARTSS [4] and mT5-base [5], trained on 4,000 pairs for two epochs (beam 4), Llama-3.2-3B-Instruct [26] with prompt *A*, drop-only repair (delete every sentence that fails verification) and Chain-of-Verification [14] on 60 items. The decoding comparison (greedy, static CAD, AdaCAD, DoLa and ours) uses 40 articles per language, that is 240 items. "
  "Gemma runs in bf16 with its per-layer embedding table in system memory. The whole experiment after data preparation took 3 h 25 min on an RTX 5050 laptop GPU with 8 GB.")
h2("Metrics")
p("Claim precision is the share of summary sentences that FactCG [12] finds supported by the English translation of the *native* article, whatever the input form, so every system is graded against the same facts. FactCG never weights training data and never reranks. Claim recall is the share of source-supported reference sentences that the summary supports. "
  f"It is undefined when no reference sentence is supported, which happens for {S['gemma_A']['n'] - S['gemma_A']['n_R']} of {S['gemma_A']['n']} items (only {100 * N['refs']['grounded']:.0f}% of {N['refs']['claims']} reference sentences are supported), so we report recall and F1 on the other {S['gemma_A']['n_R']} items. "
  "Anchor precision, recall and F1 compare normalized anchors with the source and with the source-supported anchors of the reference. ROUGE [27] uses Indic tokenization; BERTScore [28] uses IndicBERT v2 [29] and is rescaled.")
p("Intervals are 95% percentile bootstrap intervals with 1,000 resamples; pairwise differences are paired by item. Regressions have article fixed effects and standard errors clustered by article. A bootstrap p-value of zero is printed as p < 0.002. We do not correct for multiple comparisons, so isolated values near 0.05 should not be read as findings. Thresholds and samples were not changed after results were seen. Two things were decided afterwards: H3 was moved to future work for lack of human labels (a label-free proxy had been computed; it compares the router with itself and is not reported), and recall and F1 were restricted to items where recall is defined, because the original code counted some undefined cases as zero. The restriction changes no conclusion.")

h1("Results")
h2("H1: tokenizer fertility")
a, nl = H1["all"], H1["no_llama"]
p(f"H1 was tested first, on 100 other articles and three models (Sarvam-1 after fine-tuning, Gemma-4 and Llama-3.2) in native and Romanized form, {a['within model (gate)']['n']} gradeable summaries. The gate required a positive effect of fertility both within a model and across models. "
  f"Neither holds: β = {sg(a['within model (gate)']['beta'])} (one-sided p = {a['within model (gate)']['p_one_pos']:.2f}) within models and β = {sg(a['across models (gate)']['beta'])} (p = {a['across models (gate)']['p_one_pos']:.2f}) across them (Table I). "
  f"Without Llama the within-model effect is positive (p = {nl['within model (gate)']['p_one_pos']:.2f}) but disappears once the script is held fixed, so it reflects Romanization, not fertility. Llama on Gujarati is the clearest counterexample: Romanizing lowers its fertility from 9.8 to 2.5 tokens per word, and its hallucination rate rises from 0.49 to 0.77. "
  f"Sarvam-1 hallucinated {100 * N['h1']['contrasts']['sarvam1_sft-gemma4_e2b']['mean_diff']:.0f} points more than Gemma-4 (95% CI {100 * N['h1']['contrasts']['sarvam1_sft-gemma4_e2b']['ci95'][0]:.0f} to {100 * N['h1']['contrasts']['sarvam1_sft-gemma4_e2b']['ci95'][1]:.0f}), so Gemma-4 became the backbone, as the plan prescribed.")
rows = []
for k, lab in (("within model (gate)", "Within model (gate)"), ("across models (gate)", "Across models (gate)"), ("within model, script fixed", "Within model, script fixed")):
    rows.append([lab] + [x for s in (a, nl) for x in (sg(s[k]["beta"]), f"{s[k]['p_one_pos']:.2f}")])
B.append(("table", "H1: effect of tokens per word on the hallucination rate (p one-sided for β > 0)", ["Specification", "All: β", "p", "No Llama: β", "p"], rows, False, ""))

h2("H2: input form")
gA, gB, gF = (H2[s] for s in ("gemma_A", "gemma_B1", "factindic"))
p("Fig. 2 and Table II show precision by input form. "
  f"With native input as reference, a regression with article fixed effects gives a large loss for Romanized input ({reg(gA['regression']['roman'])} for *A*; {reg(gB['regression']['roman'])} for *B1*; {reg(gF['regression']['roman'])} for FactIndic) and none for script-mixed input "
  f"({reg(gA['regression']['script_mixed'])}; {reg(gB['regression']['script_mixed'])}; {reg(gF['regression']['script_mixed'])}). The loss is about twice as large in Gujarati as in Hindi for these systems, e.g. {ci(gB['gu']['native→roman'], 2)} against {ci(gB['hi']['native→roman'], 2)} for *B1*; we did not test this difference formally.")
e = N["h2_errors"]
p(f"The second clause of H2 is not supported. On native input Gujarati precision is not significantly different from Hindi ({reg(gB['regression']['gujarati_vs_hindi_native'])} for *B1*; {reg(gF['regression']['gujarati_vs_hindi_native'])} for FactIndic), and the share of unsupported entities among summary anchors is lower in Gujarati ({100 * e['gu']['ent']:.1f}% of {e['gu']['anchors']}) than in Hindi ({100 * e['hi']['ent']:.1f}% of {e['hi']['anchors']}).")
p(f"IndicBARTSS, mT5 and the fine-tuned Gemma answer all 120 Romanized items in Latin script (Llama: {N['latin_out']['llama_A']}; Gemma *B1* and FactIndic: 0). Their Romanized scores therefore combine a script failure with a translator that expects native script, and cannot be compared with the others.")
B.append(("fig", "h2.png", "Claim precision by input form (60 articles per language and form; bars: 95% CI). The vertical axis starts at 0.2.", False))
vh = ["System", "Lang.", "Native", "Mixed", "Roman", "Natural CM"]
rows = []
for s, lab in (("gemma_A", "Gemma A"), ("gemma_B1", "Gemma B1"), ("factindic", "FactIndic"), ("llama_A", "Llama A")):
    for l, ll in (("hi", "hi"), ("gu", "gu")):
        rows.append([lab if l == "hi" else "", ll] + [f2(vr[s][f"{l}/{v}"]["mean"]) for v in ("native", "script_mixed", "roman", "natural_cm")])
B.append(("table", "Claim precision by language and input form. Natural CM uses different articles (ILSUM)", vh, rows, False, ""))

h2("Prompting, fine-tuning and baselines")
t = T["gemma_B2-gemma_B1"]
p(f"Table III lists all systems on the {S['gemma_A']['n']} test-bed items and Fig. 3 their precision. The instruction-tuned Gemma is more precise than both fine-tuned sequence-to-sequence baselines and Llama, also without Romanized items ({f2(S['gemma_A']['P_nonroman']['mean'])} for *A* against {f2(S['mt5']['P_nonroman']['mean'])}, {f2(S['indicbartss']['P_nonroman']['mean'])} and {f2(S['llama_A']['P_nonroman']['mean'])}), while Llama has the higher anchor recall. The anchor checklist (*B1*) raises anchor F1 over the standard prompt in the pre-specified test on all 440 items ({ci(BA['F1_anchor'])}, {pv(BA['F1_anchor']['p'])}); "
  f"on the test bed alone the gain is in anchor recall ({ci(T['gemma_B1-gemma_A']['R_anchor'])}), while anchor F1 is not significant ({ci(T['gemma_B1-gemma_A']['F1_anchor'])}). Claim precision does not change significantly ({ci(BA['P_fact'])}). "
  f"The second pass of *B2* raises anchor recall ({ci(t['R_anchor'])}) but lowers precision ({ci(t['P_fact'])}, {pv(t['P_fact']['p'])}) and lengthens summaries from {S['gemma_B1']['words']:.0f} to {S['gemma_B2']['words']:.0f} words: denser summaries contain more errors. The verb hint (*B3*) does not help.")
t = T["gemma_sft-gemma_A"]
p(f"Fine-tuning lowered overall precision ({ci(t['P_fact'])}), because the adapter, trained on native text only, learned to copy the input script. Without Romanized items its precision is {f3(S['gemma_sft']['P_nonroman']['mean'])}, higher than that of *A* ({ci(t['P_fact_nonroman'])}); this subset was examined after seeing the results. Its ROUGE-L is {S['gemma_sft']['rougeL'][2]:.1f} against {S['gemma_A']['rougeL'][2]:.1f}.")
B.append(("fig", "ladder.png", f"Claim precision of the systems on the {S['gemma_A']['n']} test-bed items. The horizontal axis does not start at 0.", False))
NAMES = [("mt5", "mT5-base"), ("indicbartss", "IndicBARTSS"), ("llama_A", "Llama-3.2-3B, A"), ("gemma_A", "Gemma A"), ("gemma_B1", "Gemma B1"), ("gemma_B2", "Gemma B2"),
         ("gemma_B3", "Gemma B3"), ("gemma_sft", "Gemma LoRA-SFT"), ("dec_anchor_cad_dola", "B1 + Anchor-CAD + DoLa"), ("drop_only", "Drop-only†"), ("factindic", "FactIndic"), ("cove", "CoVe (60 items)‡")]
rows = []
for s, lab in NAMES:
    r = S[s]
    tr = lambda v: "/".join(f"{x:.0f}" for x in v)
    rows.append([lab, f3(r["P"]["mean"]), f2(r["R"]["mean"]), f2(r["F1"]["mean"]), f2(r["Pa"]["mean"]), f2(r["Ra"]["mean"]), f2(r["Fa"]["mean"]),
                 tr(r["rouge1"]), tr(r["rouge2"]), tr(r["rougeL"]), f2(r["bert"]["mean"]), f"{r['words']:.0f}", f"{r['tokens_gemma']:.0f}", f"{r['CR']:.1f}", f2(r["latency"])])
B.append(("table", "All systems on the test bed (360 items): claim-level, anchor-level and n-gram precision (P), recall (R) and F1, length, compression and time",
          ["System", "P", "R*", "F1*", "P", "R", "F1", "ROUGE-1 P/R/F", "ROUGE-2 P/R/F", "ROUGE-L P/R/F", "BERT-S", "Words", "Tok.", "CR %", "s"], rows, True,
          f"Columns 2–4: claims (FactCG); 5–7: anchors. *Recall and F1 on the {S['gemma_A']['n_R']} items with a source-supported reference claim ({S['cove']['n_R']} for CoVe). Tok.: Gemma-4 tokens. CR: words of the summary as a share of the article. s: seconds per summary, batched generation plus verification. "
          f"†{100 * S['drop_only']['empty']:.0f}% of its summaries are empty; they have no precision and are left out of P ({S['drop_only']['n_P']} items). ‡Not comparable with the other rows."))

h2("H4: anchor-aware decoding")
g = lambda ref: H4[f"anchor_cad_dola−{ref}"]
p(f"Table IV compares the decoders on the same 240 items, with greedy decoding as the B1 prompt alone. Anchor-CAD with DoLa does not improve entity F1 over static CAD ({ci(g('dec_static_cad')['F1_anchor_ent'])}) or AdaCAD ({ci(g('dec_adacad')['F1_anchor_ent'])}), nor claim precision ({ci(g('dec_static_cad')['P_fact'])} and {ci(g('dec_adacad')['P_fact'])}). "
  f"Numeric F1 is lower than with static CAD and AdaCAD ({ci(g('dec_static_cad')['F1_anchor_num'], 2)} and {ci(g('dec_adacad')['F1_anchor_num'], 2)}), but only {g('dec_static_cad')['F1_anchor_num']['n']} items have numbers on both sides, so this measure carries little weight either way. "
  f"Perplexity under the plain model is higher than with AdaCAD ({ci(g('dec_adacad')['ppl'])}) and greedy decoding ({ci(g('dec_greedy')['ppl'])}), although this measure favours greedy decoding by construction. H4 is not supported.")
rows = [[lab, f3(D[s]["P"]["mean"]), f3(D[s]["ent"]["mean"]), f"{D[s]['num']['mean']:.2f} ({D[s]['num']['n']})", f3(D[s]["ppl"]["mean"]), f2(D[s]["latency"])]
        for s, lab in (("dec_greedy", "Greedy (B1)"), ("dec_static_cad", "Static CAD"), ("dec_adacad", "AdaCAD"), ("dec_dola", "DoLa"), ("dec_anchor_cad_dola", "Anchor-CAD + DoLa"))]
B.append(("table", "Decoding methods on the same 240 items", ["Method", "P", "Ent. F1", "Num. F1 (n)", "PPL", "s"], rows, False, ""))

h2("The full pipeline")
tb = T["factindic-gemma_A"]
p(f"On all 440 items FactIndic's claim precision differs from the standard prompt by {ci(FA['P_fact'])} and from *B1* by {ci(FB['P_fact'])}; neither is significant. Anchor F1 is higher than with the standard prompt ({ci(FA['F1_anchor'])}, {pv(FA['F1_anchor']['p'])}; test bed only: {ci(tb['F1_anchor'])}) but not than with *B1* ({ci(FB['F1_anchor'])}). "
  f"Stages 3 and 4 therefore add no measurable benefit to the checklist prompt. Claim F1, on items where it is defined, shows no difference either (vs. *A*: {ci(tb['F1_fact_defined'])}).")
p(f"Deleting every rejected sentence (drop-only) has precision {f3(S['drop_only']['P']['mean'])} on the summaries that remain, but leaves {100 * S['drop_only']['empty']:.0f}% empty; FactIndic has higher anchor F1 ({ci(FD['F1_anchor'])}) at a precision that is not significantly different ({ci(FD['P_fact'])}). Stage 4 kept the Stage 3 candidate in {100 * sh['0']:.0f}% of cases and pruned {100 * N['stage4']['pruned_rate']:.0f}% of summaries. "
  f"Between {100 * min(nli.values()):.0f}% and {100 * max(nli.values()):.0f}% of claims went to the native verifier because an anchor did not survive translation, even for native input, so the router rarely trusted the translation.")
h2("Cost and detection")
p(f"FactIndic takes {S['factindic']['latency']:.1f} s per summary against {S['gemma_B1']['latency']:.2f} s for *B1* in batches; one summary at a time takes {lat1['latency_gemma_B1']['mean_s']:.1f} s with *B1* ({lat1['latency_gemma_B1']['n']} items). "
  f"A Lookback Lens detector [18] trained on Gemma's attention reaches an AUROC of {N['lookback']['auroc']:.2f} on {N['lookback']['n_test']} test sentences ({N['lookback']['auroc_mean_ratio']:.2f} for the mean ratio alone). Its labels are FactCG verdicts, so it predicts the grader rather than human judgement.")

h1("Discussion")
p("*Romanization is the problem worth solving.* It is the only large effect we find, and none of our interventions removes it. The anchors of Romanized test items came from the aligned native article, which a deployed system does not have; real Romanized input is likely to do worse.")
p("*A checklist goes a long way.* Listing the source's names, numbers and dates costs almost nothing and gives most of what the pipeline gives. Anchor precision is already 0.96 to 0.99 for every Gemma system, which leaves decoding methods that protect anchors little to correct. We did not classify the remaining errors; an anchor list cannot catch wrong relations between correct names.")
p("*Silent faults.* Three could have distorted our conclusions. PyTorch's memory-efficient attention kernel returned wrong Gemma-4 logits on our GPU at certain sequence lengths (513, 673 and 1,089 among those tested; errors of 11 to 23), so batches padded to such a length produced empty or broken summaries; we disabled it, discarded the first run and repeated it. "
  "The development score computed anchor recall without entities, which made it zero for every system and left the choice of prompt to ROUGE. A library filled finished beams with end-of-sequence markers that a tokenizer did not strip, which would have added text to one baseline's summaries. All three were found by reading outputs, not from error messages.")

h1("Limitations")
li("No human evaluation. All precision values rest on one automatic checker reading machine translations, not validated on our data; in H1 two verifiers agreed only weakly "
   f"(κ = {N['h1']['agreement']['all']['kappa']:.2f}). Nobody rated the naturalness of the script-mixed and Romanized texts.")
li("Sizes are smaller than planned (60 test, 24 development and 40 naturally mixed articles per language), so moderate effects cannot be excluded. The hyperparameter grid has four points.")
li("Claim recall is close to zero and defined for 38% of items, so claim F1 is nearly uninformative; we rank systems by precision and anchors." if round(100 * S['gemma_A']['n_R'] / S['gemma_A']['n']) == 38 else None)
li("Romanized text is rule-based and uses the same rules as the vocabulary that converts it back, so it is easier than real user spelling.")
li("H1 used 4-bit Gemma, NLLB-200 translation and an older Romanizer; the other experiments use bf16, IndicTrans2 and the corrected Romanizer. Their numbers are not directly comparable.")
li("The sampled candidates of Stage 4 and the fine-tuned models come from single runs with one seed.")

h1("Conclusion and Future Work")
p("On one laptop GPU and at no cost, a four-stage factuality pipeline for Hindi and Gujarati was not measurably more precise than a prompt with an anchor checklist. Tokenizer fertility did not explain hallucination, and anchor-aware decoding did not improve on simpler contrastive decoding. Romanized input was by far the largest source of error, about twice as costly in Gujarati as in Hindi. "
  "The practical advice for small open models is to list the source's anchors in the prompt and to treat Romanized input with care.")
p("*Future work.* The first step is to test H3: whether translate-then-verify agrees less with human claim labels on Romanized than on native input, and whether the round-trip router closes part of that gap. It needs two annotators per language; the same labels would validate FactCG on this data. "
  "The second is to work on Romanized input itself, with real user spellings and learned transliteration. The third is to measure errors beyond names and numbers.")
h1("Code and Data")
p("Code, summaries, per-item grades and the run report are at https://github.com/aaryavrdhanpandey-star/factindic; commit f4b74db is the code of the reported run. XL-Sum and ILSUM remain under their own licences.")
h1("Acknowledgment")
p("Claude Code (Anthropic) was used to write code, run the experiments and draft this manuscript. No paid model was used in the experiments. [Authors: confirm this statement and describe your own review.]")

REFS = [
    'J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, "On faithfulness and factuality in abstractive summarization," in *Proc. ACL*, 2020, pp. 1906–1919.',
    'T. Hasan et al., "XL-Sum: Large-scale multilingual abstractive summarization for 44 languages," in *Findings of ACL-IJCNLP*, 2021, pp. 4693–4703.',
    'S. Satapara, B. Modha, S. Modha, and P. Mehta, "Findings of the first shared task on Indian language summarization (ILSUM): Approaches, challenges and the path ahead," in *FIRE 2022 Working Notes*, CEUR Vol. 3395, pp. 369–382.',
    'R. Dabre et al., "IndicBART: A pre-trained model for Indic natural language generation," in *Findings of ACL*, 2022, pp. 1849–1863.',
    'L. Xue et al., "mT5: A massively multilingual pre-trained text-to-text transformer," in *Proc. NAACL*, 2021, pp. 483–498.',
    'H. Singh, N. Gupta, S. Bharadwaj, D. Tewari, and P. Talukdar, "IndicGenBench: A multilingual benchmark to evaluate generation capabilities of LLMs on Indic languages," in *Proc. ACL*, 2024, pp. 11047–11073.',
    'J. A. Husain et al., "RomanSetu: Efficiently unlocking multilingual capabilities of large language models via Romanization," in *Proc. ACL*, 2024, pp. 15593–15615.',
    'C. Wendler, V. Veselovsky, G. Monea, and R. West, "Do Llamas work in English? On the latent language of multilingual transformers," in *Proc. ACL*, 2024, pp. 15366–15394.',
    'R. Aharoni et al., "Multilingual summarization with factual consistency evaluation," in *Findings of ACL*, 2023, pp. 3562–3591.',
    'Y. Qiu, Y. Ziser, A. Korhonen, E. Ponti, and S. Cohen, "Detecting and mitigating hallucinations in multilingual summarisation," in *Proc. EMNLP*, 2023, pp. 8914–8932.',
    'L. Tang, P. Laban, and G. Durrett, "MiniCheck: Efficient fact-checking of LLMs on grounding documents," in *Proc. EMNLP*, 2024, pp. 8818–8847.',
    'D. Lei et al., "FactCG: Enhancing fact checkers with graph-based multi-hop data," in *Proc. NAACL*, 2025, pp. 5002–5020.',
    'G. Adams, A. Fabbri, F. Ladhak, E. Lehman, and N. Elhadad, "From sparse to dense: GPT-4 summarization with chain of density prompting," in *Proc. NewSum Workshop*, 2023, pp. 68–74.',
    'S. Dhuliawala et al., "Chain-of-verification reduces hallucination in large language models," in *Findings of ACL*, 2024, pp. 3563–3578.',
    'W. Shi et al., "Trusting your evidence: Hallucinate less with context-aware decoding," in *Proc. NAACL (Short Papers)*, 2024, pp. 783–791.',
    'H. Wang, A. Prasad, E. Stengel-Eskin, and M. Bansal, "AdaCAD: Adaptively decoding to balance conflicts between contextual and parametric knowledge," in *Proc. NAACL*, 2025, pp. 11636–11652.',
    'Y.-S. Chuang et al., "DoLa: Decoding by contrasting layers improves factuality in large language models," in *Proc. ICLR*, 2024.',
    'Y.-S. Chuang et al., "Lookback Lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps," in *Proc. EMNLP*, 2024, pp. 1419–1436.',
    'A. Mhaske et al., "Naamapadam: A large-scale named entity annotated data for Indic languages," in *Proc. ACL*, 2023, pp. 10441–10456.',
    'Gemma Team, "Gemma 4 technical report," arXiv:2607.02770, 2026.',
    'E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," arXiv:2106.09685, 2021.',
    'J. Gala et al., "IndicTrans2: Towards high-quality and accessible machine translation models for all 22 scheduled Indian languages," *Trans. Mach. Learn. Res.*, 2023.',
    'P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," arXiv:2111.09543, 2021.',
    'M. Laurer, W. van Atteveldt, A. Casas, and K. Welbers, "Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI," *Political Analysis*, vol. 32, no. 1, pp. 84–100, 2024.',
    'B. Gambäck and A. Das, "Comparing the level of code-switching in corpora," in *Proc. LREC*, 2016, pp. 1850–1855.',
    'A. Grattafiori et al., "The Llama 3 herd of models," arXiv:2407.21783, 2024.',
    'C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Text Summarization Branches Out*, 2004, pp. 74–81.',
    'T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in *Proc. ICLR*, 2020.',
    'S. Doddapaneni et al., "Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages," in *Proc. ACL*, 2023, pp. 12402–12426.',
]
B = [b for b in B if b[1] is not None]


def typo(t):
    """Typographic quotes and apostrophes."""
    import re
    return re.sub(r'(^|[\s(\[—])"', r"\1“", t).replace('"', "”").replace("'", "’") if isinstance(t, str) else t


B = [tuple(typo(x) if i and not str(x).endswith(".png") else x for i, x in enumerate(b)) for b in B]
REFS, ABSTRACT, TITLE = [typo(r) for r in REFS], typo(ABSTRACT), typo(TITLE)
import re  # noqa: E402
cited = {int(x) for b in B for x in re.findall(r"\[(\d+)\]", str(b))}
assert cited == set(range(1, len(REFS) + 1)), (set(range(1, len(REFS) + 1)) - cited, cited - set(range(1, len(REFS) + 1)))
