# Kaggle run (partial replication)

A run of the same experiment on Kaggle (2 × T4, Gemma-4 in 4-bit) with older code, before the fixes made on the laptop. The main result is
`results/final_run` (laptop). This folder holds per-item grades and summaries only, so that the replication paragraph of the paper can be checked
(`python paper/extra_numbers.py`).

Known faults, which is why it is not the main result:
- The development score computed anchor recall without entities (nearly zero everywhere), so the choice of B3 as base prompt rests on almost nothing.
  On the laptop, with the fault fixed, B1 was chosen. FactIndic and the decoders therefore build on B3 here.
- IndicBARTSS training had a NaN loss at every logged step; its outputs are kept but not used.
- Seven optional jobs were dropped for time: Llama, mT5, B2, CoVe, perplexity, Lookback, single-item latency.
- Whether the attention-kernel fault found on the laptop GPU also occurs on this hardware was not tested.
