# FactIndic: instructions for Claude on the laptop

Course project (CSET 346): factuality-aware summarization for Hindi and Gujarati, spec in
`Factuality-Aware Indic Summarization Research.pdf`. **Deadline: 26 November 2026, so every hour counts.**
The user wants you to run and watch everything on this machine: Lenovo LOQ, RTX 5050, 8 GB VRAM.

## State
- H1 is finished and **not supported**. Never edit `notebooks/00-h1-gate.ipynb`
  (md5 `a384bba1ff60b9a817f7ef739b6747c6`). Its results are in `h1_report/` and `factindic/h1_results/`.
- Notebook 01 ran on Kaggle (data version 2); its output is `inputs/notebook01_v2.zip`. The code is at data version 3,
  which repairs the natural code-mixed set and needs no new NER for the testbed. `run_local.py data` does it.
- **Everything after that (seq2seq, prompting, SFT, decoding, Stage 4, evaluation) has never run on a GPU.**
  The laptop changes (`IS_LOCAL`, `_gemma_split`) have not run either. Expect first-run bugs and fix them.
- `notebooks/FactIndic_run_all.ipynb` is the same code for Kaggle (fallback and the file the teacher reads).

- The Kaggle smoke test of 25 Nov ran 22 of 27 job types cleanly. Fixed afterwards, not yet re-run: NaN logits in batched
  fp16 Gemma (now fp32 stream on T4, bf16 here, plus `llm._guard`), a too strict mDeBERTa check, mT5 memory, `d["empty"]`
  in `analysis.main_table`. H1 is not affected (batch size 1; 199 of 200 Gemma summaries end properly), but H1's Gemma
  *latency* is inflated (generation seems to have run on after the end token in 82 of 200), so do not report it.

## Run it
1. Linux or WSL2: `bash setup_local.sh`. Native Windows: `setup_local.ps1` (IndicTrans2 has no Windows wheel, so it
   uses NLLB-200: set `FACTINDIC_MT=nllb` for every command). Under WSL2 with 16 GB RAM, put `memory=12GB` and `swap=16GB`
   in `%UserProfile%\.wslconfig`, run `wsl --shutdown`, and clone inside the Linux home, not under `/mnt/c`.
   Laptop plugged in, sleep disabled.
2. `hf auth login` (the user types the token). The account needs IndicTrans2, IndicNER; Llama is optional.
3. `python run_local.py doctor`, then `download` (~30 GB), `data`, `smoke`, `full`. Run the long ones in the background
   (`... > runs/full.log 2>&1`) and check `python run_local.py status` every 10–15 minutes.
4. `python run_local.py package` → `results/factindic_final.zip`.

Rough time after the downloads: data 20 min, smoke 30–45 min, full 3–4.5 h (guess; no RTX 5050 measurement yet).
`full` drops optional jobs at `--soft 4` hours and stops at `--hard 5.5`. Set these from the time that is really left.

## While it runs
- One job at a time, each its own process. Logs: `runs/factindic/logs/auto_<job>.<shard>.log` (smoke: `runs/smoke/factindic/logs/`).
- Finished items are never repeated. To fix a bug: stop the run, edit, start the same command again.
  Jobs that already launched keep the old code; queued jobs read the new code.
- After an edit that changes an output's meaning, delete that system's `outputs/<system>*.jsonl` and `eval/eval_<system>*.jsonl`.
- Check the content too, not only "ok": read a few summaries per system (right script, not empty, not cut, no loops).
- Update a failed package version only in `factindic/requirements.txt` + `requirements-local.txt` together (preflight compares).

## Most likely problems
| Where | What to check |
|---|---|
| `llm._gemma_split` | Gemma-4 in bf16, per-layer embedding table in RAM, towers dropped, ~5 GB VRAM. If the device map fails it loads through RAM (~11 GB). Last resort: 4-bit via `_trials` |
| `sft.train_lora` | micro-batch 1, gradient checkpointing with Gemma-4's shared KV layers. If SFT fails, the rest continues without it |
| `decoding._contrastive_generate` | manual cache loop; compare one greedy output with `llm.generate` (α=0, β=0 must match) |
| `stage2.train_seq2seq` | mT5 uses Adafactor + bf16 + checkpointing on this GPU |
| IndicBARTSS tokenizer | asserts stop the job if tags or accents are broken; mT5 remains as baseline |
| out of memory | batches split themselves; also close other GPU programs. On Windows the driver spills into RAM and only gets slow |

## Rules
- ₹0, open models only, no paid APIs. FactCG is held out: never for training weights or reranking.
- Don't change the hypotheses, thresholds or samples after seeing results. Report negative results as they are.
- State in the paper: reduced sizes (`n_use` in `config.py`), the natural code-mixed CMI actually reached (spec: ≥ 8),
  bf16 here versus 4-bit in H1, old Romanizer in H1, H3 planned but not evaluated (no human labels; do not report the label-free proxy).
- After any change in `factindic/`, run `python tools/build_notebooks.py` so the Kaggle notebook stays identical.
- Don't commit `runs/`, models or tokens. Commit code, figures, tables and `results/factindic_final.zip` (if < 50 MB).

## After the run
Figures are in `runs/factindic/figures` (PNG, SVG, CSV), tables in `eval/main_table.csv`, tests in `eval/results.json`.
Still to write: IEEE paper, slides, optionally the results dashboard (H1 version: https://claude.ai/artifact/Dg2maoLN5pDQZJ3XYTPZVb).
