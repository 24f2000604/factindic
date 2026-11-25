"""Paths, model inventory, budgets, and session setup for Kaggle / Colab T4 runs."""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/working")
IN_COLAB = "google.colab" in sys.modules
IS_LOCAL = not IS_KAGGLE and not IN_COLAB      # a laptop with one 8 GB GPU that has native bf16 (RTX 40/50)
REPO = Path(__file__).resolve().parent.parent


def _root() -> Path:
    if os.environ.get("FACTINDIC_ROOT"):
        return Path(os.environ["FACTINDIC_ROOT"])
    if IS_KAGGLE:
        return Path("/kaggle/working/factindic")
    if IN_COLAB:
        return Path("/content/drive/MyDrive/factindic")
    return REPO / "runs" / "factindic"


ROOT = _root()
DIRS = {k: ROOT / k for k in
        ["data", "outputs", "verify", "eval", "figures", "models", "cache", "logs"]}
H1_DIR = ROOT.parent / "factindic_h1"
DATA_VERSION = 3                          # bump when the data files of notebook 01 must be rebuilt
VERSION_FILE = DIRS["data"] / "VERSION"


def data_version(root=None):
    f = Path(root) / "data" / "VERSION" if root else VERSION_FILE
    try:
        return int(f.read_text())
    except (OSError, ValueError):
        return 0


def h1_pilot_ids():
    return set(Path(__file__).with_name("h1_pilot_ids.txt").read_text().split())

LANGS = {
    "hi": dict(name="Hindi", script="Devanagari", xlsum="hindi", ilsum="Hindi", it2="hin_Deva",
               block=(0x0900, 0x097F)),
    "gu": dict(name="Gujarati", script="Gujarati", xlsum="gujarati", ilsum="Gujarati", it2="guj_Gujr",
               block=(0x0A80, 0x0AFF)),
}
VARIANTS = ["native", "script_mixed", "roman"]
SEEDS = [42, 108, 2026]

MODELS = {
    # generators
    "gemma": "google/gemma-4-E2B-it",           # backbone chosen by the H1 gate
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "sarvam": "sarvamai/sarvam-1",
    "indicbartss": "ai4bharat/IndicBARTSS",
    "mt5": "google/mt5-base",
    # stage 1
    "indicner": "ai4bharat/IndicNER",
    "latin_ner": "Babelscape/wikineural-multilingual-ner",
    # stage 4
    "it2": "ai4bharat/indictrans2-indic-en-dist-200M",
    "nllb": "facebook/nllb-200-distilled-600M",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "minicheck": "lytang/MiniCheck-Flan-T5-Large",
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    # held-out evaluators (never used for training weights or reranking)
    "factcg": "yaxili96/FactCG-DeBERTa-v3-Large",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
}

# Default budgets sized for a free T4 (Kaggle: 30 GPU-h/week; T4 x2 is used when present).
CFG = dict(
    seed=42,
    n_test_per_lang=150,          # controlled parallel testbed (300 articles; cut from 250 to fit the GPU budget)
    n_dev_per_lang=50,            # dev split for tuning (100 articles)
    # articles per language actually run (balanced over the entity-density tertiles); None = all
    n_use=dict(testbed=60, dev=24, natural_cm=40),
    n_natural_cm_per_lang=100,    # ILSUM natural code-mixed subset
    # spec: CMI >= 8. Too few clean articles reach it, so each language uses the highest step with enough of them
    cmi_natural=[8.0, 6.0, 5.0, 4.0, 3.0],
    cmi_max_run=5,                # longer runs of Latin words are captions, tweets or headlines, not mixing
    max_article_words=350,
    n_weighted_per_lang=4000,     # sanitized training pool (spec §9.3); seq2seq baselines train on all of it
    n_score_per_lang=2000,        # pool subset that gets mFACT faithfulness scores (Gemma SFT samples from it)
    n_sft_per_lang=600,           # subset of the weighted pool used for Gemma LoRA SFT (one T4 session)
    w_min=0.30,
    sft=dict(max_len=1024, micro_bs=2, accum=8, epochs=1, lr=2e-4, r=16, alpha=32, dropout=0.05),
    # Llama-3.2 spends ~10 tokens per Gujarati word: 44 of 100 H1 summaries needed more than 160 tokens
    gen=dict(batch_size=8, max_new_tokens=160, max_new_tokens_by_model={"llama": 320}, latency_probe_n=15),
    mt_backend="it2",             # "it2" (IndicTrans2, isolated environment) or "nllb"; never mixed silently
    cad=dict(alpha_base=[0.2, 0.4, 0.6], alpha_anchor=[0.6, 0.9, 1.2], beta=[0.0, 0.2, 0.4],
             eta=0.1, static_alpha=0.5, tune_docs=20),
    n_decoding_ablation_per_lang=40,    # Exp 4a-4d run on 80 articles x 3 variants
    K=5, sample_T=0.7, sample_top_p=0.9,
    n_cove=60,                     # CoVe self-rewrite baseline (Exp 5b) on a subset
    tau_ver=0.5, lambda_len=0.2,
    bootstrap=1000,
    n_s2s_train=4000, s2s_epochs=2,      # training pairs and epochs of the two seq2seq baselines
    n_tune=4,                            # decoding settings tried on dev
)
if IS_LOCAL:
    CFG["sft"].update(micro_bs=1, accum=16)          # same effective batch of 16
    CFG["gen"]["batch_size_by_model"] = {"llama": 4}
CFG.update(json.loads(os.environ.get("FACTINDIC_CFG", "{}")))       # overrides for the smoke test

SYSTEMS = {
    # name: (description, experiment id)
    "indicbartss": ("IndicBARTSS fine-tuned, beam 4", "Exp 1a"),
    "mt5": ("mT5-base fine-tuned, beam 4", "Exp 1b"),
    "gemma_A": ("Gemma-4 zero-shot, Cond A standard prompt", "Exp 2b"),
    "gemma_B1": ("Gemma-4 zero-shot, Cond B1 + anchor checklist", "Exp 3a"),
    "gemma_B2": ("Gemma-4 zero-shot, Cond B2 + 2-step CoD", "Exp 3b"),
    "gemma_B3": ("Gemma-4 zero-shot, Cond B3 + matrix-verb hint", "Exp 3c"),
    "gemma_sft": ("Gemma-4 LoRA-SFT (faithfulness-weighted), greedy", "Exp 2a"),
    "llama_A": ("Llama-3.2-3B zero-shot, Cond A", "Exp 2c"),
    "dec_greedy": ("Stage-2 best + plain greedy", "Exp 4-ref"),
    "dec_static_cad": ("Static CAD", "Exp 4a"),
    "dec_adacad": ("AdaCAD (token-level JSD alpha)", "Exp 4b"),
    "dec_dola": ("DoLa (dynamic JSD premature layer)", "Exp 4c"),
    "dec_anchor_cad_dola": ("Anchor-aware dynamic CAD + DoLa", "Exp 4d"),
    "drop_only": ("Greedy + drop every unverified sentence", "Exp 5a"),
    "cove": ("Chain-of-Verification self-rewrite", "Exp 5b"),
    "factindic": ("Full pipeline: Stage 1-3 + K=5 router rerank + prune", "Exp 6"),
}


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def restore_inputs():
    """Kaggle: merge outputs of earlier notebooks attached as inputs into /kaggle/working."""
    if not IS_KAGGLE:
        return []
    restored = []

    def copy(src, dst):              # keep whichever file has more finished work
        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):
            shutil.copy2(src, dst)
        return dst

    for name in ("factindic", "factindic_h1"):
        found = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        for src in sorted(found, key=data_version, reverse=True):
            # only real run folders (never a copy of the code package, which is also named factindic)
            is_run = any(os.path.isdir(os.path.join(src, d)) for d in ("data", "outputs", "eval")) or \
                os.path.exists(os.path.join(src, "pilot.jsonl"))
            if os.path.isdir(src) and is_run:
                if name == "factindic" and data_version(src) < data_version():
                    print(f"ignored (older data version): {src}")
                    continue
                shutil.copytree(src, ROOT.parent / name, dirs_exist_ok=True, copy_function=copy)
                restored.append(src)
    return restored


def hf_login():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if not tok:                          # `hf auth login` on a laptop
        from huggingface_hub import get_token
        tok = get_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
        from huggingface_hub import login
        login(token=tok)
    else:
        print("WARNING: HF_TOKEN not set; gated models will fail.")
    return bool(tok)


def safe_attention():
    """On the RTX 5050 (torch 2.8, CUDA 12.8) PyTorch's memory-efficient attention kernel silently gives wrong
    results for Gemma-4 when the sequence length is 32k + 1 (513, 673, 1089: logits off by 10-30). The math
    kernel agrees with eager attention at every length."""
    if IS_LOCAL:
        import torch
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)


def max_new_tokens(model_key):
    g = CFG["gen"]
    return g["max_new_tokens_by_model"].get(model_key, g["max_new_tokens"])


def seed_everything(seed=42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def setup(need_gpu=True):
    if IN_COLAB and not os.environ.get("FACTINDIC_ROOT"):
        from google.colab import drive
        drive.mount("/content/drive")
    os.environ["FACTINDIC_ROOT"] = str(ROOT)   # parallel worker processes inherit the same root
    restored = restore_inputs()
    ensure_dirs()
    h1 = Path(__file__).with_name("h1_results")          # the finished H1 run, for its figures
    if not (H1_DIR / "h1_tests.json").exists() and h1.exists():
        shutil.copytree(h1, H1_DIR, dirs_exist_ok=True)
    hf_login()
    seed_everything(CFG["seed"])
    n_gpu = 0
    try:
        import torch
        n_gpu = torch.cuda.device_count()
        if need_gpu:
            assert n_gpu > 0, "No GPU visible (Kaggle: GPU T4 x2; Colab: T4; laptop: python run_local.py doctor)."
        # asking for the name opens a CUDA context in this process, which an 8 GB card cannot spare
        names = [f"{n_gpu} visible"] if IS_LOCAL else [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    except ImportError:
        names = []
    info = dict(root=str(ROOT), restored=restored, gpus=names)
    print(json.dumps(info, indent=2))
    return info
