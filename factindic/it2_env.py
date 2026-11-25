"""Builds the isolated environment IndicTrans2 needs.

IndicTrans2 cannot run on transformers 5 (needed by Gemma-4): its toolkit and tokenizer import
`transformers.tokenization_utils`, its config imports `transformers.onnx` (both removed), and its
model code uses the legacy tuple cache. A small venv with upstream's own pin (4.53.2) sidesteps all
three, and the rest of the project stays on one environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from .config import IS_LOCAL, REPO

LOCK = Path(__file__).with_name("requirements-it2.txt")
SERVER = Path(__file__).with_name("it2_server.py")
ENV_DIR = Path(os.environ.get("FACTINDIC_IT2_ENV") or (REPO / ".it2_env" if IS_LOCAL else "/tmp/factindic_it2_env"))
TORCH_LOCAL, INDEX_LOCAL = "torch==2.8.0", "https://download.pytorch.org/whl/cu128"   # the locked 2.6 has no RTX 50 kernels


def clean_env():
    """Parent environment without the settings that would leak the notebook's packages into the venv."""
    drop = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_CONSTRAINT", "PIP_TARGET"}
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("UV_")}
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", HF_HUB_ENABLE_HF_TRANSFER="0", TOKENIZERS_PARALLELISM="false",
               HF_MODULES_CACHE=str(ENV_DIR) + "_modules")
    return env


def _run(cmd, what):
    p = subprocess.run([str(c) for c in cmd], env=clean_env(), capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"IndicTrans2 environment: {what} failed\n{(p.stdout + p.stderr)[-3000:]}")
    return p.stdout


def _uv():
    if shutil.which("uv"):
        return ["uv"]
    if importlib.util.find_spec("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"], "installing uv")
    return [sys.executable, "-m", "uv"]


@contextmanager
def _lock(path):
    with open(path, "w") as lk:
        try:
            import fcntl
            fcntl.flock(lk, fcntl.LOCK_EX)
        except ImportError:              # Windows: one job at a time there, so no lock is needed
            pass
        yield


def ensure():
    """Create the venv once (safe to call from several workers); returns its python path."""
    py = ENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stamp = ENV_DIR / ".ready"
    digest = hashlib.md5(LOCK.read_bytes() + (TORCH_LOCAL if IS_LOCAL else "").encode()).hexdigest()
    with _lock(str(ENV_DIR) + ".lock"):
        if py.exists() and stamp.exists() and stamp.read_text() == digest:
            return py
        if ENV_DIR.exists():
            shutil.rmtree(ENV_DIR)
        uv = _uv()
        print("building the IndicTrans2 environment (2-4 minutes, once per session)...", flush=True)
        try:
            _run([*uv, "venv", "--python", "3.11", ENV_DIR], "creating a Python 3.11 venv")
        except RuntimeError as e:
            print("Python 3.11 unavailable, using the notebook's Python:", str(e)[-200:])
            shutil.rmtree(ENV_DIR, ignore_errors=True)
            _run([*uv, "venv", "--python", sys.executable, ENV_DIR], "creating a venv")
        lock = LOCK
        if IS_LOCAL:                     # the GPU packages come with their own CUDA 12.8 build of torch
            _run([*uv, "pip", "install", "--python", py, TORCH_LOCAL, "--index-url", INDEX_LOCAL], "installing torch")
            lock = ENV_DIR / "requirements-local.txt"
            lock.write_text("\n".join(x for x in LOCK.read_text(encoding="utf-8").splitlines()
                                      if not x.startswith(("torch==", "triton==", "sympy==", "nvidia-", "#"))), encoding="utf-8")
        _run([*uv, "pip", "install", "--python", py, "--no-deps", "-r", lock], "installing packages")
        _run([py, "-c", "import torch; from transformers import AutoModelForSeq2SeqLM; "
                        "from IndicTransToolkit.processor import IndicProcessor; IndicProcessor(inference=True); "
                        "assert torch.cuda.is_available(), 'no CUDA in the IndicTrans2 environment'; "
                        "torch.ones(8, device='cuda').sum().item()"], "import check")
        stamp.write_text(digest)
    return py
