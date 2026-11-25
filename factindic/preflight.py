"""Run at the top of every notebook: fail in the first minutes, not hours in."""
from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

from .config import DATA_VERSION, MODELS, data_version

PINS = Path(__file__).with_name("requirements.txt")
PROBES = {"hi": "सूरत के अस्पताल में सोमवार को 42 मरीज भर्ती हुए।",
          "gu": "સુરતની હોસ્પિટલમાં સોમવારે 42 દર્દીઓ દાખલ થયા."}


def versions():
    bad = []
    for name, want in re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", PINS.read_text(), re.M):
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        if got != want:
            bad.append(f"{name}: need {want}, found {got}")
    if bad:
        raise RuntimeError("Wrong package versions. Re-run the install cell, then restart the kernel if a "
                           "package was already imported:\n  " + "\n  ".join(bad))
    print("versions ok")


def access(required=(), optional=()):
    """Gated-model access for the account behind HF_TOKEN."""
    from huggingface_hub import HfApi
    missing = []
    for key in (*required, *optional):
        try:
            HfApi().auth_check(MODELS[key])
            print("access ok:", MODELS[key])
        except Exception as e:
            msg = f"{MODELS[key]} ({type(e).__name__}): accept the terms at https://huggingface.co/{MODELS[key]}"
            if key in required:
                missing.append(msg)
            else:
                print("WARNING, optional model not accessible:", msg)
    if missing:
        raise RuntimeError("Hugging Face access missing:\n  " + "\n  ".join(missing))


def translation():
    """Build the IndicTrans2 environment and translate one Hindi and one Gujarati sentence."""
    from .mt import Translator
    mt = Translator(cache_name="mt_preflight.jsonl")
    try:
        for lang, s in PROBES.items():
            en = mt._run([s], lang)[0]          # bypass the cache: the point is to exercise the engine
            print(f"{mt.backend} {lang}: {en}")
            if "42" not in en or "hospital" not in en.lower():
                raise RuntimeError(f"translation sanity check failed for {lang}: {en!r}")
    finally:
        mt.close()


def data():
    v = data_version()
    if v != DATA_VERSION:
        raise RuntimeError(f"Notebook 01 output is missing or old (data version {v}, need {DATA_VERSION}). "
                           "Attach the output of the latest notebook 01 run (Add Input → Your Work).")
    print("data ok")


def check(mt=False, required=(), optional=(), need_data=True):
    versions()
    if need_data:
        data()
    access(required, optional)
    if mt:
        translation()
    print("preflight passed")
