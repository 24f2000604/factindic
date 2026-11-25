"""Run the project on a laptop with one 8 GB GPU (RTX 40/50).

    python run_local.py doctor          check GPU, memory, disk, packages, Hugging Face access
    python run_local.py download        all models, about 30 GB (do this first, it can take an hour)
    python run_local.py data            part 1: data version 3 (reuses inputs/notebook01_v2.zip)
    python run_local.py smoke           part 2: every step on one article per language (25-40 min)
    python run_local.py full            part 3: everything, then tables and figures
    python run_local.py all             data, smoke (skipped once it has passed), full
    python run_local.py status          what is running, finished, failed
    python run_local.py package         results zip without models and caches

Every step writes JSONL and skips finished work: after a crash or a fix, run the same command again.
Outputs: runs/factindic. Job logs: runs/factindic/logs/auto_<job>.<shard>.log (smoke: runs/smoke/...).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not sys.flags.utf8_mode:                              # Windows consoles default to cp1252
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv], env=dict(os.environ, PYTHONUTF8="1")))
os.chdir(HERE)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
PASSED = HERE / "runs" / ".smoke_passed"


def seed_inputs():
    """Output of the finished notebook 01 (testbed with its entities, training pool, translations, scores)."""
    data = HERE / "runs" / "factindic" / "data"
    z = HERE / "inputs" / "notebook01_v2.zip"
    if not (data / "testbed.jsonl").exists() and not (data / "VERSION").exists() and z.exists():
        (HERE / "runs").mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(HERE / "runs", members=[m for m in f.namelist() if m.startswith("factindic/") and "/logs/" not in m
                                                 and "/figures/" not in m])
        print("restored the notebook 01 output from", z.name)


def doctor():
    import shutil
    ok = True

    def say(good, text, hard=True):
        nonlocal ok
        ok &= good or not hard
        print(("ok    " if good else ("FAIL  " if hard else "warn  ")) + text)

    nllb = os.environ.get("FACTINDIC_MT") == "nllb"
    say(os.name != "nt" or nllb, f"platform {sys.platform}" + (" (native Windows needs FACTINDIC_MT=nllb, or use WSL2)" if os.name == "nt" else ""))
    import psutil
    import torch
    say(torch.cuda.is_available(), f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        free, total = (x / 1e9 for x in torch.cuda.mem_get_info())
        say(total > 7.5, f"{torch.cuda.get_device_name(0)}: {free:.1f} of {total:.1f} GB free")
        say(free > 6.8, "at least 6.8 GB of GPU memory free (close games, browsers, other GPU programs)", hard=False)
        say(torch.cuda.is_bf16_supported(including_emulation=False), "native bf16")
        try:
            (torch.ones(64, 64, device="cuda", dtype=torch.bfloat16) @ torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)).sum().item()
            say(True, "GPU kernels run")
        except Exception as e:
            say(False, f"GPU kernels: {e!r}")
    ram = psutil.virtual_memory()
    say(ram.total > 11e9, f"RAM {ram.total / 1e9:.0f} GB, {ram.available / 1e9:.0f} GB free (WSL2: raise it in %UserProfile%\\.wslconfig)", hard=False)
    say(shutil.disk_usage(HERE).free > 60e9, f"disk {shutil.disk_usage(HERE).free / 1e9:.0f} GB free (models need about 45 GB)", hard=False)
    say((HERE / "inputs" / "notebook01_v2.zip").exists(), "inputs/notebook01_v2.zip", hard=False)
    if hasattr(psutil, "sensors_battery") and psutil.sensors_battery():
        say(psutil.sensors_battery().power_plugged, "plugged in (on battery the GPU is throttled)", hard=False)
    from factindic import preflight
    from factindic.config import hf_login
    try:
        preflight.versions()
        say(hf_login(), "Hugging Face token (hf auth login)")
        preflight.access((() if nllb else ("it2",)) + ("gemma", "indicner"), ("llama", "indicbartss", "indicbert"))
        preflight.translation()
    except Exception as e:
        say(False, str(e)[-1500:])
    print("\nREADY" if ok else "\nNOT READY")
    return 0 if ok else 1


def download():
    """Only the files transformers loads, so nothing waits for the network during the run."""
    from huggingface_hub import HfApi, snapshot_download
    from factindic.config import MODELS, hf_login
    hf_login()
    bad = []
    for key, mid in MODELS.items():
        if key in ("sarvam", "qwen") or (key == "nllb") != (os.environ.get("FACTINDIC_MT") == "nllb") and key in ("nllb", "it2"):
            continue                                     # H1 only / optional / the translator not in use
        try:
            files = HfApi().list_repo_files(mid)
            top = [f for f in files if "/" not in f]
            st = any(f.endswith(".safetensors") for f in top)
            want = [f for f in top if f.endswith((".json", ".model", ".txt", ".jinja", ".py", ".vocab", ".safetensors"))
                    or (f.startswith("pytorch_model") and not st) or (key == "it2" and f.startswith(("dict", "model")))]
            print(key, mid, flush=True)
            snapshot_download(mid, allow_patterns=want)
        except Exception as e:
            bad.append(f"{key}: {repr(e)[:200]}")
    print("\n".join(["downloaded everything"] if not bad else ["PROBLEMS:", *bad]))
    return 1 if bad else 0


def status():
    for label, root in (("smoke", HERE / "runs" / "smoke" / "factindic"), ("full", HERE / "runs" / "factindic")):
        logs = sorted((root / "logs").glob("auto_*.log"), key=lambda p: p.stat().st_mtime) if (root / "logs").exists() else []
        if not logs:
            continue
        f = root / "logs" / f"auto_status{'_smoke' if label == 'smoke' else ''}.json"
        print(f"== {label}", {k: v for k, v in json.loads(f.read_text()).items() if v != "ok"} if f.exists() else "(in progress)")
        for p in logs[-4:]:
            tail = p.read_text(errors="ignore").replace("\r", "\n").strip().splitlines()[-1:]
            print(f"  {p.name}: {(time.time() - p.stat().st_mtime) / 60:.0f} min ago: {tail[0][:140] if tail else ''}")
        out = {p.name.split(".")[0]: 0 for p in (root / "outputs").glob("*.jsonl")}
        for p in (root / "outputs").glob("*.jsonl"):
            out[p.name.split(".")[0]] += sum(1 for _ in open(p, encoding="utf-8"))
        print("  outputs:", out)
    return 0


def package():
    root, path = HERE / "runs" / "factindic", HERE / "results" / "factindic_final.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if p.is_file() and rel.parts[0] not in ("models", "cache"):
                z.write(p, Path("factindic") / rel)
    print(path, f"{path.stat().st_size / 1e6:.0f} MB")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["doctor", "download", "data", "smoke", "full", "all", "status", "package"])
    ap.add_argument("--soft", type=float, default=4.0, help="hours after which optional jobs are dropped")
    ap.add_argument("--hard", type=float, default=5.5, help="hours after which all jobs stop and the analysis runs")
    a = ap.parse_args()
    if a.command in ("doctor", "download", "status", "package"):
        return globals()[a.command]()
    seed_inputs()
    from factindic.config import setup
    setup()
    from factindic.pipeline import auto
    if a.command in ("data", "all"):
        auto.part1_data()
    if a.command == "smoke" or (a.command == "all" and not PASSED.exists()):
        auto.part2_smoke()
        PASSED.write_text("ok")
    if a.command in ("full", "all"):
        print(auto.part3_full(a.soft, a.hard))
        package()
    return 0


if __name__ == "__main__":
    sys.exit(main())
