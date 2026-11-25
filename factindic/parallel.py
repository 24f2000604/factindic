"""Run a sharded job on every visible GPU (Kaggle T4 x2 → 2 processes), else in-process.

A job is a function `f(shard: int, n_shards: int, **kwargs)` that writes its own shard file through
`JsonlAppender(..., shard=shard)`. Free GPU memory in the notebook kernel before launching.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _resolve(path):
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)


def run(func_path, n_gpus=None, **kwargs):
    import torch
    n = n_gpus or max(1, torch.cuda.device_count())
    if n == 1:
        return _resolve(func_path)(shard=0, n_shards=1, **kwargs)
    from .config import DIRS
    DIRS["logs"].mkdir(parents=True, exist_ok=True)
    tag = func_path.replace(":", ".") + f".{int(time.time())}"
    job = DIRS["logs"] / f"{tag}.json"
    job.write_text(json.dumps(kwargs, ensure_ascii=False, default=str))
    pkg_root = str(Path(__file__).resolve().parent.parent)
    procs, logs = [], []
    for i in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i),
                   PYTHONPATH=pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        log = DIRS["logs"] / f"{tag}.gpu{i}.log"
        logs.append(log)
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", "factindic.parallel", func_path,
                                       str(job), str(i), str(n)], env=env,
                                      stdout=open(log, "w"), stderr=subprocess.STDOUT))
    print(f"launched {n} workers for {func_path}; logs: {DIRS['logs']}/{tag}.gpu*.log")
    last, shown = 0.0, {}
    while any(p.poll() is None for p in procs):
        time.sleep(15)
        if time.time() - last > 120:
            last = time.time()
            for i, log in enumerate(logs):
                tail = log.read_text(errors="ignore").strip().splitlines()[-1:] if log.exists() else []
                if tail and tail != shown.get(i):
                    shown[i] = tail
                    print(f"  gpu{i}: {tail[0][:160]}")
    bad = [i for i, p in enumerate(procs) if p.returncode != 0]
    if bad:
        for i in bad:
            print(f"--- gpu{i} log tail ---\n" + "\n".join(logs[i].read_text(errors='ignore').splitlines()[-40:]))
        raise RuntimeError(f"workers {bad} failed for {func_path}")
    return None


if __name__ == "__main__":
    func_path, job, i, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    kwargs = json.loads(Path(job).read_text())
    _resolve(func_path)(shard=i, n_shards=n, **kwargs)
