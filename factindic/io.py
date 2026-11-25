"""JSONL checkpointing, resumable job helpers, GPU hygiene, timing."""
from __future__ import annotations

import gc
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path


def jl_read(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # truncated last line after a disconnect
    return rows


def jl_read_glob(path_pattern):
    """Read `name.jsonl` plus shard files `name.shard*.jsonl` written by parallel workers."""
    p = Path(path_pattern)
    rows = jl_read(p)
    for sh in sorted(p.parent.glob(p.stem + ".shard*.jsonl")):
        rows += jl_read(sh)
    return rows


def jl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


class JsonlAppender:
    """Append rows with flush; `done` holds keys already present so reruns skip finished work."""

    def __init__(self, path, key_fn, shard=None):
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        # parallel workers each append to their own shard file; `done` spans base + all shards
        self.path = base if shard is None else base.with_name(f"{base.stem}.shard{shard}.jsonl")
        self.key_fn = key_fn
        self.done = {key_fn(r) for r in jl_read_glob(base)}
        self._f = open(self.path, "a", encoding="utf-8")

    def add(self, row):
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        self.done.add(self.key_fn(row))

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def json_load(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def md5(*parts):
    return hashlib.md5("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def free_gpu():
    """Call after `del model` in the caller (a callee cannot drop the caller's references)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


@contextmanager
def cuda_timer(out: dict, key="seconds"):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out[key] = time.perf_counter() - t


def peak_vram_gb():
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def shard(items, i, n):
    return [x for j, x in enumerate(items) if j % n == i]


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
