"""Everything after notebook 01 in one unattended run.

A queue of small jobs runs on all GPUs. Each job is its own process, so an error or a crash stops only
that job; it is retried once, and later jobs use whatever exists. A short smoke test of the whole
pipeline runs first. Optional jobs are dropped at `soft_h`, everything stops at `hard_h`, and the
analysis always runs. Finished work is never repeated, so a rerun continues where the last one stopped.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CFG, DIRS, IS_KAGGLE, MODELS, ROOT
from ..io import free_gpu, jl_read_glob, json_dump, json_load

SELECTED = DIRS["outputs"] / "stage2_selected.json"
ADAPTER = DIRS["models"] / "gemma_sft" / "final"
SMOKE = Path("/tmp/factindic_smoke") if IS_KAGGLE else ROOT.parent / "smoke"


# --------------------------------------------------------------------------- jobs not found elsewhere
def prep(shard, n_shards, **_):
    from ..annotate import export_naturalness
    from ..data import load_set
    from . import stage1
    stage1.build_all()
    stage1.prepare_weights()
    export_naturalness(load_set("testbed"))


def sft(shard, n_shards, **_):
    from ..llm import load_tokenizer
    from ..sft import train_lora
    from .stage2 import sft_examples
    if (ADAPTER / "adapter_config.json").exists():
        return
    ex = sft_examples(load_tokenizer(MODELS["gemma"]))
    print(len(ex), "SFT examples", flush=True)
    train_lora(MODELS["gemma"], ex, ADAPTER.parent, CFG["sft"])
    shutil.rmtree(ADAPTER.parent / "ckpt", ignore_errors=True)


def sft_gen(shard, n_shards, **kw):
    from .stage2 import llm_gen_job
    if not (ADAPTER / "adapter_config.json").exists():
        raise RuntimeError("no SFT adapter (the sft job failed)")
    llm_gen_job(shard, n_shards, system="gemma_sft", model_key="gemma", cond="A", adapter=str(ADAPTER), **kw)


def s2s(shard, n_shards, name, **_):
    from .stage2 import _eval_items, seq2seq_gen_job, train_seq2seq
    done = {r["item"] for r in jl_read_glob(DIRS["outputs"] / f"{name}.jsonl")}
    if {x[0] for x in _eval_items()} <= done:
        return
    train_seq2seq(name)
    seq2seq_gen_job(0, 1, name=name)
    shutil.rmtree(DIRS["models"] / name, ignore_errors=True)     # 1-2 GB, not needed again


def select(shard, n_shards, **_):
    """Dev-set choice of the Stage-3 base (B2 is two-pass, so it cannot be the single-pass decoding base)."""
    from . import stage2
    have = {f.name.split(".")[0] for f in DIRS["outputs"].glob("gemma_*.jsonl")}
    scores = {s: stage2.dev_grade(s) for s in ("gemma_A", "gemma_B1", "gemma_B3", "gemma_sft") if s in have}
    stage2.release_dev_router()
    json_dump(scores, DIRS["outputs"] / "stage2_dev_scores.json")
    cands = {"gemma_B1": ("B1", None), "gemma_B3": ("B3", None), "gemma_sft": ("A", str(ADAPTER))}
    cands = {s: c for s, c in cands.items() if scores.get(s, {}).get("n")}
    best = max(cands, key=lambda s: (scores[s]["score"], scores[s]["rougeL"]))
    json_dump(dict(system=best, model_key="gemma", cond=cands[best][0], adapter=cands[best][1]), SELECTED)
    print(json.dumps(scores, indent=1), "\nStage-3 base:", best, flush=True)


def pick(shard, n_shards, **_):
    from .stage3 import pick_params
    print(pick_params()[0], flush=True)


def ppl(shard, n_shards, **_):
    from .stage3 import ppl_job, selected
    ppl_job(shard, n_shards, systems=[selected()["system"], "dec_static_cad", "dec_adacad", "dec_dola", "dec_anchor_cad_dola"])


def lookback(shard, n_shards, **_):
    from .stage3 import lookback_job, selected
    lookback_job(shard, n_shards, systems=[selected()["system"], "dec_anchor_cad_dola"])


def ablation(shard, n_shards, method, **_):
    from .stage3 import ablation_uids, decode_job
    decode_job(shard, n_shards, method=method, system=f"dec_{method}", uids=ablation_uids())


def latency(shard, n_shards, **_):
    from .stage2 import latency_probe
    from .stage3 import selected
    sel = selected()
    print(latency_probe("gemma_A", "gemma", "A"), flush=True)
    print(latency_probe(sel["system"], "gemma", sel["cond"], sel["adapter"]), flush=True)


def materialize(shard, n_shards, **_):
    from .stage4 import materialize as run
    print(run(), "final summaries", flush=True)


def bertscore(shard, n_shards, **_):
    from .evaluate import bertscore_job
    bertscore_job(shard, n_shards, systems=[s for s in EVAL if list(DIRS["outputs"].glob(f"{s}.*jsonl"))])


def final(shard, n_shards, **_):
    """CPU only: sheets, label-free H3, Lookback detector, tests, tables, figures."""
    import traceback
    from . import analysis, human, stage1
    for name, f in (("annotation sheets", human.export_sheets), ("H3 proxy and Lookback", lambda: human.run(gpu=False)),
                    ("analysis", analysis.run_all), ("data figures", stage1.figures)):
        try:
            f()
            print("done:", name, flush=True)
        except Exception:
            print("FAILED:", name, flush=True)
            traceback.print_exc()


# --------------------------------------------------------------------------- the plan
P = "factindic.pipeline."
EVAL = ["gemma_A", "gemma_B1", "gemma_B3", "gemma_sft", "dec_anchor_cad_dola", "dec_static_cad", "dec_adacad", "dec_dola",
        "indicbartss", "mt5", "llama_A", "gemma_B2", "cove", "drop_only", "factindic"]
DEV = dict(sets=["dev"], variants=["native", "roman"])


@dataclass
class Job:
    name: str
    func: str
    kw: dict = field(default_factory=dict)
    shards: int = 1
    after: tuple = ()            # finished, failed or dropped: order only
    prio: int = 50               # lower runs first
    optional: bool = False       # dropped at soft_h
    core: bool = False           # a failure in the smoke test stops the run
    cpu: bool = False


def plan(smoke):
    g = lambda c: dict(system=f"gemma_{c}", model_key="gemma", cond=c)
    gen, dec = P + "stage2:llm_gen_job", P + "stage3:decode_job"
    J = [Job("weights", P + "stage1:weights_job", shards=2, prio=0),
         Job("refs", P + "evaluate:ref_job", shards=2, prio=20, core=True),
         Job("sft", P + "auto:sft", after=("weights",), prio=1),
         Job("sft_dev", P + "auto:sft_gen", DEV, after=("sft",), prio=2),
         Job("dev_A", gen, dict(g("A"), **DEV), prio=3), Job("dev_B1", gen, dict(g("B1"), **DEV), prio=3, core=True),
         Job("dev_B3", gen, dict(g("B3"), **DEV), prio=3),
         Job("select", P + "auto:select", after=("dev_A", "dev_B1", "dev_B3", "sft_dev"), prio=4),
         Job("tune", P + "stage3:tune_generate_job", shards=2, after=("select",), prio=5),
         Job("pick", P + "auto:pick", after=("tune",), prio=6),
         Job("anchor_cad_dola", dec, dict(method="anchor_cad_dola", system="dec_anchor_cad_dola", sets=["testbed", "natural_cm"]),
             shards=2, after=("pick",), prio=7, core=True),
         Job("samples", P + "stage3:samples_job", shards=2, after=("select",), prio=8, core=True),
         Job("gen_B1", gen, g("B1"), shards=2, prio=10, core=True), Job("gen_A", gen, g("A"), shards=2, prio=11),
         Job("gen_sft", P + "auto:sft_gen", shards=2, after=("sft",), prio=12), Job("gen_B3", gen, g("B3"), shards=2, prio=13),
         Job("stage4", P + "stage4:verify_job", shards=2, after=("anchor_cad_dola", "samples", "gen_B1", "gen_B3", "gen_sft"),
             prio=9, core=True),
         Job("materialize", P + "auto:materialize", after=("stage4",), prio=9, core=True, cpu=True),
         Job("static_cad", P + "auto:ablation", dict(method="static_cad"), after=("pick",), prio=14),
         Job("indicbartss", P + "auto:s2s", dict(name="indicbartss"), prio=15),
         Job("dola", P + "auto:ablation", dict(method="dola"), after=("pick",), prio=16),
         Job("adacad", P + "auto:ablation", dict(method="adacad"), after=("pick",), prio=17),
         Job("llama", gen, dict(system="llama_A", model_key="llama", cond="A"), shards=2, prio=24, optional=True),
         Job("mt5", P + "auto:s2s", dict(name="mt5"), prio=25, optional=True),
         Job("cove", P + "stage3:cove_job", after=("select", "gen_B1", "gen_B3", "gen_sft"), prio=26, optional=True),
         Job("gen_B2", gen, g("B2"), shards=2, prio=27, optional=True),
         Job("ppl", P + "auto:ppl", after=("anchor_cad_dola", "static_cad", "dola", "adacad", "gen_B1", "gen_B3", "gen_sft"),
             prio=28, optional=True),
         Job("lookback", P + "auto:lookback", after=("anchor_cad_dola", "gen_B1", "gen_B3", "gen_sft"), prio=29, optional=True),
         Job("latency", P + "auto:latency", after=("select", "sft"), prio=30, optional=True)]
    src = dict(gemma_A="gen_A", gemma_B1="gen_B1", gemma_B3="gen_B3", gemma_B2="gen_B2", gemma_sft="gen_sft", llama_A="llama",
               indicbartss="indicbartss", mt5="mt5", cove="cove", dec_anchor_cad_dola="anchor_cad_dola",
               dec_static_cad="static_cad", dec_adacad="adacad", dec_dola="dola", drop_only="materialize", factindic="materialize")
    for s in EVAL:
        J.append(Job(f"eval_{s}", P + "evaluate:eval_job", dict(systems=[s]), after=("refs", src[s]),
                     prio=9 if s == "factindic" else 21, core=s == "factindic"))
    J.append(Job("bertscore", P + "auto:bertscore", after=tuple(f"eval_{s}" for s in EVAL), prio=40))
    if smoke:                    # one of each kind; the others differ only in prompt or method name
        drop = {"dev_A", "dev_B3", "gen_A", "gen_B3", "static_cad", "dola", "eval_gemma_A", "eval_gemma_B3", "eval_dec_static_cad",
                "eval_dec_dola", "eval_gemma_sft", "eval_dec_adacad", "eval_indicbartss", "eval_mt5", "eval_llama_A",
                "eval_gemma_B2", "eval_cove", "eval_drop_only", "latency"}
        J = [j for j in J if j.name not in drop]
        for j in J:
            j.shards = 1
    return J


# --------------------------------------------------------------------------- the runner
def _launch(job, shard, n, gpu, root, cfg):
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    kwf = logs / f"auto_{job.name}.json"
    kwf.write_text(json.dumps(job.kw))
    log = logs / f"auto_{job.name}.{shard}.log"
    env = dict(os.environ, FACTINDIC_ROOT=str(root), FACTINDIC_CFG=json.dumps(cfg), CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               PYTORCH_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.Popen([sys.executable, "-u", "-m", "factindic.parallel", job.func, str(kwf), str(shard), str(n)], env=env,
                         stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    return p, log


def _kill(p):
    import signal
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        return p.wait()
    try:
        os.killpg(p.pid, signal.SIGKILL)         # with its IndicTrans2 helper process
    except ProcessLookupError:
        pass
    p.wait()


def _tail(log, n):
    return "\n".join(log.read_text(errors="ignore").replace("\r", "\n").strip().splitlines()[-n:])


def run_plan(jobs, root, cfg, t0, n_gpu, soft_h=None, hard_h=None, timeout_min=150, retries=1, label=""):
    """Returns {job: "ok" | "failed" | "dropped"}. t0: start of the whole session, for soft_h and hard_h."""
    names = {j.name for j in jobs}
    todo = {(j.name, s): j for j in jobs for s in range(j.shards)}
    left = {j.name: j.shards for j in jobs}
    status, tries, running, free = {}, {}, {}, list(range(n_gpu))
    last = time.time()
    hours = lambda: (time.time() - t0) / 3600

    def finish(name, state):
        left[name] -= 1
        status[name] = "failed" if "failed" in (status.get(name), state) else ("dropped" if "dropped" in (status.get(name), state) else "ok")
        if left[name] == 0:
            print(f"[{hours():.2f} h] {label}{name}: {status[name]}", flush=True)

    while todo or running:
        stop = hard_h is not None and hours() >= hard_h
        soft = soft_h is not None and hours() >= soft_h
        for key in list(todo):
            if stop or (soft and todo[key].optional):
                del todo[key]
                finish(key[0], "dropped")
        for key, (p, log, job, gpu, start) in list(running.items()):
            rc = p.poll()
            late = stop or (soft and job.optional and hours() >= soft_h + 0.4) or time.time() - start > 60 * timeout_min
            if rc is None and not late:
                continue
            if rc is None:
                _kill(p)
            del running[key]
            if gpu is not None:
                free.append(gpu)
            if rc == 0:
                finish(key[0], "ok")
            elif rc is not None and tries.get(key, 0) < retries:
                tries[key] = tries.get(key, 0) + 1
                todo[key] = job
                print(f"[{hours():.2f} h] {label}{key[0]}#{key[1]} failed, retrying: {_tail(log, 1)[:200]}", flush=True)
            elif rc is None:
                print(f"[{hours():.2f} h] {label}{key[0]}#{key[1]} stopped (time)", flush=True)
                finish(key[0], "dropped")
            else:
                print(f"----- {label}{key[0]}#{key[1]} FAILED -----\n{_tail(log, 25)}\n-----", flush=True)
                finish(key[0], "failed")
        ready = [k for k, j in todo.items() if all(left[a] == 0 for a in j.after if a in names)]
        for key in sorted(ready, key=lambda k: (todo[k].prio, k[1])):
            job = todo[key]
            if not job.cpu and not free:
                continue
            gpu = None if job.cpu else free.pop(0)
            p, log = _launch(job, key[1], job.shards, gpu, root, cfg)
            running[key] = (p, log, job, gpu, time.time())
            del todo[key]
        if time.time() - last > 600:
            last = time.time()
            done = sum(v == 0 for v in left.values())
            print(f"[{hours():.2f} h] {label}{done}/{len(left)} jobs finished; running: " +
                  " | ".join(f"{k[0]}#{k[1]}: {_tail(log, 1)[:70]}" for k, (_, log, *_r) in running.items()), flush=True)
        time.sleep(5)
    json_dump(status, Path(root) / "logs" / f"auto_status{'_smoke' if label else ''}.json")
    return status


SMOKE_CFG = dict(n_use=dict(testbed=1, dev=1, natural_cm=1), n_sft_per_lang=8, n_decoding_ablation_per_lang=1, n_cove=2, n_tune=1,
                 n_s2s_train=32, s2s_epochs=1, n_score_per_lang=2, bootstrap=50, n_annot_summaries=6, n_annot_dev=2)


def smoke(t0, n_gpu):
    """Every kind of job on one article per language, in a scratch folder."""
    root = SMOKE / "factindic"
    shutil.rmtree(root.parent, ignore_errors=True)
    for d in ("data", "cache"):
        shutil.copytree(DIRS[d], root / d)
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    keys = json_load(root / "data" / "score_subset.json")
    old = {w["key"] for w in jl_read_glob(root / "data" / "weights.jsonl")}
    json_dump([k for k in keys if k in old] + [k for k in keys if k not in old][:4], root / "data" / "score_subset.json")
    h1 = ROOT.parent / "factindic_h1"
    if h1.exists():
        shutil.copytree(h1, root.parent / "factindic_h1", ignore=shutil.ignore_patterns("*sft*", "mt_cache*"), dirs_exist_ok=True)
    jobs = plan(smoke=True) + [Job("final", P + "auto:final", after=("bertscore",), cpu=True)]
    st = run_plan(jobs, root, SMOKE_CFG, t0, n_gpu, timeout_min=25 if IS_KAGGLE else 60, retries=0, label="smoke ")
    bad = [k for k, v in st.items() if v != "ok"]
    print("SMOKE TEST:", "all jobs passed" if not bad else f"problems in {bad}", flush=True)
    print(_tail(root / "logs" / "auto_final.0.log", 12))
    core = [j.name for j in jobs if j.core and st.get(j.name) != "ok"]
    if core:
        raise RuntimeError(f"smoke test: essential jobs failed: {core}. Send the messages above before starting the long run.")
    return st


START = time.time()


def _gpus():
    import torch
    return max(1, torch.cuda.device_count())


def part1_data():
    """Notebook 01: sanitation, testbed, dev, natural code-mixed set, anchors, training pool, score subset."""
    st = run_plan([Job("prep", P + "auto:prep", core=True)], ROOT, {}, START, _gpus(), timeout_min=100 if IS_KAGGLE else 240)
    keep = ("ILSUM san", "testbed/", "dev/", "natural_cm/", "pairs to score", "scores reused", "roman_vocab")
    log = (DIRS["logs"] / "auto_prep.0.log").read_text(errors="ignore").replace("\r", "\n")
    print("\n".join(x for x in log.splitlines() if any(k in x for k in keep)))
    if st["prep"] != "ok":
        raise RuntimeError("data preparation failed")
    for name in ("testbed", "dev", "natural_cm", "train_pool"):
        n = {}
        for r in jl_read_glob(DIRS["data"] / f"{name}.jsonl"):
            n[r["lang"]] = n.get(r["lang"], 0) + 1
        print(f"{name}: {n}")


def part2_smoke():
    smoke(START, _gpus())
    shutil.rmtree(SMOKE, ignore_errors=True)


def part3_full(soft_h=6.25, hard_h=8.25):
    """Notebook 01 scores + 02-06 + analysis. Hours count from the import of this module."""
    t0, n = START, _gpus()
    st = run_plan(plan(smoke=False), ROOT, {}, t0, n, soft_h, hard_h)
    f = run_plan([Job("final", P + "auto:final", cpu=True)], ROOT, {}, t0, n, timeout_min=40)
    print(_tail(DIRS["logs"] / "auto_final.0.log", 12))
    free_gpu()
    return dict(st, **f)
