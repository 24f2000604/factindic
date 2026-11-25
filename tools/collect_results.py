"""Gather a finished run into one folder: python tools/collect_results.py [results/final_run]

H1 is left out on purpose (it is finished and lives in h1_report/). The folder is rebuilt from runs/factindic each
time; REPORT.md, if somebody wrote one there, is kept."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "factindic"
sys.path.insert(0, str(REPO))


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge(name_dir, dst):
    """One jsonl per name instead of its shards."""
    dst.mkdir(parents=True, exist_ok=True)
    groups = {}
    for f in sorted((RUN / name_dir).glob("*.jsonl")):
        groups.setdefault(f.name.split(".")[0], []).append(f)
    for n, fs in groups.items():
        with open(dst / f"{n}.jsonl", "w", encoding="utf-8") as out:
            for f in fs:
                for line in open(f, encoding="utf-8"):
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")


def main():
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "results" / "final_run"
    for d in ("1_reports", "2_figures", "3_summaries", "4_evaluation", "5_models", "6_data", "7_logs"):
        shutil.rmtree(out / d, ignore_errors=True)
        (out / d).mkdir(parents=True)
    # reports: tables and test results, without H1
    res = json.loads((RUN / "eval" / "results.json").read_text(encoding="utf-8"))
    res.pop("h1", None)
    old = res.pop("h3", None) or {}                              # runs made before H3 was dropped
    if old.get("lookback"):
        res.setdefault("lookback", old["lookback"])
    (out / "1_reports" / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2).replace(str(REPO) + "/", ""), encoding="utf-8")
    copy(RUN / "eval" / "main_table.csv", out / "1_reports" / "main_table.csv")
    for f in (RUN / "outputs").glob("*.json"):           # dev scores, chosen base, decoding grid, latency
        copy(f, out / "1_reports" / "selection_and_latency" / f.name)
    for f in (RUN / "logs").glob("auto_status*.json"):
        copy(f, out / "1_reports" / "job_status.json")
    for f in (RUN / "figures").iterdir():
        if not f.name.startswith(("h1_", "h3_")):
            copy(f, out / "2_figures" / f.suffix.lstrip(".") / f.name)
    merge("outputs", out / "3_summaries")
    merge("eval", out / "4_evaluation" / "eval")
    merge("verify", out / "4_evaluation" / "verify")
    if (RUN / "models").exists():
        shutil.copytree(RUN / "models", out / "5_models", dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt"))
    for n in ("testbed.jsonl", "dev.jsonl", "natural_cm.jsonl", "sanitation_report.json", "VERSION"):
        copy(RUN / "data" / n, out / "6_data" / n)
    merge_w = out / "6_data" / "faithfulness_weights.jsonl"
    with open(merge_w, "w", encoding="utf-8") as o:
        for f in sorted((RUN / "data").glob("weights*.jsonl")):
            o.write(f.read_text(encoding="utf-8"))
    for f in (RUN / "logs").glob("*.log"):
        if not f.name.startswith("it2_"):
            copy(f, out / "7_logs" / f.name)
    if (REPO / "runs" / "full.log").exists():
        copy(REPO / "runs" / "full.log", out / "7_logs" / "full.log")
    # what produced it
    from factindic.config import CFG, MODELS
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    rev = {}
    for k, m in MODELS.items():
        ref = hub / ("models--" + m.replace("/", "--")) / "refs" / "main"
        if ref.exists():
            rev[k] = dict(model=m, revision=ref.read_text().strip())
    sh = lambda *c: subprocess.run(c, cwd=REPO, capture_output=True, text=True).stdout
    (out / "1_reports" / "environment.json").write_text(json.dumps(dict(
        pretrained_models=rev, config=CFG, git_commit=sh("git", "rev-parse", "HEAD").strip(),
        uncommitted_files=sh("git", "diff", "--name-only").split(),
        gpu=sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader").strip(),
        packages=sh("uv", "pip", "freeze", "--python", ".venv/bin/python").split("\n")), indent=1, default=str))
    (out / "1_reports" / "code_changes.diff").write_text(sh("git", "diff", "--", "factindic", "requirements-local.txt"), encoding="utf-8")
    with open(out / "CHECKSUMS.md5", "w") as c:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name != "CHECKSUMS.md5":
                h = hashlib.md5()
                with open(f, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        h.update(b)
                c.write(f"{h.hexdigest()}  {f.relative_to(out)}\n")
    print(out)


if __name__ == "__main__":
    main()
