"""Convert a `# %%` percent-format script into a Jupyter notebook (no execution)."""
import json, sys, re
src, dst = sys.argv[1], sys.argv[2]
cells, cur, kind = [], [], "code"
def flush():
    if any(l.strip() for l in cur):
        body = cur[:]
        if kind == "markdown":
            body = [re.sub(r"^# ?", "", l) for l in body]
        while body and not body[-1].strip(): body.pop()
        cells.append(dict(cell_type=kind, metadata={}, source="\n".join(body),
                          **({"outputs": [], "execution_count": None} if kind == "code" else {})))
for line in open(src, encoding="utf-8").read().split("\n"):
    m = re.match(r"^# %%(.*)$", line)
    if m:
        flush(); cur = []
        kind = "markdown" if "[markdown]" in m.group(1) else "code"
        if kind == "code" and m.group(1).strip():
            cur.append(f"# {m.group(1).strip()}")
        continue
    cur.append(line)
flush()
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                    "accelerator": "GPU", "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}})
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(cells)} cells -> {dst}")
