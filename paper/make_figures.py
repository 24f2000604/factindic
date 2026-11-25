"""Figures of the paper from paper/numbers.json: python paper/make_figures.py → paper/figures/*.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "numbers.json").read_text())
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8880"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 8, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": .6, "savefig.dpi": 300})


def pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 1.75))
    ax.set(xlim=(-.6, 100.6), ylim=(0, 24)); ax.axis("off")
    boxes = [("Document D", "native, script-mixed\nor Romanized\nHindi / Gujarati"),
             ("1  Anchors", "normalize, language ID,\nback-transliterate;\nentities, numbers, dates"),
             ("2  Prompt B1", "Gemma-4 E2B with\nanchor checklist\n(chosen on dev)"),
             ("3  Decode", "Anchor-CAD + DoLa\n(1 candidate) and\n4 nucleus samples"),
             ("4  Verify, rerank", "anchors survive MT?\nyes: MiniCheck (En.)\nno: native mDeBERTa"),
             ("Summary S*", "best J(S), then one\nprune pass; native\nmatrix-language script")]
    w, gap = 14.2, 2.96
    for i, (t, s) in enumerate(boxes):
        x = i * (w + gap)
        ends = i in (0, 5)
        ax.add_patch(FancyBboxPatch((x + .3, 3), w - .6, 18, boxstyle="round,pad=0.25,rounding_size=.8", lw=.9,
                                    ec=GRAY if ends else BLUE, fc="#f6f5f2" if ends else "#eef4fc"))
        ax.text(x + w / 2, 16.8, t, ha="center", va="center", weight="bold", fontsize=8)
        ax.text(x + w / 2, 9.8, s, ha="center", va="center", fontsize=6.6, color=INK2, linespacing=1.25)
        if i < 5:
            ax.annotate("", (x + w + gap - .35, 12), (x + w + .35, 12), arrowprops=dict(arrowstyle="-|>", lw=.9, color=INK2, mutation_scale=8))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(HERE / "figures/pipeline.png")


def h2():
    S = [("gemma_A", "Gemma A"), ("gemma_B1", "Gemma B1"), ("factindic", "FactIndic")]
    V = [("native", "Native", BLUE, "o"), ("script_mixed", "Script-mixed", ORANGE, "s"), ("roman", "Romanized", GREEN, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), sharey=True)
    for ax, l, t in zip(axes, ("hi", "gu"), ("Hindi", "Gujarati")):
        for i, (s, _) in enumerate(S):
            for j, (v, lab, c, m) in enumerate(V):
                d = N["variants"][s][f"{l}/{v}"]
                x = i + (j - 1) * .24
                ax.plot([x, x], [d["lo"], d["hi"]], color=c, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(x, d["mean"], m, color=c, ms=4.6, mec="white", mew=.7, zorder=3, label=lab if i == 0 else None)
        ax.set_xticks(range(3), [n.replace(" ", "\n") for _, n in S], fontsize=7)
        ax.set_title(t, fontsize=8.5, weight="bold"); ax.set_ylim(.2, 1.02); ax.grid(axis="y"); ax.tick_params(length=0)
    axes[0].set_ylabel("Claim precision (FactCG)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7, handletextpad=.2, columnspacing=1)
    fig.subplots_adjust(.13, .27, .995, .9, wspace=.06)
    fig.savefig(HERE / "figures/h2.png")


def ladder():
    S = [("mt5", "mT5-base"), ("indicbartss", "IndicBARTSS"), ("llama_A", "Llama-3.2-3B, A"), ("gemma_sft", "Gemma LoRA-SFT"),
         ("gemma_A", "Gemma A (standard)"), ("gemma_B3", "Gemma B3 (verb hint)"), ("gemma_B2", "Gemma B2 (CoD)"), ("gemma_B1", "Gemma B1 (anchors)"),
         ("dec_anchor_cad_dola", "B1 + Anchor-CAD + DoLa"), ("factindic", "FactIndic (full)")]
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for y, (s, lab) in enumerate(reversed(S)):
        d = N["systems"][s]["P"]
        c = ORANGE if s == "factindic" else BLUE
        ax.plot([d["lo"], d["hi"]], [y, y], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(d["mean"], y, "o", color=c, ms=4.6, mec="white", mew=.7)
        ax.text(d["hi"] + .008, y, f"{d['mean']:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(S)), [x for _, x in reversed(S)], fontsize=7.2)
    ax.set_xlim(.38, .86); ax.grid(axis="x"); ax.tick_params(length=0)
    ax.set_xlabel("Claim precision (FactCG), 360 items, 95% CI")
    fig.subplots_adjust(.375, .135, .99, .99)
    fig.savefig(HERE / "figures/ladder.png")


for f in (pipeline, h2, ladder):
    f()
print("ok")
