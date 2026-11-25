"""Figures that test H1–H4 and justify each pipeline stage. Every figure is saved as PNG + SVG
with its data as CSV (the table view). Style: validated categorical palette in fixed slot order,
colour follows the entity (never its rank), one y-axis per chart, recessive grid, direct labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIRS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9a9890"
# fixed entity → colour (never re-assigned when a filter drops entities)
ENTITY = {"gemma": SLOTS[0], "sarvam": SLOTS[1], "llama": SLOTS[2], "indicbartss": SLOTS[3], "mt5": SLOTS[4],
          "native": SLOTS[0], "script_mixed": SLOTS[1], "roman": SLOTS[2], "natural_cm": SLOTS[3],
          "hi": SLOTS[0], "gu": SLOTS[1]}
VARIANT_LABEL = {"native": "Native", "script_mixed": "Script-mixed", "roman": "Romanized", "natural_cm": "Natural CM"}
LANG_LABEL = {"hi": "Hindi", "gu": "Gujarati"}


def color_of(name):
    for k, c in ENTITY.items():
        if name == k or name.startswith(k):
            return c
    return NEUTRAL


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9, "lines.linewidth": 2,
        "font.family": ["DejaVu Sans"],
    })


def save(fig, name, data: pd.DataFrame | None = None, caption=""):
    DIRS["figures"].mkdir(parents=True, exist_ok=True)
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.savefig(DIRS["figures"] / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(DIRS["figures"] / f"{name}.svg", bbox_inches="tight")
    if data is not None:
        data.to_csv(DIRS["figures"] / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return DIRS["figures"] / f"{name}.png"


def _bar(ax, x, h, color, width, label=None):
    """Thin bars with a 2px surface gap between neighbours."""
    return ax.bar(x, h, width=width, color=color, label=label, edgecolor=SURFACE, linewidth=1.5, zorder=2)


# ----------------------------------------------------------------------------------------- H1
def h1_fertility(fert: pd.DataFrame):
    """Grouped bars: tokens/word by tokenizer for each (language, variant)."""
    import matplotlib.pyplot as plt
    style()
    toks = [t for t in ["sarvam1_sft", "gemma4_e2b", "llama32_3b", "indicbartss"] if t in set(fert.tokenizer)]
    variants = [v for v in ["native", "script_mixed", "roman"] if v in set(fert.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    w = 0.8 / max(1, len(toks))
    for ax, lang in zip(axes, ["hi", "gu"]):
        for i, t in enumerate(toks):
            vals = [fert[(fert.tokenizer == t) & (fert.lang == lang) & (fert.variant == v)].fertility_corpus.mean()
                    for v in variants]
            xs = np.arange(len(variants)) + (i - (len(toks) - 1) / 2) * w
            _bar(ax, xs, vals, color_of(t), w, label=t)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(variants)), [VARIANT_LABEL[v] for v in variants])
        ax.set_title(LANG_LABEL[lang])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Tokenizer fertility (tokens / word)")
    axes[1].legend(loc="upper left", ncols=2)
    fig.suptitle("Romanization raises tokens per word for Sarvam-1 and IndicBARTSS but lowers it for Llama-3.2", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    return save(fig, "h1_fertility", fert)


def h1_shift(per_summary: pd.DataFrame):
    """Arrow per (model, language): native → Romanized in (fertility, hallucination) space.
    H1 predicts arrows point up-and-right; a leftward-up arrow is a counterexample."""
    import matplotlib.pyplot as plt
    style()
    g = (per_summary.groupby(["model", "lang", "script"])
         .agg(F=("input_fertility", "mean"), H=("halluc", "mean")).reset_index())
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rows = []
    for (m, lang), d in g.groupby(["model", "lang"]):
        d = d.set_index("script")
        if not {"native", "roman"} <= set(d.index):
            continue
        c = color_of(m)
        x0, y0, x1, y1 = d.loc["native", "F"], d.loc["native", "H"], d.loc["roman", "F"], d.loc["roman", "H"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2, shrinkA=4, shrinkB=4))
        ax.scatter([x0], [y0], s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([x1], [y1], s=64, color=SURFACE, edgecolor=c, linewidth=2, zorder=3)
        ax.text(x1 + 0.03, y1, f"{m} · {LANG_LABEL[lang]}", fontsize=8, color=INK, va="center")
        rows.append(dict(model=m, lang=lang, F_native=x0, H_native=y0, F_roman=x1, H_roman=y1))
    ax.set_xlabel("Input fertility (tokens / word)")
    ax.set_ylabel("Claim hallucination rate (1 − P_fact)")
    ax.set_title("Native ● → Romanized ○: hallucination rises even where fertility falls", loc="left")
    fig.tight_layout()
    return save(fig, "h1_shift", pd.DataFrame(rows),
                caption="Each arrow joins the same 50 articles fed in native script and Romanized.")


def coef_plot(tests: dict, outcome_labels: dict, name, title, xlabel="β on input fertility (95% CI)"):
    """Dot + CI per (spec, outcome). tests[outcome][spec] = {beta, se, p_one_pos}."""
    import matplotlib.pyplot as plt
    style()
    rows = []
    for o, lab in outcome_labels.items():
        for spec, r in tests.get(o, {}).items():
            if "beta" in r:
                rows.append(dict(outcome=lab, spec=spec, beta=r["beta"], lo=r["beta"] - 1.96 * r["se"],
                                 hi=r["beta"] + 1.96 * r["se"], p=r.get("p_one_pos", r.get("p_one"))))
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(df) + 1.2))
    specs = list(dict.fromkeys(df.spec))
    for i, r in enumerate(df.itertuples()):
        c = SLOTS[specs.index(r.spec) % len(SLOTS)]
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r.beta], [i], s=40, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(max(r.hi, r.beta) + 0.01, i, f"p={r.p:.3f}" if r.p == r.p else "", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(df)), [f"{r.outcome} · {r.spec.replace('_', ' ')}" for r in df.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, name, df)


# ----------------------------------------------------------------------------------------- generic
def dot_ci(df, name, title, xlabel, label_col="label", highlight=None, ref_line=None, caption=""):
    """Single-series horizontal dot plot with 95% CIs; one colour, optional highlighted row."""
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(df) + 1.3))
    for i, r in enumerate(df.itertuples()):
        c = SLOTS[1] if highlight and getattr(r, label_col) in highlight else SLOTS[0]
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, solid_capstyle="round", alpha=0.9)
        ax.scatter([r.mean], [i], s=44, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(r.hi, i, f"  {r.mean:.3f}", va="center", fontsize=7.5, color=INK2)
    if ref_line is not None:
        ax.axvline(ref_line, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(df)), list(df[label_col]))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, name, df, caption)


def lines_by_variant(df, name, title, ylabel, series_col="system", caption=""):
    """Small multiples per language: x = variant, y = metric (mean, lo, hi), one line per series (≤4)."""
    import matplotlib.pyplot as plt
    style()
    series = list(dict.fromkeys(df[series_col]))[:4]
    variants = [v for v in ["native", "script_mixed", "roman", "natural_cm"] if v in set(df.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, lang in zip(axes, ["hi", "gu"]):
        for i, s in enumerate(series):
            d = df[(df[series_col] == s) & (df.lang == lang)].set_index("variant").reindex(variants)
            xs = np.arange(len(variants))
            c = SLOTS[i]
            ax.fill_between(xs, d["lo"], d["hi"], color=c, alpha=0.12, lw=0)
            ax.plot(xs, d["mean"], color=c, marker="o", ms=6, mec=SURFACE, mew=1.5, label=s)
            if len(series) <= 4 and not d["mean"].isna().all():
                ax.text(xs[-1] + 0.08, d["mean"].iloc[-1], s, fontsize=7.5, color=INK, va="center")
        ax.set_xticks(range(len(variants)), [VARIANT_LABEL[v] for v in variants])
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlim(-0.3, len(variants) - 0.2 + 0.9)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    return save(fig, name, df, caption)


def grouped_bars(df, name, title, ylabel, x_col, group_col, val_col, err=None, order_x=None, order_g=None,
                 color_fn=None, caption="", pct=False):
    import matplotlib.pyplot as plt
    style()
    xs_vals = order_x or list(dict.fromkeys(df[x_col]))
    groups = order_g or list(dict.fromkeys(df[group_col]))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(xs_vals) * max(1, len(groups)) * 0.55 + 2), 3.8))
    w = 0.8 / max(1, len(groups))
    for i, g in enumerate(groups):
        d = df[df[group_col] == g].set_index(x_col).reindex(xs_vals)
        xs = np.arange(len(xs_vals)) + (i - (len(groups) - 1) / 2) * w
        c = color_fn(g) if color_fn else SLOTS[i % len(SLOTS)]
        _bar(ax, xs, d[val_col], c, w, label=str(g))
        if err:
            lo, hi = d[err[0]], d[err[1]]
            ax.errorbar(xs, d[val_col], yerr=[d[val_col] - lo, hi - d[val_col]], fmt="none",
                        ecolor=INK2, elinewidth=1, capsize=0, zorder=3)
        for x, v in zip(xs, d[val_col]):
            if v == v:
                ax.text(x, v, f"{v:.0%}" if pct else f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(xs_vals)), [VARIANT_LABEL.get(x, LANG_LABEL.get(x, x)) for x in xs_vals])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    if len(groups) > 1:
        ax.legend(loc="upper left", ncols=min(4, len(groups)))
    fig.tight_layout()
    return save(fig, name, df, caption)


def pr_scatter(df, name, title, caption=""):
    """Precision vs recall per system (≤3 colours: pipeline / baselines / repair)."""
    import matplotlib.pyplot as plt
    style()
    kind_color = {"FactIndic": SLOTS[1], "baseline": SLOTS[0], "repair": SLOTS[2]}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in df.itertuples():
        c = kind_color.get(r.kind, NEUTRAL)
        ax.plot([r.R_lo, r.R_hi], [r.P, r.P], color=c, lw=1, alpha=0.6)
        ax.plot([r.R, r.R], [r.P_lo, r.P_hi], color=c, lw=1, alpha=0.6)
        ax.scatter([r.R], [r.P], s=56, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(r.R + 0.006, r.P + 0.006, r.system, fontsize=7.5, color=INK)
    for k, c in kind_color.items():
        ax.scatter([], [], color=c, s=40, label=k)
    ax.legend(loc="lower left")
    ax.set_xlabel("Claim-level factual recall R_fact")
    ax.set_ylabel("Claim-level factual precision P_fact")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return save(fig, name, df, caption)


def roc(curves: dict, name, title, caption=""):
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    rows = []
    for i, (lab, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=SLOTS[i], label=f"{lab} (AUROC {auc:.2f})")
        rows += [dict(curve=lab, fpr=a, tpr=b) for a, b in zip(fpr, tpr)]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, name, pd.DataFrame(rows), caption)


def histogram(values, name, title, xlabel, vline=None, vlabel="", caption=""):
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (lab, v) in enumerate(values.items()):
        ax.hist(v, bins=30, range=(0, 1), color=SLOTS[i], alpha=0.55, label=lab, edgecolor=SURFACE, linewidth=1)
    if vline is not None:
        ax.axvline(vline, color=INK, lw=1, ls=(0, (3, 3)))
        ax.text(vline, ax.get_ylim()[1] * 0.95, f" {vlabel}", fontsize=8, color=INK, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training pairs")
    ax.set_title(title, loc="left")
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    data = pd.DataFrame([dict(group=k, value=x) for k, v in values.items() for x in v])
    return save(fig, name, data, caption)
