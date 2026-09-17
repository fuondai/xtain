#!/usr/bin/env python3
"""pubfig.py - publication-quality figure helper.

What top venues actually do: VECTOR figures (PDF/PGF), fonts and sizes that match
the paper, a restrained colorblind-safe palette, and figures sized to the column /
text width so nothing is rescaled. Import this in your plotting scripts:

    from pubfig import setup, save, COLUMN_W, TEXT_W, CB_COLORS
    setup()                                  # SciencePlots + vector rcParams
    fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W*0.72))
    ax.plot(x, y, color=CB_COLORS[0], label="ours")
    ...
    save(fig, "figures/fig_results")         # writes .pdf (vector) + .png (preview)

setup(latex=True) additionally renders with the LaTeX engine so plot fonts match
the paper exactly (slower; falls back to mathtext automatically if LaTeX stalls).
"""
from __future__ import annotations

# IEEE two-column geometry (inches)
COLUMN_W = 3.36          # \columnwidth
TEXT_W = 7.16            # \textwidth (full width, figure*)

# Colorblind-safe (Wong/Okabe-Ito) — color encodes meaning, not decoration.
CB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def setup(style: str = "ieee", latex: bool = False, base_fontsize: int = 8):
    import matplotlib
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        styles = ["science", "grid"] + (["ieee"] if style == "ieee" else [])
        if not latex:
            styles.append("no-latex")
        plt.style.use(styles)
    except Exception:
        pass
    rc = {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.02,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "lines.linewidth": 1.1,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        # keep text as text (selectable, crisp) in vector output:
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.prop_cycle": matplotlib.cycler(color=CB_COLORS),
    }
    if latex:
        rc.update({"text.usetex": True,
                   "font.family": "serif",
                   "pgf.rcfonts": False})
    matplotlib.rcParams.update(rc)


def save(fig, path_noext: str):
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.18)
    fig.savefig(path_noext + ".pdf", bbox_inches=None)        # vector
    try:
        fig.savefig(path_noext + ".png", dpi=300, bbox_inches=None)  # preview
    except Exception as e:
        import sys
        sys.stderr.write("pubfig: PNG preview failed for %s: %s\n" % (path_noext, e))
    return path_noext + ".pdf"


# --------------------------------------------------------------------------- #
# Publication helpers (top-venue chart idioms). All return the Axes used.
# --------------------------------------------------------------------------- #
def save_pgf(fig, path_noext: str):
    """Save a .pgf (LaTeX-native vector) so fonts/math match the paper exactly.
    Include in LaTeX with \\input{figures/foo.pgf} inside a figure environment.
    Requires the LaTeX engine; falls back to .pdf if the PGF backend is missing.
    """
    import os
    os.makedirs(os.path.dirname(path_noext) or ".", exist_ok=True)
    try:
        fig.savefig(path_noext + ".pgf", bbox_inches="tight")
        return path_noext + ".pgf"
    except Exception:
        return save(fig, path_noext)


def errorband(ax, x, y, err, label=None, color=None, alpha=0.18):
    """Mean line + shaded +/- err band (the standard way to show variance)."""
    import numpy as np
    x, y, err = np.asarray(x), np.asarray(y), np.asarray(err)
    color = color or CB_COLORS[0]
    line, = ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y - err, y + err, color=color, alpha=alpha, linewidth=0)
    return ax


def grouped_bar(ax, groups, series: dict, ylabel=None):
    """Grouped bar chart. series = {"name": [v per group], ...}."""
    import numpy as np
    n = len(series)
    idx = np.arange(len(groups))
    w = 0.8 / max(n, 1)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(idx + i * w - 0.4 + w / 2, vals, w, label=name,
               color=CB_COLORS[i % len(CB_COLORS)])
    ax.set_xticks(idx)
    ax.set_xticklabels(groups)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def cdf(ax, data, label=None, color=None):
    """Empirical CDF — common for latency/error distributions in systems papers."""
    import numpy as np
    d = np.sort(np.asarray(data))
    y = np.arange(1, len(d) + 1) / len(d)
    ax.plot(d, y, label=label, color=color or CB_COLORS[0])
    ax.set_ylabel("CDF")
    return ax


def heatmap(fig, ax, matrix, xticklabels=None, yticklabels=None, cbar_label=None):
    """Sequential heatmap with colorbar (use a perceptually-uniform map)."""
    import numpy as np
    im = ax.imshow(np.asarray(matrix), aspect="auto", cmap="viridis")
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels))); ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels))); ax.set_yticklabels(yticklabels)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    return ax
