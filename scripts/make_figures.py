#!/usr/bin/env python3
"""Paper figures (PDF for the manuscript, PNG for review), drawn only from committed results files.

  fig_anytime            one READER read at every deadline vs banks of separately retrained specialists
  fig_exact_duration     the SAME trained checkpoints given only the first part of each test trial
  fig_reader_vs_zoo      READER minus every baseline re-run here, subject-level mean and 95% CI
  fig_landscape          accuracy against parameter count, every model, every corpus
  fig_subject_dumbbell   READER against BiTE one subject at a time
  fig_subject_heatmap    every model against every subject
  fig_confusion          which classes READER confuses
  fig_complementarity    whose errors are whose, and the oracle ceiling over the two
  fig_calibration        reliability: can the confidence be believed?
Every figure has a table of the same numbers in results/ (named in results/figures/README.md).

Colour follows the model in every figure (validated categorical order, light surface): READER blue,
Compact orange, BiTE aqua, FF-Control yellow, Compact-Mean magenta. Marker shape repeats the identity so
the figures survive grayscale printing and colour-vision deficiency.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
COLOR = {"reader": "#2a78d6", "compact": "#eb6834", "bite": "#1baf7a", "ff_control": "#eda100",
         "compact_mean": "#e87ba4"}
MARKER = {"reader": "o", "compact": "s", "bite": "^", "ff_control": "D", "compact_mean": "v"}
LABEL = {"reader": "READER", "compact": "Compact", "bite": "BiTE", "ff_control": "FF-Control",
         "compact_mean": "Compact-Mean"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TITLE = {"2a": "BCI IV-2a", "2b": "BCI IV-2b", "hgd": "HGD", "sdssvep": "SD-SSVEP"}
FS = {"2a": 250.0, "2b": 250.0, "sdssvep": 256.0}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def subject_curve(rows, value="acc"):
    """{(sub, seed): v} -> mean over subjects of the seed mean, and SE across subjects."""
    by = defaultdict(list)
    for (sub, _), v in rows.items():
        by[sub].append(v)
    means = np.array([np.mean(v) for v in by.values()])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("->", (OUT / name).relative_to(ROOT), "(.pdf, .png)")


def fig_anytime():
    curves = defaultdict(lambda: defaultdict(dict))       # (arm, ds) -> ms -> {(sub, seed): acc}
    for r in csv.DictReader(open(RES / "raw/anytime_curves.csv")):
        if r["protocol"] != "within":
            continue
        curves[(r["arm"], r["dataset"])][float(r["ends_at_ms"])][(int(r["subject"]), int(r["seed"]))] = float(r["test_acc"])
    bank = defaultdict(lambda: defaultdict(dict))         # (family, ds) -> seconds -> {(sub, seed): acc}
    for r in csv.DictReader(open(RES / "raw/runs.csv")):
        key = (int(r["subject"]), int(r["seed"]))
        if r["group"] == "bank":
            fam, _, tag = r["arm"].rpartition("_")
            bank[(fam, r["dataset"])][float(tag[:-1])][key] = float(r["test_acc"])
        elif r["group"] == "cohort" and r["arm"] in ("bite", "compact"):
            full = 1.0 if r["dataset"] == "sdssvep" else 4.0
            bank[(r["arm"], r["dataset"])][full][key] = float(r["test_acc"])

    corpora = ["2a", "2b", "sdssvep"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=False)
    for ax, ds in zip(axes, corpora):
        for arm, style, lw, name in (("reader_anytime", "-", 1.6, "READER + prefix supervision (1 model)"),
                                     ("reader", "--", 1.0, "READER, endpoint loss (1 model)")):
            c = curves.get((arm, ds))
            if not c:
                continue
            xs = sorted(c)
            ys = [subject_curve(c[x])[0] for x in xs]
            ax.plot(np.array(xs) / 1000, ys, style, color=COLOR["reader"], lw=lw, label=name, zorder=3)
        for fam in ("bite", "compact"):
            b = bank.get((fam, ds))
            if not b or len(b) < 2:
                continue
            xs = sorted(b)
            stats = [subject_curve(b[x]) for x in xs]
            ax.errorbar(xs, [s[0] for s in stats], yerr=[s[1] for s in stats], fmt=MARKER[fam], color=COLOR[fam],
                        ms=5, lw=0.8, capsize=2, mec="white", mew=0.8, zorder=4,
                        label=f"{LABEL[fam]} specialists (1 model per deadline)")
        ax.set_title(TITLE[ds] + ("" if (("bite", ds) in bank and len(bank[("bite", ds)]) > 1) else "  (bank pending)"))
        ax.set_xlabel("decision deadline (s)")
        ax.set_xlim(0, 1.02 if ds == "sdssvep" else 4.05)
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    save(fig, "fig_anytime")


def subject_band(rows, draws=10000, seed=0):
    """Mean over subjects of the seed mean, with a 95% percentile bootstrap interval over subjects.

    Subjects are the unit, seeds are averaged inside a subject first -- the paper's convention
    everywhere else, so the band here means what the intervals in the tables mean.
    """
    by = defaultdict(list)
    for (sub, _), v in rows.items():
        by[sub].append(v)
    means = np.array([np.mean(v) for v in by.values()])
    if len(means) < 2:
        return float(means.mean()), float(means.mean()), float(means.mean())
    draw = np.random.default_rng(seed).choice(means, (draws, len(means)), replace=True).mean(axis=1)
    return float(means.mean()), float(np.percentile(draw, 2.5)), float(np.percentile(draw, 97.5))


# The four arms this figure contrasts, drawn controls-first so READER lands on top, and named the way
# the manuscript names them rather than the way the harness does.
EXACT_ARMS = [("compact_mean", "Fwd mean", 1.2, "-"), ("ff_control", "Two forward", 1.2, (0, (3.5, 1.6))),
              ("compact", "No reverse", 1.4, "-"), ("reader", "REACT", 2.5, "-")]
HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]
CHANCE = {"2a": 25.0, "2b": 50.0, "sdssvep": 100 / 12}
NCLASS = {"2a": 4, "2b": 2, "sdssvep": 12}
# The early deadline each corpus's headline number is quoted at; the gap is annotated there.
DEADLINE = {"2a": 250, "2b": 250, "sdssvep": 64}


def fig_exact_duration():
    """One checkpoint per model, trained once at the full trial, then re-read on truncated input.

    The endpoint is the right-hand end of every curve, so this figure and the endpoint table are the
    same numbers from the same runs -- the last point of READER is its within-subject accuracy.
    """
    recs = json.loads((RES / "exact_duration.json").read_text())["records"]
    acc = defaultdict(lambda: defaultdict(dict))           # (arm, ds) -> n -> {(sub, seed): acc}
    for r in recs:
        for n, a in r["acc"].items():
            acc[(r["arm"], r["dataset"])][int(n)][(int(r["subject"]), int(r["seed"]))] = 100 * float(a)

    corpora = ["2a", "2b", "sdssvep"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
    for ax, ds in zip(axes, corpora):
        ax.axhline(CHANCE[ds], color=MUTED, lw=0.8, ls=(0, (1, 2.2)), zorder=1)
        ax.annotate("chance", xy=(0.90, CHANCE[ds]), xycoords=("axes fraction", "data"),
                    va="bottom", ha="right", fontsize=6, color=MUTED, zorder=3, path_effects=HALO)
        for arm, label, lw, ls in EXACT_ARMS:
            c = acc.get((arm, ds))
            if not c:
                continue
            ns = sorted(c)
            xs = np.array(ns) / FS[ds]
            band = [subject_band(c[n]) for n in ns]
            ys, lo, hi = (np.array([b[i] for b in band]) for i in range(3))
            lead = arm == "reader"
            ax.fill_between(xs, lo, hi, color=COLOR[arm], alpha=0.18 if lead else 0.09, lw=0, zorder=2)
            ax.plot(xs, ys, ls=ls, color=COLOR[arm], lw=lw, marker=MARKER[arm],
                    ms=3.4 if lead else 2.6, mec="white", mew=0.5,
                    label=label, zorder=6 if lead else 4, alpha=1.0 if lead else 0.85)

        # The gap the paper quotes, drawn where it is quoted: READER against its matched control.
        n = DEADLINE[ds]
        top = subject_band(acc[("reader", ds)][n])[0]
        bot = subject_band(acc[("compact", ds)][n])[0]
        x = n / FS[ds]
        ax.annotate("", xy=(x, top), xytext=(x, bot), zorder=7,
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=1.5, shrinkB=1.5))
        ax.annotate(f"{top - bot:+.1f} pts\nat {x:g} s", xy=(x, (top + bot) / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=6.5, color=INK,
                    zorder=8, path_effects=HALO)

        ax.set_title(f"{TITLE[ds]}   ({NCLASS[ds]} classes)", pad=6)
        ax.set_ylim(0, 100)
        grid = np.array(sorted(acc[("reader", ds)])) / FS[ds]
        pad = 0.6 * (grid[1] - grid[0])
        ax.set_xlim(grid[0] - pad, grid[-1] + pad)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0] if ds == "sdssvep" else [1, 2, 3, 4])
        ax.tick_params(length=2)
    axes[0].set_ylabel("test accuracy (%)")
    h, l = axes[-1].get_legend_handles_labels()
    order = [l.index(x) for x in ("REACT", "No reverse", "Two forward", "Fwd mean") if x in l]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.10), columnspacing=1.6, handlelength=1.9)
    fig.supxlabel("EEG observed before the decision (s)", fontsize=8, color=INK, y=-0.04)
    fig.subplots_adjust(wspace=0.09)
    save(fig, "fig_exact_duration")


def fig_reader_vs_zoo():
    report = json.loads((RES / "main_table.json").read_text())
    corpora = ["2a", "2b", "hgd", "sdssvep"]
    names = sorted({k for ds in corpora for k in report["corpora"][ds]["reader_minus"]} |
                   {k for k in report["models"] if k not in ("**READER**",) and not k.startswith("Compact")})
    names = ["BiTE"] + sorted(n for n in names if n != "BiTE")
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, ds in zip(axes, corpora):
        fam = report["corpora"][ds]["reader_minus"]
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for yi, n in zip(y, names):
            r = fam.get(n)
            if r is None:
                ran = (report["table"].get(n, {}).get(ds) or {}).get("mean") is not None
                ax.text(0.03, yi, "pending" if ran else "not run", fontsize=6, color=MUTED, ha="left", va="center",
                        transform=ax.get_yaxis_transform())
                continue
            lo, hi = r["ci95"]
            ax.plot([lo, hi], [yi, yi], color=COLOR["reader"], lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot(r["mean"], yi, "o", color=COLOR["reader"], ms=4.5, mec="white", mew=0.6, zorder=3)
        ax.set_title(TITLE[ds])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    fig.supxlabel("READER minus model, subject-level mean and 95% CI (percentage points; right of 0 favours READER)",
                  fontsize=8, y=-0.04)
    save(fig, "fig_reader_vs_zoo")


def load_runs(group=None, protocol="within"):
    rows = [r for r in csv.DictReader(open(RES / "raw/runs.csv")) if r["protocol"] == protocol]
    return [r for r in rows if group is None or r["group"] == group]


def acc_by_subject(rows, arm_field="arm"):
    """rows -> {(arm, dataset): {subject: seed-mean accuracy}}, the main table's statistic."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r[arm_field], r["dataset"])][int(r["subject"])].append(float(r["test_acc"]))
    return {k: {s: float(np.mean(v)) for s, v in sorted(d.items())} for k, d in by.items()}


def named_models():
    """Every model in the within-subject comparison, with a display name and its runs.csv key."""
    return ([("READER", "cohort", "reader"), ("Compact", "cohort", "compact"), ("BiTE", "cohort", "bite")] +
            [(m, "zoo", m) for m in sorted({r["arm"] for r in load_runs("zoo")})])


def declutter(fig, texts, obstacles=(), x_pad=1.0, y_pad=1.0, rounds=80):
    """Push overlapping labels apart vertically, in display space.

    Scatter labels collide whenever two models are close in both accuracy and size -- which is exactly
    where this figure is interesting, so the collisions cannot be avoided by choosing a nicer layout.
    Each round measures the drawn boxes and separates any overlapping pair along y by half their
    overlap, which converges in a few dozen rounds for the label counts here.

    `obstacles` are (axes, x, y, radius_in_points) markers that do not move: a label pushed clear of
    every other label can still be sitting on top of somebody else's data point.
    """
    base = {t: t.axes.transData for t in texts}
    # A label starts above its marker, or below it when the marker is near the top of its axes: on a
    # corpus where everything good is bunched against the ceiling there is no room above, and pushing
    # up regardless drives the labels out of the panel.
    offsets = {t: (-3.0 if t._below else 3.0) for t in texts}
    for t in texts:
        t.set_va("top" if t._below else "bottom")

    def apply():
        for t in texts:
            t.set_transform(base[t] + transforms.ScaledTranslation(
                0, offsets[t] / 72.0, t.figure.dpi_scale_trans))

    apply()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(rounds):
        boxes = {t: t.get_window_extent(renderer).expanded(1 + x_pad / 30, 1 + y_pad / 30) for t in texts}
        moved = False
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                if a.axes is not b.axes:
                    continue
                ba, bb = boxes[a], boxes[b]
                if not ba.overlaps(bb):
                    continue
                overlap = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
                step = (overlap / 2 + 0.5) * 72.0 / fig.dpi
                up, down = (a, b) if ba.y0 >= bb.y0 else (b, a)
                offsets[up] += step
                offsets[down] -= step
                moved = True
        for t in texts:
            box = boxes[t]
            for ax, x, y, radius in obstacles:
                if ax is not t.axes:
                    continue
                cx, cy = ax.transData.transform((x, y))
                r = radius * fig.dpi / 72.0
                if box.x0 - r < cx < box.x1 + r and box.y0 - r < cy < box.y1 + r:
                    # Move the label further along the side it was assigned, never back across the point.
                    push = (box.y1 - (cy - r)) if t._below else ((cy + r) - box.y0)
                    offsets[t] += (-push if t._below else push) * 72.0 / fig.dpi
                    moved = True
        if not moved:
            break
        apply()
        fig.canvas.draw()

        # Keep every label inside its own panel: the pairwise push has no bound of its own, and a label
        # driven out of the axes is worse than one that still overlaps a little.
        for t in texts:
            frame = t.axes.get_window_extent(renderer)
            box = t.get_window_extent(renderer)
            if box.y1 > frame.y1:
                offsets[t] -= (box.y1 - frame.y1) * 72.0 / fig.dpi
            elif box.y0 < frame.y0:
                offsets[t] += (frame.y0 - box.y0) * 72.0 / fig.dpi
        apply()
        fig.canvas.draw()
    return offsets


def fig_landscape():
    """Accuracy against parameter count. The question a reader asks first: is it big or is it good?

    The ten baselines are numbered rather than named on the plot. Their names are long and several of
    them land on nearly the same point on more than one corpus, so spelled-out labels either overlap
    each other or drift far enough from their marker to be ambiguous; a one- or two-character label
    stays next to its own point. The three models the paper is about are in the legend, with a distinct
    marker shape each so they are also separable in greyscale.
    """
    rows = load_runs()
    acc = acc_by_subject(rows)
    params = {(r["arm"], r["dataset"]): int(r["parameters"]) for r in rows}
    corpora = ["2a", "2b", "hgd", "sdssvep"]
    highlight = {"reader": ("READER", COLOR["reader"], "o", 70), "bite": ("BiTE", COLOR["bite"], "^", 44),
                 "compact": ("Compact", COLOR["compact"], "s", 44)}
    baselines = [name for name, _, key in named_models() if key not in highlight]
    number = {name: i + 1 for i, name in enumerate(baselines)}

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.3))
    for ax, ds in zip(axes.ravel(), corpora):
        pts = [(name, key, params.get((key, ds)), float(np.mean(list(acc[(key, ds)].values()))))
               for name, _, key in named_models() if (key, ds) in acc and (key, ds) in params]
        if not pts:
            continue
        lo, hi = min(v for *_, v in pts), max(v for *_, v in pts)
        labels, marks = [], []
        for name, key, p, a in pts:
            if key in highlight:
                continue
            ax.scatter(p, a, s=22, facecolor="none", edgecolor=MUTED, lw=0.9, zorder=2)
            t = ax.text(p, a, str(number[name]), fontsize=5.8, color=MUTED, ha="center", zorder=3)
            t._orig_xy, t._below = (p, a), a > lo + 0.70 * (hi - lo)
            labels.append(t)
            marks.append((ax, p, a, 3.2))
        for name, key, p, a in pts:                       # named entities drawn last, on top
            if key not in highlight:
                continue
            label, colour, marker, size = highlight[key]
            ax.scatter(p, a, s=size, color=colour, marker=marker, lw=0.8, edgecolor="white", zorder=4,
                       label=label if ds == corpora[0] else None)
            marks.append((ax, p, a, 4.5))
        ax.set_xscale("log")
        ax.set_title(TITLE[ds])
        # Accuracy has a hard ceiling; headroom above it would invite reading a point as >100%.
        ax.set_ylim(lo - (hi - lo) * 0.12 - 0.5, min(100.0, hi + (hi - lo) * 0.22 + 1.0))
        ax.margins(x=0.16)
        ax._labels, ax._marks = labels, marks
    for ax in axes[1]:
        ax.set_xlabel("parameters (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    declutter(fig, [t for ax in axes.ravel() for t in getattr(ax, "_labels", [])],
              [m for ax in axes.ravel() for m in getattr(ax, "_marks", [])])
    key_line = "   ".join(f"{number[n]} {n}" for n in baselines)
    fig.text(0.5, 0.045, key_line, ha="center", fontsize=6, color=MUTED)
    fig.text(0.5, 0.012, "Up is more accurate, left is smaller. Hollow numbered circles are BiTE's ten "
             "released baselines, re-run here with 3 seeds.", ha="center", fontsize=6.5, color=MUTED)
    save(fig, "fig_landscape")


def fig_subject_dumbbell():
    """Per subject, READER against BiTE. A mean hides which subjects a decoder actually helps."""
    acc = acc_by_subject(load_runs("cohort"))
    corpora = ["2a", "2b", "hgd", "sdssvep"]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.7),
                             gridspec_kw={"width_ratios": [9, 9, 14, 10]})
    for ax, ds in zip(axes, corpora):
        a, b = acc.get(("reader", ds), {}), acc.get(("bite", ds), {})
        subs = sorted(set(a) & set(b))
        y = np.arange(len(subs))[::-1]
        for yi, s in zip(y, subs):
            better = a[s] >= b[s]
            # The two markers are split by a fifth of a row: where a subject's two accuracies are equal
            # -- which happens -- a single row would hide one model completely behind the other.
            ax.plot([b[s], a[s]], [yi - 0.2, yi + 0.2], color=COLOR["reader"] if better else COLOR["bite"],
                    lw=1.3, alpha=0.5, solid_capstyle="round", zorder=2)
            ax.plot(b[s], yi - 0.2, MARKER["bite"], color=COLOR["bite"], ms=4.2, mec="white", mew=0.7, zorder=3)
            ax.plot(a[s], yi + 0.2, MARKER["reader"], color=COLOR["reader"], ms=4.2, mec="white", mew=0.7, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"S{s}" for s in subs], fontsize=6)
        ax.set_title(TITLE[ds])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("accuracy (%)")
    axes[0].plot([], [], MARKER["reader"], color=COLOR["reader"], ms=4.5, mec="white", mew=0.7, label="READER")
    axes[0].plot([], [], MARKER["bite"], color=COLOR["bite"], ms=4.5, mec="white", mew=0.7, label="BiTE")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig_subject_dumbbell")


def fig_subject_heatmap():
    """Model x subject accuracy. Rows sorted by mean; the dark columns are the hard subjects."""
    acc = acc_by_subject(load_runs())
    corpora = ["2a", "2b", "hgd", "sdssvep"]
    models = named_models()
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#184f95", "#0d366b"])
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 3.1), gridspec_kw={"width_ratios": [9, 9, 14, 10]})
    order = sorted(models, key=lambda m: -np.mean([np.mean(list(acc[(m[2], ds)].values()))
                                                   for ds in corpora if (m[2], ds) in acc] or [0]))
    for ax, ds in zip(axes, corpora):
        subs = sorted({s for name, _, key in order if (key, ds) in acc for s in acc[(key, ds)]})
        grid = np.full((len(order), len(subs)), np.nan)
        for i, (name, _, key) in enumerate(order):
            for j, s in enumerate(subs):
                if (key, ds) in acc and s in acc[(key, ds)]:
                    grid[i, j] = acc[(key, ds)][s]
        vmin = np.nanpercentile(grid, 2)
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=100)
        size = 4.6 if len(subs) <= 10 else 4.0
        for i in range(len(order)):
            for j in range(len(subs)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=size, color=MUTED)
                    continue
                shade = (grid[i, j] - vmin) / max(100 - vmin, 1e-9)
                # "100" does not fit a cell this narrow; it is the only three-digit value possible.
                label = "100" if grid[i, j] >= 99.5 else f"{grid[i, j]:.0f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=size - (1.0 if len(label) == 3 else 0),
                        color="white" if shade > 0.55 else INK)
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels([str(s) for s in subs], fontsize=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([n for n, _, _ in order] if ax is axes[0] else [], fontsize=5.6)
        ax.set_title(TITLE[ds])
        ax.set_xlabel("subject")
        ax.grid(visible=False)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=5, length=0)
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == "READER":
            lab.set_color(COLOR["reader"])
            lab.set_fontweight("bold")
    fig.supxlabel("Accuracy (%), seeds averaged per subject. Rows ordered by mean over corpora. Each corpus has "
                  "its OWN colour scale\n(they span different ranges), so compare within a panel, not across "
                  "panels. A dash is a model that cannot run on that corpus.",
                  fontsize=6.5, color=MUTED, y=-0.05)
    save(fig, "fig_subject_heatmap")


def fig_confusion():
    """Which classes READER actually confuses -- the accuracy tables cannot show this."""
    rep = json.loads((RES / "error_analysis.json").read_text())
    corpora = [ds for ds in ["2a", "2b", "hgd", "sdssvep"] if ds in rep["corpora"]]
    cmap = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#9ec5f4", "#3987e5", "#0d366b"])
    # SD-SSVEP has twelve classes against the others' two to four. Sharing a row would either squash it
    # or stretch them, so the small matrices take the top row and SD-SSVEP takes the bottom on its own.
    small = [ds for ds in corpora if rep["corpora"][ds]["n_classes"] <= 4]
    big = [ds for ds in corpora if rep["corpora"][ds]["n_classes"] > 4]
    fig = plt.figure(figsize=(7.16, 3.0 + 3.4 * bool(big)))
    gs = fig.add_gridspec(1 + bool(big), 6, height_ratios=[1, 1.45] if big else [1],
                          hspace=0.55, wspace=1.7)
    axes = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(len(small))]
    axes += [fig.add_subplot(gs[1, 1:5])] if big else []
    for ax, ds in zip(axes, small + big):
        e = rep["corpora"][ds]
        m = 100 * np.array(e["arms"]["reader"]["confusion_rownorm"])
        names = e["class_names"] or [str(i + 1) for i in range(e["n_classes"])]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=100)
        ax.set_box_aspect(1)
        wide = e["n_classes"] > 4
        for i in range(len(names)):
            for j in range(len(names)):
                if m[i, j] < 0.5:
                    continue
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center",
                        fontsize=5.4 if wide else 7, color="white" if m[i, j] > 55 else INK)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=6 if wide else 6.5,
                           rotation=0 if wide else 40, ha="center" if wide else "right")
        ax.set_yticklabels(names, fontsize=6 if wide else 6.5)
        ax.set_title(TITLE[ds], fontsize=8)
        ax.set_xlabel("predicted class", fontsize=6.5, labelpad=2)
        # Only the leftmost panel of each row: these tick labels are wide, and a y-axis label on the
        # middle panels lands inside the matrix to their left.
        if ax in (axes[0], axes[len(small)] if big else axes[0]):
            ax.set_ylabel("true class", fontsize=6.5, labelpad=2)
        ax.grid(visible=False)
        ax.tick_params(length=0)
    fig.suptitle("READER's confusion, row-normalised (%), pooled over subjects and seeds",
                 fontsize=7.5, color=MUTED, y=0.98)
    save(fig, "fig_confusion")


def fig_complementarity():
    """READER and BiTE are close on average and disagree on ~13% of 2a trials."""
    rep = json.loads((RES / "error_analysis.json").read_text())
    corpora = [ds for ds in ["2a", "2b", "hgd", "sdssvep"]
               if rep["corpora"].get(ds, {}).get("complementarity_reader_vs_bite")]
    parts = [("both_correct", "both right", "#9ec5f4"), ("only_a", "only READER", COLOR["reader"]),
             ("only_b", "only BiTE", COLOR["bite"]), ("both_wrong", "both wrong", "#d0cfca")]
    fig, ax = plt.subplots(figsize=(7.16, 1.9))
    y = np.arange(len(corpora))[::-1]
    for yi, ds in zip(y, corpora):
        p = rep["corpora"][ds]["complementarity_reader_vs_bite"]["pooled"]
        left = 0.0
        for key, label, colour in parts:
            w = p[key]
            ax.barh(yi, w, left=left, height=0.52, color=colour, zorder=3,
                    label=label if yi == y[0] else None)
            if w > 3.2:
                ax.text(left + w / 2, yi, f"{w:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if colour in (COLOR["reader"], COLOR["bite"]) else INK, zorder=4)
            left += w + 0.35                               # surface gap between stacked segments
        ax.text(101.5, yi, f"oracle {p['oracle']:.1f}", va="center", fontsize=6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([TITLE[d] for d in corpora], fontsize=7)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of test trials")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.45, 1.22), fontsize=6.5)
    save(fig, "fig_complementarity")


def fig_calibration():
    """Can the confidence be believed? It must be, for any system that stops when it is sure."""
    rep = json.loads((RES / "error_analysis.json").read_text())
    corpora = [ds for ds in ["2a", "2b", "hgd", "sdssvep"] if ds in rep["corpora"]]
    fig, axes = plt.subplots(1, len(corpora), figsize=(7.16, 2.1), sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), corpora):
        ax.plot([0, 100], [0, 100], "--", color=MUTED, lw=0.8, zorder=1,
                label="perfectly calibrated" if ds == corpora[0] else None)
        for arm in ("reader", "compact", "bite"):
            a = rep["corpora"][ds]["arms"].get(arm)
            if not a:
                continue
            bins = [b for b in a["reliability_bins"] if b["n"] >= 25]
            ax.plot([100 * b["confidence"] for b in bins], [100 * b["accuracy"] for b in bins],
                    "-", color=COLOR[arm], lw=1.4 if arm == "reader" else 1.0, marker=MARKER[arm],
                    ms=3.4, mec="white", mew=0.5, zorder=3, label=LABEL[arm] if ds == corpora[0] else None)
        ax.set_title(TITLE[ds])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("confidence (%)")
    np.atleast_1d(axes)[0].set_ylabel("accuracy (%)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.supxlabel("Above the diagonal = under-confident. Bins with fewer than 25 trials are dropped.",
                  fontsize=6.5, color=MUTED, y=-0.20)
    save(fig, "fig_calibration")


def main():
    fig_anytime()
    fig_exact_duration()
    fig_reader_vs_zoo()
    fig_landscape()
    fig_subject_dumbbell()
    fig_subject_heatmap()
    if (RES / "error_analysis.json").exists():
        fig_confusion()
        fig_complementarity()
        fig_calibration()
    (OUT / "README.md").write_text("""# Figures

Drawn by `scripts/make_figures.py` from committed files only. Every figure's numbers are in a table.

| figure | shows | numbers |
|---|---|---|
| `fig_anytime` | Accuracy at each decision deadline. Blue lines: ONE READER model read at every token boundary (solid: trained with prefix supervision; dashed: endpoint loss only). Markers: separately retrained specialists, one per deadline (BiTE aqua triangles, Compact orange squares), mean ± 1 SE across subjects; the full-trial marker is the within-subject cohort run. | `results/anytime_2a.md`, `anytime_2b.md`, `anytime_sdssvep.md` |
| `fig_exact_duration` | The SAME endpoint-trained checkpoints given only the first part of every test trial (exact truncated input, partial last pooling window kept). READER against its three matched controls; line is the subject mean, band a 95% bootstrap interval over subjects, dotted line chance, arrow the READER-minus-Compact gap at the deadline the paper quotes. The right-hand end of each curve IS that model's endpoint accuracy in the within-subject table. | `results/subject_stats/SUBJECT_STATS.md`, `results/exact_duration.json` |
| `fig_reader_vs_zoo` | READER minus every baseline re-run with 3 seeds under the same protocol: subject-level mean (dot) and 95% paired bootstrap CI (bar). Right of zero favours READER. "pending" = that baseline's runs are not complete. | `results/MAIN_TABLE.md`, `results/model_zoo/MODEL_ZOO.md` |
| `fig_landscape` | Accuracy against parameter count, one panel per corpus. Hollow circles are BiTE's ten released baselines re-run here; READER, BiTE and Compact are filled and named. Up is more accurate, left is smaller. | `results/MAIN_TABLE.md`, `results/EFFICIENCY.md` |
| `fig_subject_dumbbell` | One row per subject: BiTE's accuracy and READER's, joined by a line coloured by which of the two won. Shows who the mean is made of. | `results/subject_stats/SUBJECT_STATS.md` |
| `fig_subject_heatmap` | Every model against every subject. Rows ordered by mean accuracy over corpora; dark columns are the subjects every model finds hard. | `results/model_zoo/MODEL_ZOO.md`, `results/raw/runs.csv` |
| `fig_confusion` | READER's row-normalised confusion matrix per corpus, pooled over subjects and seeds. Class names are BiTE's own (2a/2b from the official `classlabel`, HGD from its event mapping); SD-SSVEP's twelve stimuli are numbered. | `results/ERROR_ANALYSIS.md` |
| `fig_complementarity` | Of every test trial: both models right, only READER, only BiTE, neither. The `oracle` figure at the right is the ceiling a perfect selector between the two would reach. | `results/ERROR_ANALYSIS.md` |
| `fig_calibration` | Reliability: accuracy against confidence, 15 equal-width bins. Above the diagonal means under-confident, which every arm is here (label smoothing 0.1). | `results/ERROR_ANALYSIS.md` |

Colour follows the model, never its rank: READER blue, Compact orange, BiTE aqua, FF-Control yellow,
Compact-Mean magenta, and marker shape repeats the identity so the figures survive greyscale printing
and colour-vision deficiency. Heatmaps use a single-hue blue ramp (magnitude, not identity).
""")


if __name__ == "__main__":
    main()
