#!/usr/bin/env python3
"""Paper figures (PDF for the manuscript, PNG for review), drawn only from committed results files.

  fig_anytime            one READER read at every deadline vs banks of separately retrained specialists
  fig_exact_duration     the SAME trained checkpoints given only the first part of each test trial
  fig_reader_vs_zoo      READER minus every baseline re-run here, subject-level mean and 95% CI
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


def fig_exact_duration():
    recs = json.loads((RES / "exact_duration.json").read_text())["records"]
    acc = defaultdict(lambda: defaultdict(dict))           # (arm, ds) -> n -> {(sub, seed): acc}
    for r in recs:
        for n, a in r["acc"].items():
            acc[(r["arm"], r["dataset"])][int(n)][(int(r["subject"]), int(r["seed"]))] = 100 * float(a)
    corpora = ["2a", "2b", "sdssvep"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for ax, ds in zip(axes, corpora):
        for arm in ("reader", "bite", "compact_mean", "ff_control", "compact"):
            c = acc.get((arm, ds))
            if not c:
                continue
            ns = sorted(c)
            ys = [subject_curve(c[n])[0] for n in ns]
            ax.plot(np.array(ns) / FS[ds], ys, "-", color=COLOR[arm], lw=1.8 if arm == "reader" else 1.0,
                    marker=MARKER[arm], ms=4, mec="white", mew=0.6, label=LABEL[arm], zorder=4 if arm == "reader" else 3)
        ax.set_title(TITLE[ds])
        ax.set_xlabel("observed part of the trial (s)")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("test accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
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


def main():
    fig_anytime()
    fig_exact_duration()
    fig_reader_vs_zoo()
    (OUT / "README.md").write_text("""# Figures

Drawn by `scripts/make_figures.py` from committed files only. Every figure's numbers are in a table.

| figure | shows | numbers |
|---|---|---|
| `fig_anytime` | Accuracy at each decision deadline. Blue lines: ONE READER model read at every token boundary (solid: trained with prefix supervision; dashed: endpoint loss only). Markers: separately retrained specialists, one per deadline (BiTE aqua triangles, Compact orange squares), mean ± 1 SE across subjects; the full-trial marker is the within-subject cohort run. | `results/anytime_2a.md`, `anytime_2b.md`, `anytime_sdssvep.md` |
| `fig_exact_duration` | The SAME endpoint-trained checkpoints given only the first part of every test trial (exact truncated input, partial last pooling window kept). | `results/subject_stats/SUBJECT_STATS.md`, `results/exact_duration.json` |
| `fig_reader_vs_zoo` | READER minus every baseline re-run with 3 seeds under the same protocol: subject-level mean (dot) and 95% paired bootstrap CI (bar). Right of zero favours READER. "pending" = that baseline's runs are not complete. | `results/MAIN_TABLE.md`, `results/model_zoo/MODEL_ZOO.md` |
""")


if __name__ == "__main__":
    main()
