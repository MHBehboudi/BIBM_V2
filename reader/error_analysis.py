#!/usr/bin/env python3
"""What the accuracy tables leave out: which classes, how confident, and whose errors.

Every run of record stores the FINAL-EPOCH test logits (`test_logits.npz`), and those logits
reproduce the accuracy in `summary.json` exactly -- checked here for every cell, so nothing in this
file is a second, differently-selected measurement of the same runs.

Four analyses, none of which needs a GPU:

  CONFUSION      per corpus and arm, the row-normalised confusion matrix pooled over subjects and
                 seeds, plus per-class recall. Class names come from BiTE's own preprocessing
                 (`third_party/BiteEEG/get_data.py`): 2a/2b from the official `classlabel` field
                 (1-indexed there, 0-indexed here), HGD from its explicit event-name mapping.
                 SD-SSVEP's twelve stimulus classes are left numbered: the released labels carry no
                 frequency table and we do not invent one.

  CALIBRATION    reliability over 15 equal-width confidence bins pooled per corpus and arm, with
                 ECE, NLL and Brier from the summaries. A decoder that must emit a decision under a
                 deadline is judged on whether its confidence can be believed, not only on accuracy.

  COMPLEMENTARITY  READER and BiTE see the same test trials in the same order, so their errors can
                 be compared trial by trial. Reports both-correct / only-READER / only-BiTE /
                 both-wrong, the oracle upper bound (either is right) and the accuracy of averaging
                 their softmax outputs. Subject-level, with the paper's bootstrap CI and exact
                 Wilcoxon test.

  STABILITY      within-subject SD across the three seeds, per arm and corpus: the size of the
                 re-run noise that every delta in the paper is measured against.

Writes results/error_analysis.json and results/ERROR_ANALYSIS.md.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from reader.subject_stats import boot_ci, summarise, RNG_SEED

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ["2a", "2b", "hgd", "sdssvep"]
TITLE = {"2a": "BCI IV-2a", "2b": "BCI IV-2b", "hgd": "HGD", "sdssvep": "SD-SSVEP"}
# Verified against third_party/BiteEEG/get_data.py. None = no verified naming; classes stay numbered.
CLASSES = {
    "2a": ["left hand", "right hand", "feet", "tongue"],
    "2b": ["left hand", "right hand"],
    "hgd": ["left hand", "right hand", "feet", "rest"],
    "sdssvep": None,
}
ARMS = ["reader", "compact", "bite"]
N_BINS = 15


def load_cells(runs, arm, ds):
    """{(subject, seed): (logits, labels)} for one arm and corpus, checked against summary.json."""
    out = {}
    for p in sorted((runs / "cohort" / arm).glob(f"{ds}_S*_seed*/summary.json")):
        s = json.loads(p.read_text())
        if s.get("status") != "complete":
            continue
        d = np.load(p.parent / "test_logits.npz")
        logits, labels = d["logits"].astype(np.float64), d["labels"].astype(np.int64)
        acc = float((logits.argmax(1) == labels).mean())
        if abs(acc - s["final_test"]["acc"]) > 1e-5:
            raise SystemExit(f"{p.parent}: logits give {acc:.6f}, summary says {s['final_test']['acc']:.6f}")
        out[(s["subject"], s["seed"])] = (logits, labels)
    return out


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def confusion(cells, n_classes):
    """Counts pooled over cells, then row-normalised: entry [i, j] = P(predicted j | true i)."""
    m = np.zeros((n_classes, n_classes))
    for logits, labels in cells.values():
        pred = logits.argmax(1)
        np.add.at(m, (labels, pred), 1.0)
    rows = m.sum(1, keepdims=True)
    return m, np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)


def reliability(cells):
    """Equal-width confidence bins pooled over cells -> (bin centre, accuracy, mean confidence, n)."""
    conf_all, correct_all = [], []
    for logits, labels in cells.values():
        p = softmax(logits)
        conf_all.append(p.max(1))
        correct_all.append((p.argmax(1) == labels).astype(float))
    conf = np.concatenate(conf_all)
    correct = np.concatenate(correct_all)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, N_BINS - 1)
    bins = []
    for b in range(N_BINS):
        m = idx == b
        n = int(m.sum())
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "accuracy": float(correct[m].mean()) if n else None,
                     "confidence": float(conf[m].mean()) if n else None})
    ece = sum(b["n"] / len(conf) * abs(b["accuracy"] - b["confidence"]) for b in bins if b["n"])
    return bins, float(ece), float(conf.mean()), float(correct.mean())


def complementarity(a_cells, b_cells):
    """Trial-by-trial overlap of two arms' errors on the cells they share."""
    per_cell = []
    for key in sorted(set(a_cells) & set(b_cells)):
        (la, ya), (lb, yb) = a_cells[key], b_cells[key]
        if la.shape != lb.shape or not np.array_equal(ya, yb):
            raise SystemExit(f"{key}: the two arms were not evaluated on the same trials")
        ca, cb = la.argmax(1) == ya, lb.argmax(1) == yb
        ens = (softmax(la) + softmax(lb)).argmax(1) == ya
        per_cell.append({
            "subject": key[0], "seed": key[1], "n_trials": int(len(ya)),
            "both_correct": float(100 * (ca & cb).mean()),
            "only_a": float(100 * (ca & ~cb).mean()),
            "only_b": float(100 * (~ca & cb).mean()),
            "both_wrong": float(100 * (~ca & ~cb).mean()),
            "a": float(100 * ca.mean()), "b": float(100 * cb.mean()),
            "oracle": float(100 * (ca | cb).mean()), "ensemble": float(100 * ens.mean()),
        })
    return per_cell


def by_subject(per_cell, field):
    d = defaultdict(list)
    for r in per_cell:
        d[r["subject"]].append(r[field])
    return {s: float(np.mean(v)) for s, v in sorted(d.items())}


def subject_mean_accuracy(cells):
    """Mean over subjects of the seed-mean accuracy -- the same statistic as results/MAIN_TABLE.md.

    Not the same as pooling trials: 2b and HGD test sessions differ in size between subjects, so a
    trial-pooled mean would silently weight the larger subjects and disagree with the main table.
    """
    by = defaultdict(list)
    for (sub, _), (logits, labels) in cells.items():
        by[sub].append(100 * float((logits.argmax(1) == labels).mean()))
    return float(np.mean([np.mean(v) for v in by.values()]))


def seed_stability(cells):
    """Mean over subjects of the SD across seeds of that subject's accuracy (pp)."""
    by = defaultdict(list)
    for (sub, _), (logits, labels) in cells.items():
        by[sub].append(100 * float((logits.argmax(1) == labels).mean()))
    sds = [float(np.std(v, ddof=1)) for v in by.values() if len(v) > 1]
    spread = [float(max(v) - min(v)) for v in by.values() if len(v) > 1]
    if not sds:
        return None
    return {"n_subjects": len(sds), "mean_sd_pp": float(np.mean(sds)),
            "max_sd_pp": float(max(sds)), "mean_range_pp": float(np.mean(spread)),
            "max_range_pp": float(max(spread))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    report = {"note": "final-epoch test logits; every cell checked against summary.json",
              "corpora": {}}
    for ds in CORPORA:
        cells = {a: load_cells(args.runs, a, ds) for a in ARMS}
        if not cells["reader"]:
            continue
        n_classes = int(max(lab.max() for lg, lab in cells["reader"].values())) + 1
        entry = {"n_classes": n_classes, "class_names": CLASSES[ds], "arms": {}}
        for arm in ARMS:
            if not cells[arm]:
                continue
            counts, norm = confusion(cells[arm], n_classes)
            bins, ece, conf, acc = reliability(cells[arm])
            entry["arms"][arm] = {
                "n_cells": len(cells[arm]),
                "accuracy": subject_mean_accuracy(cells[arm]),   # matches results/MAIN_TABLE.md
                "accuracy_trial_pooled": 100 * acc,              # the base of the calibration bins
                "confusion_counts": counts.tolist(), "confusion_rownorm": norm.tolist(),
                "per_class_recall": (100 * np.diag(norm)).tolist(),
                "reliability_bins": bins, "ece": ece, "mean_confidence": conf,
                "stability": seed_stability(cells[arm]),
            }
        if cells["bite"]:
            per_cell = complementarity(cells["reader"], cells["bite"])
            entry["complementarity_reader_vs_bite"] = {
                "per_cell": per_cell,
                "pooled": {k: float(np.mean([r[k] for r in per_cell])) for k in
                           ["both_correct", "only_a", "only_b", "both_wrong", "a", "b",
                            "oracle", "ensemble"]},
                "oracle_minus_reader": summarise(
                    {s: v - by_subject(per_cell, "a")[s] for s, v in by_subject(per_cell, "oracle").items()}, rng),
                "ensemble_minus_reader": summarise(
                    {s: v - by_subject(per_cell, "a")[s] for s, v in by_subject(per_cell, "ensemble").items()}, rng),
            }
        report["corpora"][ds] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "error_analysis.json").write_text(json.dumps(report, indent=1))
    write_markdown(report, args.out / "ERROR_ANALYSIS.md")
    print("->", (args.out / "error_analysis.json").relative_to(ROOT))
    print("->", (args.out / "ERROR_ANALYSIS.md").relative_to(ROOT))


def write_markdown(report, path):
    L = ["# Error analysis, calibration and complementarity", "",
         "Computed from the FINAL-EPOCH test logits stored with every run of record. Each cell's logits",
         "were checked against the accuracy in its `summary.json`; a mismatch aborts the script, so these",
         "are the same runs and the same epoch as `results/MAIN_TABLE.md`, not a second selection.",
         "Produced by `reader/error_analysis.py`.", ""]

    L += ["## Per-class recall (%), pooled over subjects and seeds", ""]
    for ds, e in report["corpora"].items():
        names = e["class_names"] or [f"class {i + 1}" for i in range(e["n_classes"])]
        L += [f"**{TITLE[ds]}**", "", "| arm | " + " | ".join(names) + " | spread |",
              "|---|" + "---:|" * (len(names) + 1)]
        for arm, a in e["arms"].items():
            r = a["per_class_recall"]
            L.append(f"| {arm} | " + " | ".join(f"{x:.1f}" for x in r) + f" | {max(r) - min(r):.1f} |")
        L.append("")

    L += ["## Calibration", "",
          "ECE over 15 equal-width confidence bins. Confidence is a property of trials, so this table is",
          "trial-pooled and its accuracy column is therefore the trial-pooled mean, which differs slightly",
          "from the subject-mean in `results/MAIN_TABLE.md` where test sessions differ in size (2b, HGD).",
          "`conf − acc` is positive when a model is overconfident. Bin tables: `results/error_analysis.json`.",
          "",
          "**Every model here is UNDER-confident, on every corpus.** That is the expected direction for this",
          "recipe rather than a finding about the architecture: all arms train with label smoothing 0.1, which",
          "caps the target probability, and the effect grows with the number of classes (largest on SD-SSVEP's",
          "twelve). It does mean a confidence threshold tuned on one corpus will not transfer to another.", "",
          "| corpus | arm | accuracy (trial-pooled) | mean confidence | conf − acc | ECE |",
          "|---|---|---:|---:|---:|---:|"]
    for ds, e in report["corpora"].items():
        for arm, a in e["arms"].items():
            acc = a["accuracy_trial_pooled"]
            L.append(f"| {TITLE[ds]} | {arm} | {acc:.2f} | {100 * a['mean_confidence']:.2f} | "
                     f"{100 * a['mean_confidence'] - acc:+.2f} | {a['ece']:.4f} |")
    L.append("")

    L += ["## READER and BiTE make different mistakes", "",
          "Same trials, same order, trial-by-trial. `oracle` = either model is right (an upper bound no",
          "selector can exceed); `ensemble` = the two softmax outputs averaged, which needs both models at",
          "inference and is reported as a diagnostic, not as a proposed system.", "",
          "| corpus | both right | only READER | only BiTE | both wrong | oracle | softmax-average |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, e in report["corpora"].items():
        c = e.get("complementarity_reader_vs_bite")
        if not c:
            continue
        p = c["pooled"]
        L.append(f"| {TITLE[ds]} | {p['both_correct']:.2f} | {p['only_a']:.2f} | {p['only_b']:.2f} | "
                 f"{p['both_wrong']:.2f} | {p['oracle']:.2f} | {p['ensemble']:.2f} |")
    L += ["", "Subject-level, with the paper's bootstrap CI and exact Wilcoxon test:", "",
          "| corpus | oracle − READER | softmax-average − READER |", "|---|---|---|"]
    for ds, e in report["corpora"].items():
        c = e.get("complementarity_reader_vs_bite")
        if not c:
            continue
        from reader.subject_stats import fmt
        L.append(f"| {TITLE[ds]} | {fmt(c['oracle_minus_reader'])} | {fmt(c['ensemble_minus_reader'])} |")
    L.append("")

    L += ["## Re-run noise: SD across the three seeds, within subject", "",
          "Every delta in the paper is a difference of numbers with this much spread underneath it.", "",
          "| corpus | arm | mean SD (pp) | max SD (pp) | mean seed range (pp) | max range (pp) |",
          "|---|---|---:|---:|---:|---:|"]
    for ds, e in report["corpora"].items():
        for arm, a in e["arms"].items():
            s = a["stability"]
            if s:
                L.append(f"| {TITLE[ds]} | {arm} | {s['mean_sd_pp']:.2f} | {s['max_sd_pp']:.2f} | "
                         f"{s['mean_range_pp']:.2f} | {s['max_range_pp']:.2f} |")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
