#!/usr/bin/env python3
"""Aggregate a study: final accuracy, paired deltas, and a per-run failure-mode reading.

Failure-mode rules (every threshold printed with its measured value, never a bare label):
  capacity            final eval-mode TRAIN accuracy < 0.95
  no usable signal    within-test CV probe at the reader AND on model-free log power both below
                      chance + 25% of the headroom
  session misalign    information present in test features (within-test CV) exceeds what the
                      train-fitted probe achieves on test by > 10 points
  late overfitting    diagnostic test peak minus final > 3 points, or test NLL rose > 0.2 from its minimum
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELLS = [("2a", 2, "hard"), ("2a", 9, "easy"), ("2b", 9, "hard"), ("2b", 4, "easy"),
         ("hgd", 11, "hard"), ("hgd", 14, "easy"), ("sdssvep", 1, "hard"), ("sdssvep", 5, "easy")]
CHANCE = {"2a": .25, "2b": .5, "hgd": .25, "sdssvep": 1 / 12}


def load_run(path: Path):
    s = json.loads((path / "summary.json").read_text())
    curves = json.loads((path / "curves.json").read_text())
    diag = [json.loads(l) for l in gzip.open(path / "diag.jsonl.gz", "rt")]
    return s, curves, diag


def failure_modes(dataset, s, curves, diag):
    c = CHANCE[dataset]
    last = diag[-1]
    fit = s["final_train"]["acc"]
    test = s["final_test"]["acc"]
    peak = s["test_peak_diagnostic"]["acc"]
    nlls = [row["test"]["nll"] for row in curves]
    reader = last["key_points"].get("reader", {})
    info = (s["input_information"].get("complex_spectrum") if dataset == "sdssvep"
            else s["input_information"].get("logpower")) or {}
    labels = []
    evidence = {"train_fit": fit, "test": test, "gap": fit - test, "peak_minus_final": peak - test,
                "test_nll_rise": nlls[-1] - min(nlls), "reader_probe_train_to_test": reader.get("probe_train_to_test"),
                "reader_probe_within_test_cv": reader.get("probe_within_test_cv"),
                "input_logpower_within_test_cv": info.get("probe_within_test_cv"),
                "input_logpower_train_to_test": info.get("probe_train_to_test")}
    if fit < .95:
        labels.append("capacity")
    floor = c + .25 * (1 - c)
    if (reader.get("probe_within_test_cv", 1) < floor) and (info.get("probe_within_test_cv", 1) < floor):
        labels.append("no_signal")
    if reader.get("probe_within_test_cv") is not None and \
            reader["probe_within_test_cv"] - reader["probe_train_to_test"] > .10:
        labels.append("session_misalignment")
    if peak - test > .03 or nlls[-1] - min(nlls) > .2:
        labels.append("late_overfitting")
    return labels or ["none_detected"], evidence


def layer_table(diag, points=("carriers", "spatial", "tokens", "time_stream", "freq_stream", "reader")):
    last = diag[-1]
    rows = {}
    for kp in points:
        r = last["key_points"].get(kp)
        if r and "erank_train" in r:
            rows[kp] = {k: r.get(k) for k in ("erank_train", "erank_test", "nc1_train", "nc1_test",
                                               "class_direction_cos", "probe_train_fit",
                                               "probe_train_to_test", "probe_within_test_cv")}
    return rows


def module_health(diag):
    last = diag[-1]
    worst_shift = sorted(((v.get("unit_mean_shift", 0), k) for k, v in last["modules"].items()), reverse=True)[:4]
    dead = {k: v["dead_unit_frac"] for k, v in last["modules"].items() if v.get("dead_unit_frac", 0) > 0}
    bn = {k: (round(v.get("test_mean_shift", 0), 3), round(v.get("test_log_var_ratio", 0), 3))
          for k, v in last["batchnorm"].items()}
    params = last["parameters"]
    tiny = {k: v["grad_to_weight"] for k, v in params.items() if v["grad_to_weight"] < 1e-4}
    caps = {k: v["frac_rows_at_cap"] for k, v in params.items() if "frac_rows_at_cap" in v}
    return {"largest_unit_mean_shift": worst_shift, "dead_units": dead, "bn_test_shift_logvar": bn,
            "near_zero_grad_to_weight": tiny, "max_norm_rows_at_cap": caps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", type=Path)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs = defaultdict(dict)
    for arm_dir in sorted(p for p in args.study.iterdir() if p.is_dir()):
        for run in sorted(arm_dir.iterdir()):
            if (run / "summary.json").exists():
                runs[arm_dir.name][run.name] = run
    arms = sorted(runs)
    ref = args.reference or arms[0]
    lines = [f"# Study {args.study.name}", "", f"reference arm: `{ref}`", ""]
    # ---------------------------------------------------------------- accuracy table
    header = "| cell | " + " | ".join(arms) + " |"
    lines += ["## Final-epoch test accuracy (mean over seeds, n)", "", header, "|" + "---|" * (len(arms) + 1)]
    table = defaultdict(dict)
    for ds, sub, kind in CELLS:
        cells = []
        for arm in arms:
            vals = []
            for name, path in runs[arm].items():
                if name.startswith(f"{ds}_S{sub}_"):
                    vals.append(json.loads((path / "summary.json").read_text())["final_test"]["acc"])
                    table[arm][name] = vals[-1]
            cells.append(f"{100 * np.mean(vals):.2f} ({len(vals)})" if vals else "-")
        lines.append(f"| {ds} S{sub} {kind} | " + " | ".join(cells) + " |")
    # ---------------------------------------------------------------- paired deltas
    lines += ["", f"## Paired delta vs `{ref}` (same cell, same seed), percentage points", ""]
    lines += ["| arm | pairs | mean | se | wins/ties/losses | per-dataset means |", "|---|---|---|---|---|---|"]
    for arm in arms:
        if arm == ref:
            continue
        deltas, by_ds = [], defaultdict(list)
        for name, acc in table[arm].items():
            if name in table[ref]:
                d = 100 * (acc - table[ref][name])
                deltas.append(d)
                by_ds[name.split("_")[0]].append(d)
        if not deltas:
            continue
        d = np.asarray(deltas)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        per = ", ".join(f"{k} {np.mean(v):+.2f}" for k, v in sorted(by_ds.items()))
        lines.append(f"| {arm} | {len(d)} | {d.mean():+.2f} | {se:.2f} | {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()} | {per} |")
    # ---------------------------------------------------------------- failure modes
    lines += ["", "## Failure-mode reading per run (final epoch)", ""]
    detail = {}
    for arm in arms:
        lines += [f"### {arm}", "", "| run | modes | fit | test | gap | peak-final | nll rise | reader P(tr->te) | reader P(cv te) | input logpow P(cv te) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, path in sorted(runs[arm].items()):
            s, curves, diag = load_run(path)
            modes, ev = failure_modes(s["dataset"], s, curves, diag)
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            lines.append(f"| {name} | {', '.join(modes)} | {fmt(ev['train_fit'])} | {fmt(ev['test'])} | {fmt(ev['gap'])} | "
                         f"{fmt(ev['peak_minus_final'])} | {fmt(ev['test_nll_rise'])} | {fmt(ev['reader_probe_train_to_test'])} | "
                         f"{fmt(ev['reader_probe_within_test_cv'])} | {fmt(ev['input_logpower_within_test_cv'])} |")
            detail[f"{arm}/{name}"] = {"modes": modes, "evidence": ev, "layers": layer_table(diag),
                                       "health": module_health(diag),
                                       "mechanism": diag[-1].get("mechanism"),
                                       "telemetry_final": curves[-1].get("model")}
        lines.append("")
    # ---------------------------------------------------------------- layer trajectories, hard vs easy
    lines += ["## Layer trajectories (mean over seeds): train->test probe / within-test CV probe / erank(test)", ""]
    marks = (10, 50, 100, 200, 400, 600)
    for arm in arms:
        lines += [f"### {arm}", ""]
        for ds, sub, kind in CELLS:
            paths = [p for n, p in runs[arm].items() if n.startswith(f"{ds}_S{sub}_")]
            if not paths:
                continue
            per_epoch = defaultdict(lambda: defaultdict(list))
            accs = defaultdict(lambda: defaultdict(list))
            for path in paths:
                _, curves, diag = load_run(path)
                for row in diag:
                    if row["epoch"] in marks:
                        for kp, r in row["key_points"].items():
                            if "probe_train_to_test" in r:
                                per_epoch[kp][row["epoch"]].append((r["probe_train_to_test"], r.get("probe_within_test_cv", np.nan), r["erank_test"]))
                        accs["acc"][row["epoch"]].append((row["logits"]["train"]["acc"], row["logits"]["test"]["acc"], row["logits"]["test"]["nll"]))
            lines.append(f"**{ds} S{sub} ({kind})**")
            lines.append("| point | " + " | ".join(f"ep{e}" for e in marks) + " |")
            lines.append("|---|" + "---|" * len(marks))
            row = []
            for e in marks:
                v = accs["acc"].get(e)
                row.append("-" if not v else f"tr {np.mean([a[0] for a in v]):.2f} te {np.mean([a[1] for a in v]):.2f} nll {np.mean([a[2] for a in v]):.2f}")
            lines.append("| logits | " + " | ".join(row) + " |")
            for kp, by in per_epoch.items():
                row = []
                for e in marks:
                    v = by.get(e)
                    row.append("-" if not v else f"{np.mean([a[0] for a in v]):.2f}/{np.nanmean([a[1] for a in v]):.2f}/{np.mean([a[2] for a in v]):.0f}")
                lines.append(f"| {kp} | " + " | ".join(row) + " |")
            lines.append("")
    out = args.out or args.study / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps(detail, indent=1, default=float))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
