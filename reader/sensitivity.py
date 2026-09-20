#!/usr/bin/env python3
"""Hyper-parameter sensitivity of the reverse reader's advantage (co-author request, 2026-09-20).

Two questions, one factor moved at a time around the declared operating point:

  Q1  prefix-supervision weight   lambda in {0, 0.1, 0.3, 1.0}, pooling at the corpus default
  Q2  token resolution            p giving ~16 / ~32 / ~64 tokens, lambda fixed

The reported quantity is the PAIRED difference between READER and its forward-only control trained
at the SAME setting, because a level that moves with lambda or p says nothing about the mechanism:

  dnAUC = nAUC(READER) - nAUC(no-reverse)      per subject, seeds averaged first

nAUC is the normalised accuracy-duration area of Eq. (nauc) in the paper: trapezoidal integration of
the seed-averaged exact-duration accuracy curve, divided by the width of the integration range, so
it is in accuracy points and is comparable across corpora with different trial lengths. Integration
runs over the SAME wall-clock grid for every cell -- the exact-duration grid in SAMPLES, not the
token grid -- which is what makes pooling factors comparable at all: p changes the token grid, and a
token-indexed summary would compare different durations across the arms of Q2.

Two ranges are reported. `nauc` is the paper's range (2a/2b 1.0-4.0 s, SD-SSVEP 0.25-1.0 s), kept so
the numbers are comparable with the main text; `nauc_full` integrates the whole evaluated grid
(from 0.5 s / 0.125 s). The full range is the larger of the two for every cell measured so far.

Endpoint accuracy is the full-trial accuracy of the same runs, from their own summary.json, so the
lambda = 0 / default-pool row reproduces the within-subject table exactly.

Uncertainty: 95% percentile bootstrap over SUBJECTS as paired units (10,000 resamples, fixed seed),
the same procedure as every other paired comparison in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reader.data import SPECS  # noqa: E402

CORPORA = ("2a", "2b", "sdssvep")
LABEL = {"2a": "BCICIV-2A", "2b": "BCICIV-2B", "sdssvep": "SD-SSVEP"}
# The paper's nAUC integration range, as the first grid point to keep.
PAPER_START = {"2a": 250, "2b": 250, "sdssvep": 64}
BOOTSTRAP = 10000


def nauc(samples, acc, fs, start=None):
    """Trapezoidal area under accuracy-vs-duration, divided by the duration range (accuracy points)."""
    t = np.asarray(samples, float) / fs
    a = np.asarray(acc, float)
    if start is not None:
        keep = t >= start / fs - 1e-9
        t, a = t[keep], a[keep]
    return float(np.trapezoid(a, t) / (t[-1] - t[0]))


def bootstrap_ci(values, seed=0):
    d = np.asarray(values, float)
    if len(d) < 2:
        return float(d.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


class Cell:
    """One (corpus, arm, lambda, pool) block of runs: 9-10 subjects x 3 seeds."""

    def __init__(self, dataset, arm, lam, pool, source):
        self.dataset, self.arm, self.lam, self.pool, self.source = dataset, arm, lam, pool, source
        self.endpoint = defaultdict(list)     # subject -> [acc per seed]
        self.curves = defaultdict(list)       # subject -> [{samples: acc} per seed]
        self.devices, self.params = set(), set()

    # ---------------------------------------------------------------- loading
    def add_run(self, summary, curve):
        s = summary["subject"]
        self.endpoint[s].append(100 * summary["final_test"]["acc"])
        self.curves[s].append(curve)
        self.devices.add(summary.get("device", "?"))
        self.params.add(summary.get("parameters"))

    @property
    def subjects(self):
        return sorted(self.endpoint)

    @property
    def n_runs(self):
        return sum(len(v) for v in self.endpoint.values())

    def endpoint_by_subject(self):
        return {s: float(np.mean(v)) for s, v in self.endpoint.items()}

    def nauc_by_subject(self, full_range):
        fs = SPECS[self.dataset]["fs"]
        start = None if full_range else PAPER_START[self.dataset]
        out = {}
        for s, curves in self.curves.items():
            grid = sorted(int(k) for k in curves[0])
            # exact_duration stores accuracy as a FRACTION; nAUC is reported in accuracy points
            mean = [100 * float(np.mean([c[str(n)] for c in curves])) for n in grid]
            out[s] = nauc(grid, mean, fs, start)
        return out

    def token_count(self):
        return -(-SPECS[self.dataset]["samples"] // self.pool)


def load(runs_root, exact_root, plan):
    """plan: list of (dataset, arm, lambda, pool, run_dir, exact_dir). Missing cells are skipped."""
    cells = {}
    for dataset, arm, lam, pool, run_dir, exact_dir in plan:
        run_dir, exact_dir = Path(run_dir), Path(exact_dir)
        if not run_dir.exists() or not exact_dir.exists():
            continue
        cell = Cell(dataset, arm, lam, pool, str(run_dir))
        for record in sorted(exact_dir.glob(f"{dataset}_S*_seed*.json")):
            rec = json.loads(record.read_text())
            run = run_dir / record.stem
            summary_path = run / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if abs(rec["final_test_acc_logged"] - summary["final_test"]["acc"]) > 1e-9:
                raise SystemExit(f"{record} was computed from a different run than {summary_path}")
            if rec.get("pool") not in (None, pool):
                raise SystemExit(f"{record} has pool {rec.get('pool')}, expected {pool}")
            if summary.get("prefix_weight") != lam:
                raise SystemExit(f"{summary_path} has lambda {summary.get('prefix_weight')}, expected {lam}")
            cell.add_run(summary, rec["acc"])
        if cell.n_runs:
            cells[(dataset, arm, lam, pool)] = cell
    return cells


def compare(reader, control, full_range=False, seed=0):
    """Paired subject-level difference between two cells trained at the same setting."""
    subs = sorted(set(reader.subjects) & set(control.subjects))
    if not subs:
        return None
    a_end, b_end = reader.endpoint_by_subject(), control.endpoint_by_subject()
    a_auc, b_auc = reader.nauc_by_subject(full_range), control.nauc_by_subject(full_range)
    d_auc = [a_auc[s] - b_auc[s] for s in subs]
    d_end = [a_end[s] - b_end[s] for s in subs]
    mean, lo, hi = bootstrap_ci(d_auc, seed)
    e_mean, e_lo, e_hi = bootstrap_ci(d_end, seed + 1)
    return {
        "subjects": len(subs), "runs": [reader.n_runs, control.n_runs],
        "reader_nauc": float(np.mean([a_auc[s] for s in subs])),
        "control_nauc": float(np.mean([b_auc[s] for s in subs])),
        "delta_nauc": mean, "delta_nauc_ci": [lo, hi],
        "wins": int(sum(d > 0 for d in d_auc)), "losses": int(sum(d < 0 for d in d_auc)),
        "reader_endpoint": float(np.mean([a_end[s] for s in subs])),
        "control_endpoint": float(np.mean([b_end[s] for s in subs])),
        "delta_endpoint": e_mean, "delta_endpoint_ci": [e_lo, e_hi],
    }


def fmt(row, key):
    lo, hi = row[f"{key}_ci"]
    return f"{row[key]:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def table(cells, axis, dataset, values, fixed, full_range):
    """One corpus's rows for one sweep. `axis` is 'lam' or 'pool'; `fixed` names the other factor."""
    lines, rows = [], {}
    for v in values:
        key_r = (dataset, "reader", v, fixed) if axis == "lam" else (dataset, "reader", fixed, v)
        key_c = (dataset, "compact", v, fixed) if axis == "lam" else (dataset, "compact", fixed, v)
        if key_r not in cells or key_c not in cells:
            lines.append(f"| {v:g} | _pending_ | | | | | |")
            continue
        reader, control = cells[key_r], cells[key_c]
        row = compare(reader, control, full_range)
        row["tokens"] = reader.token_count()
        row["token_ms"] = 1000 * reader.pool / SPECS[dataset]["fs"]
        rows[f"{v:g}"] = row
        first = f"{v:g}" if axis == "lam" else f"{v:g} ({row['tokens']} tokens, {row['token_ms']:.1f} ms)"
        lines.append(f"| {first} | {row['reader_nauc']:.2f} | {row['control_nauc']:.2f} | "
                     f"{fmt(row, 'delta_nauc')} | {row['wins']}/{row['losses']} | "
                     f"{row['reader_endpoint']:.2f} | {row['control_endpoint']:.2f} | "
                     f"{fmt(row, 'delta_endpoint')} |")
    return lines, rows


HEADER = ("| {} | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | "
          "no-reverse endpoint | dendpoint [95% CI] |")
RULE = "|---|---:|---:|---:|---:|---:|---:|---:|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True,
                    help="JSON list of cells: {dataset, arm, lambda, pool, runs, exact}")
    ap.add_argument("--out", type=Path, default=ROOT / "results/SENSITIVITY.md")
    ap.add_argument("--full-range", action="store_true",
                    help="integrate nAUC over the whole evaluated grid instead of the paper's range")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    cells = load(None, None, [(c["dataset"], c["arm"], c["lambda"], c["pool"], c["runs"], c["exact"])
                              for c in plan])
    lambdas = sorted({c["lambda"] for c in plan})
    default_pool = {ds: SPECS[ds]["pool"] for ds in CORPORA}
    pools = {ds: sorted({c["pool"] for c in plan if c["dataset"] == ds}, reverse=True) for ds in CORPORA}

    rng = "the whole evaluated grid" if args.full_range else "the paper's range"
    lines = [
        "# Sensitivity of the reverse-reader advantage to its two main hyper-parameters", "",
        "Every row is a PAIRED comparison: REACT and its forward-only control (no reverse reader, "
        "everything else identical, shared initialisation at the same seed) trained at the SAME "
        "setting, 3 seeds per subject, 600 epochs, fixed final epoch, official session roles.", "",
        f"nAUC is Eq. (nauc) integrated over {rng}: 2a/2b "
        f"{'0.5' if args.full_range else '1.0'}-4.0 s, SD-SSVEP "
        f"{'0.125' if args.full_range else '0.25'}-1.0 s, on the exact-duration grid in samples "
        "(the same wall-clock durations for every cell, so pooling factors are comparable). "
        "Endpoint is full-trial accuracy. W/L counts subjects by the sign of the nAUC difference.", "",
    ]
    detail = {"integration": "full" if args.full_range else "paper", "lambda_sweep": {}, "pool_sweep": {}}

    lines += ["## Q1  Prefix-supervision weight (pooling at the corpus default)", "",
              "The reverse reader is not what the prefix loss pays for: it is compared with its own "
              "control at each weight.", ""]
    for ds in CORPORA:
        sub, rows = table(cells, "lam", ds, lambdas, default_pool[ds], args.full_range)
        if not rows:
            continue
        lines += [f"### {LABEL[ds]}  (p = {default_pool[ds]}, "
                  f"{-(-SPECS[ds]['samples'] // default_pool[ds])} tokens)", "",
                  HEADER.format("lambda"), RULE, *sub, ""]
        detail["lambda_sweep"][ds] = rows

    lines += ["## Q2  Token resolution (prefix-supervision weight fixed)", "",
              "p sets the token duration and the token count; all three settings stay inside the "
              "71-token receptive field of the TCN, so no condition is limited by history it cannot "
              "reach.", ""]
    for ds in CORPORA:
        for lam in lambdas:
            sub, rows = table(cells, "pool", ds, pools[ds], lam, args.full_range)
            if len(rows) < 2:
                continue
            lines += [f"### {LABEL[ds]}, lambda = {lam:g}", "",
                      HEADER.format("pooling p"), RULE, *sub, ""]
            detail["pool_sweep"].setdefault(ds, {})[f"{lam:g}"] = rows

    provenance = sorted({(c.dataset, c.arm, c.lam, c.pool, c.n_runs, "/".join(sorted(c.devices)))
                         for c in cells.values()})
    lines += ["## Cells", "", "| corpus | arm | lambda | p | runs | GPUs |", "|---|---|---:|---:|---:|---|"]
    lines += [f"| {d} | {a} | {l:g} | {p} | {n} | {g} |" for d, a, l, p, n, g in provenance]
    lines += ["", f"{sum(c.n_runs for c in cells.values())} runs in {len(cells)} cells."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(detail, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
