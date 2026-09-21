#!/usr/bin/env python3
"""Enumerate the sensitivity cells, evaluate the ones that are missing, and write the scorer's plan.

The two sweeps (prefix-supervision weight, token resolution) are anchored on cells that already
exist in the record and only add the settings that do not:

  lambda 0.0, default p  ->  runs/cohort/{reader,compact}                (the within-subject table)
  lambda 0.3, default p  ->  runs/anytime/{reader,compact}_anytime       (the prefix-supervision ablation)
  everything else        ->  <harness>/runs/sens_20260920/<arm>_p<p>_ps<lambda>

Every cell is summarised the same way, by `reader/exact_duration.py` on the trained checkpoint: the
model is given only the first n samples of each test trial and its endpoint prediction is scored, on
a grid in SAMPLES that does not depend on the pooling factor. `--accuracy-only` skips the gate
interventions and the trained-weight causality checks, which are established on the endpoint-trained
checkpoints of record and are not what a sensitivity sweep is asking about.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Where the sweep's runs live. `runs/sensitivity` is what scripts/reproduce_sensitivity.sh writes;
# SENS_RUNS points at the development harness that produced the runs of record for this paper.
ROOTS = [ROOT / "runs/sensitivity",
         Path(os.environ.get("SENS_RUNS",
                             "/work/mxb190076/BIBM2026/bite_workshop/fusion/runs/sens_20260920"))]
EXACT = ROOT / "runs/exact_duration"
DEFAULT_POOL = {"2a": 32, "2b": 32, "sdssvep": 4}
SWEEP_POOLS = {"2a": (64, 16), "2b": (64, 16), "sdssvep": (16, 8)}
LAMBDAS = (0.0, 0.1, 0.3, 1.0)
ANCHOR = {0.0: {"reader": "cohort/reader", "compact": "cohort/compact"},
          0.3: {"reader": "anytime/reader_anytime", "compact": "anytime/compact_anytime"}}


def names(arm, pool, lam):
    """Cell directory names, newest spelling first.

    The runs of record were written with the weight's leading zero stripped (`ps.3`); the reproduce
    script writes it in full (`ps0.3`). Both are accepted so the same collector reads either.
    """
    return [f"{arm}_p{pool}_ps{lam:g}", f"{arm}_p{pool}_ps{('%g' % lam).replace('0.', '.')}"]


def find_cell(arm, pool, lam):
    """(run_dir, label) for the first cell directory that exists, else (None, canonical label)."""
    for root in ROOTS:
        for name in names(arm, pool, lam):
            if (root / name).is_dir():
                return root / name, name
    return None, names(arm, pool, lam)[0]


#   Q1 moves lambda at the default p; Q2 moves p at the two lambdas the paper reports (the
#   endpoint-only loss of the main table, and the 0.3 the nAUC claim is made at). The full
#   lambda x p grid is not run: neither question needs it and it would quadruple the GPU cost.
POOL_SWEEP_LAMBDAS = (0.0, 0.3)


def cells():
    """(dataset, arm, lambda, pool, run_dir, exact_label) for every cell of both sweeps."""
    for ds in ("2a", "2b", "sdssvep"):
        for lam in LAMBDAS:
            pools = (DEFAULT_POOL[ds],)
            if lam in POOL_SWEEP_LAMBDAS:
                pools += SWEEP_POOLS[ds]
            for pool in pools:
                for arm in ("reader", "compact"):
                    if pool == DEFAULT_POOL[ds] and lam in ANCHOR:
                        run = ROOT / "runs" / ANCHOR[lam][arm]
                        label = Path(ANCHOR[lam][arm]).name
                    else:
                        found, label = find_cell(arm, pool, lam)
                        run = found or ROOTS[0] / label
                    yield ds, arm, lam, pool, run, label


def complete(run_dir: Path, dataset: str):
    return sorted(p.parent for p in run_dir.glob(f"{dataset}_S*_seed*/summary.json"))


def fresh(exact_dir: Path, run_dirs):
    """Records that were computed from the run of record as it stands NOW.

    A run can be replaced under the same name -- a MIG original by its full-GPU re-run -- which
    leaves the record count unchanged. Counting files would then call the cell done and keep the
    superseded numbers, so the record is matched against its run the same way
    `reader/exact_duration.py` matches it before recomputing.
    """
    n = 0
    for run in run_dirs:
        target = exact_dir / f"{run.name}.json"
        if not target.exists():
            continue
        rec = json.loads(target.read_text())
        summary = json.loads((run / "summary.json").read_text())
        if rec.get("device_trained") == summary.get("device") and \
                abs(float(rec.get("final_test_acc_logged", -1)) - summary["final_test"]["acc"]) < 1e-9:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--plan", type=Path, default=ROOT / "results/sensitivity_plan.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # One collector at a time. Two of them evaluate the same cells into the same files, which wastes
    # the machine and, if they ever finish a file together, can leave a torn record.
    lock_path = EXACT.parent / ".sensitivity_collect.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another collector holds {lock_path}; not starting a second one")

    plan, jobs, missing = [], [], []
    for ds, arm, lam, pool, run, label in cells():
        runs = complete(run, ds) if run.exists() else []
        exact_dir = EXACT / ds / label
        done = fresh(exact_dir, runs) if exact_dir.exists() else 0
        if not runs:
            missing.append((ds, arm, lam, pool, label, 0))
            continue
        if done < len(runs):
            jobs.append([args.python, str(ROOT / "reader/exact_duration.py"), "--dataset", ds,
                         "--arm", arm, "--runs", str(run), "--label", label,
                         "--accuracy-only", "--threads", "2"]
                        + ([] if pool == DEFAULT_POOL[ds] else ["--option", f"pool={pool}"]))
        plan.append({"dataset": ds, "arm": arm, "lambda": lam, "pool": pool,
                     "runs": str(run), "exact": str(exact_dir), "trained": len(runs)})
        if done < len(runs):
            missing.append((ds, arm, lam, pool, label, len(runs)))

    for ds, arm, lam, pool, label, n in missing:
        print(f"  {'trained ' + str(n) if n else 'NOT TRAINED':>12s}  {ds:8s} {label}")
    print(f"{len(plan)} cells with runs, {len(jobs)} need exact-duration evaluation")
    if args.dry_run:
        return

    def run_one(cmd):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        name = cmd[cmd.index("--label") + 1]
        if proc.returncode:
            print(f"FAILED {name}: {proc.stderr.strip().splitlines()[-1] if proc.stderr else '?'}")
        else:
            print(f"evaluated {cmd[4]:8s} {name}")
        return proc.returncode

    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool_:
            codes = list(pool_.map(run_one, jobs))
        if any(codes):
            raise SystemExit("some evaluations failed")

    args.plan.write_text(json.dumps(plan, indent=1))
    print(f"plan -> {args.plan} ({len(plan)} cells)")


if __name__ == "__main__":
    main()
