#!/usr/bin/env python3
"""Assemble the RUNS OF RECORD that every table in results/ is computed from.

The paper's runs were trained in a development harness under dated study directories. This script links
them into the layout this repository's scorers read (runs/cohort, runs/controls, runs/anytime, runs/bank,
runs/loso, runs/zoo), applying two declared rules (run card 2026-09-17, fixed before any re-run finished):

1. HARDWARE. A MIG-partitioned GPU slice does not reproduce a full-GPU run bit-for-bit (H100 80GB and
   H200 NVL do). Any cell whose original ran on a MIG slice is represented by its full-GPU re-run once
   that re-run exists; the MIG original is linked under runs/mig/ for the reproducibility report. Until
   the re-run exists the original stays in the record and is flagged `mig_pending`.
2. BiTE CROSS-SUBJECT CLIPPING. BiTE's release does not clip gradients and every within-subject BiTE
   run here uses --clip 0. The first cross-subject BiTE seed was run at the trainer default (clip 5) by
   mistake; the record uses the --clip 0 re-runs only, and the clip-5 runs are linked under
   runs/superseded/ so the difference can be reported.

Writes runs/RECORD.json (local) and results/raw/record_provenance.csv (committed): one row per linked run
with its source directory, GPU, and why it was chosen. A repository clone that trains its own runs with the
reproduce_*.sh scripts writes this layout directly and does not need this script.
"""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARNESS = ROOT.parent / "bite_workshop" / "fusion" / "runs"

# view -> ordered list of (harness study/arm directory, cell filter). Later sources never override earlier
# ones for the same cell; MIG replacement is handled separately via fullgpu_20260917/<study>/<arm>/<cell>.
VIEWS = {
    "cohort/bite": ["cohort_20260915/bite"],
    "cohort/compact": ["cohort_20260915/compact"],
    "cohort/reader": ["cohort_20260915/reader"],
    "controls/ff_control": ["controls_20260916/ff_reader"],
    "controls/compact_mean": ["controls_20260916/compact_mean"],
    "anytime/reader_anytime": ["anytime_20260915/reader_anytime"],
    "anytime/compact_anytime": ["anytime_20260917/compact_anytime"],
    "loso/bite": ["loso_20260917/bite"],
    "loso/compact": ["loso_20260915/compact", "loso_20260917/compact"],
    "loso/reader": ["loso_20260915/reader", "loso_20260917/reader"],
    "superseded/loso_bite_clip5": ["loso_20260915/bite"],
}
BANK_STUDIES = ["bank_20260915", "bank_20260917"]
ZOO_STUDY = "zoo_20260916"


def summary(run: Path):
    p = run / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s if s.get("status") == "complete" else None


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, default=DEFAULT_HARNESS, help="directory holding the dated studies")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    h = args.harness.resolve()

    views = dict(VIEWS)
    for study in BANK_STUDIES:
        for arm in sorted((h / study).glob("*")) if (h / study).exists() else []:
            if arm.is_dir():
                views.setdefault(f"bank/{arm.name}", []).append(f"{study}/{arm.name}")
    if (h / ZOO_STUDY).exists():
        for arm in sorted((h / ZOO_STUDY).glob("*")):
            if arm.is_dir():
                views[f"zoo/{arm.name}"] = [f"{ZOO_STUDY}/{arm.name}"]

    rows = []
    for view, sources in sorted(views.items()):
        seen = set()
        for source in sources:
            for run in sorted((h / source).glob("*_S*_seed*")):
                s = summary(run)
                if s is None or run.name in seen:
                    continue
                seen.add(run.name)
                chosen, reason, mig_src = run, "original", None
                if "MIG" in s["device"]:
                    rerun = h / "fullgpu_20260917" / source / run.name
                    rs = summary(rerun)
                    if rs is not None and "MIG" not in rs["device"]:
                        chosen, reason, mig_src = rerun, "full-GPU re-run replaces MIG original", run
                        s = rs
                    else:
                        reason = "mig_pending"
                link(args.runs / view / run.name, chosen)
                if mig_src is not None:
                    link(args.runs / "mig" / view / run.name, mig_src)
                rows.append({"view": view, "cell": run.name, "source": str(chosen.relative_to(h)),
                             "device": s["device"].replace("NVIDIA ", ""), "status": reason,
                             "mig_original": str(mig_src.relative_to(h)) if mig_src else ""})

    # Exact-duration evaluations already computed in the harness (CPU, deterministic). Copied, not linked:
    # reader/exact_duration.py recomputes any record whose run of record has since been replaced.
    exact = h.parent / "analysis" / "exact_duration"
    if exact.exists():
        import shutil
        for src in sorted(exact.glob("*/*/*")):
            ds, arm = src.parent.parent.name, src.parent.name
            dst = args.runs / "exact_duration" / ds / {"ff_reader": "ff_control"}.get(arm, arm) / src.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    args.runs.mkdir(parents=True, exist_ok=True)
    (args.runs / "RECORD.json").write_text(json.dumps({"harness": str(h), "runs": rows}, indent=1))
    out = ROOT / "results" / "raw" / "record_provenance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    counts = Counter((r["view"].split("/")[0], r["status"]) for r in rows)
    for (group, status), n in sorted(counts.items()):
        print(f"{group:12s} {status:40s} {n}")
    print(f"{len(rows)} runs linked -> {args.runs}; provenance -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
