#!/usr/bin/env python3
"""Anytime comparison: ONE reader model against a bank of per-deadline retrained specialists.

The reader emits a legal decision at every 128 ms token boundary; its accuracy at deadline d is read
from the stored per-deadline curve of the SAME trained model. Each baseline is a DIFFERENT model,
trained from scratch on trials truncated to d (deadline-specialist mode), which is what a deployed
per-deadline system would have to do. Paired by subject and seed.
"""
import argparse, gzip, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def deadline_index(deadline_s: float, pool_ms: float, n_tokens: int) -> int:
    """Index of the token reported for a wall-clock deadline.

    ceil, not round: the token must END AT OR AFTER the deadline, otherwise we report a number the
    model reached with less signal than claimed. On the 128 ms grid round() put the "3.0 s" column
    on a token ending at 2944 ms.
    """
    return min(int(math.ceil(deadline_s * 1000 / pool_ms - 1e-9)) - 1, n_tokens - 1)


def reader_curve(run: Path):
    """Per-deadline test accuracy of one trained reader, from its final diagnostic epoch."""
    rows = [json.loads(l) for l in gzip.open(run / "diag.jsonl.gz", "rt")]
    m = rows[-1].get("mechanism") or {}
    return m.get("anytime_test_token_acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, default=ROOT / "runs/cohort_20260915")
    ap.add_argument("--bank", type=Path, default=ROOT / "runs/bank_20260915")
    ap.add_argument("--reader", type=Path, default=None,
                    help="Reader arm directory. Default <cohort>/reader, which is the "
                         "ENDPOINT-supervised arm. The headline comparison uses the "
                         "PREFIX-SUPERVISED arm (--prefix-weight 0.3), which trains as its own arm; "
                         "point this at it. --cohort still supplies the bank's 4 s row.")
    ap.add_argument("--dataset", default="2a")
    ap.add_argument("--pool-ms", type=float, default=128.0)
    ap.add_argument("--deadlines", default="1.0,2.0,3.0,4.0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/ANYTIME_REPORT.md")
    args = ap.parse_args()
    deadlines = [float(d) for d in args.deadlines.split(",")]

    reader_dir = args.reader or (args.cohort / "reader")
    reader = defaultdict(dict)       # (subject, seed) -> {deadline: acc}
    for run in sorted(reader_dir.glob(f"{args.dataset}_S*")):
        if not (run / "summary.json").exists():
            continue
        s = json.loads((run / "summary.json").read_text())
        curve = reader_curve(run)
        if not curve:
            continue
        for d in deadlines:
            idx = deadline_index(d, args.pool_ms, len(curve))
            reader[(s["subject"], s["seed"])][d] = 100 * curve[idx]

    bank = defaultdict(dict)
    for arm in sorted(args.bank.glob("*")):
        if not arm.is_dir():
            continue
        family, _, tag = arm.name.rpartition("_")
        if not tag.endswith("s"):
            continue
        d = float(tag[:-1])
        for run in arm.glob(f"{args.dataset}_S*"):
            if (run / "summary.json").exists():
                s = json.loads((run / "summary.json").read_text())
                bank[(family, s["subject"], s["seed"])][d] = 100 * s["final_test"]["acc"]
    # the 4 s (full-trial) specialist is the cohort run itself
    for family in {k[0] for k in bank}:
        for run in (args.cohort / family).glob(f"{args.dataset}_S*"):
            if (run / "summary.json").exists():
                s = json.loads((run / "summary.json").read_text())
                bank[(family, s["subject"], s["seed"])][4.0] = 100 * s["final_test"]["acc"]

    families = sorted({k[0] for k in bank})
    lines = [f"# Anytime: one reader vs per-deadline specialist banks ({args.dataset})", "",
             "Each baseline column at deadline d is a **separately trained model**; the reader column is",
             "**one model** read at d. Paired by subject and seed.", "",
             f"Reader arm: `{reader_dir.name}`. Prefix supervision moves the early deadlines by "
             "several points, so which arm this is forms part of the result, not metadata.", "",
             "The comparison of record is against the STRONGEST bank at each deadline (the most accurate",
             "retrained specialist), not the weakest; every family's paired delta is also listed.", "",
             "| deadline | reader (1 model) | " + " | ".join(f"{f} bank" for f in families) +
             " | paired delta vs STRONGEST bank | " + " | ".join(f"vs {f}" for f in families) + " |",
             "|---|---|" + "---|" * (2 * len(families) + 1)]
    detail = {}
    for d in deadlines:
        r = {k: v[d] for k, v in reader.items() if d in v}
        cells, deltas, bank_means = [], {}, {}
        for f in families:
            b = {(s, sd): v[d] for (fam, s, sd), v in bank.items() if fam == f and d in v}
            shared = sorted(set(r) & set(b))
            cells.append(f"{np.mean([b[k] for k in shared]):.2f} ({len(shared)})" if shared else "-")
            if shared:
                deltas[f] = float(np.mean([r[k] - b[k] for k in shared]))
                bank_means[f] = float(np.mean([b[k] for k in shared]))
        # strongest = the most accurate specialist family at this deadline (smallest READER advantage).
        # An earlier version reported max(delta), i.e. the WEAKEST bank; that overstated the anytime gain.
        strongest = max(bank_means, key=bank_means.get) if bank_means else None
        lines.append(f"| {d:.1f} s | {np.mean(list(r.values())):.2f} ({len(r)}) | " + " | ".join(cells) +
                     f" | {('%+.2f vs %s' % (deltas[strongest], strongest)) if strongest else '-'} | " +
                     " | ".join(f"{deltas[f]:+.2f}" if f in deltas else "-" for f in families) + " |")
        detail[d] = {"reader_mean": float(np.mean(list(r.values()))) if r else None,
                     "reader_n": len(r), "paired_delta_vs": deltas, "strongest_bank": strongest,
                     "paired_delta_vs_strongest": deltas.get(strongest) if strongest else None}
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"dataset": args.dataset, "reader_arm": reader_dir.name, "pool_ms": args.pool_ms,
         "deadlines": detail}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
