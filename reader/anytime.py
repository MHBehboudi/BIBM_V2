#!/usr/bin/env python3
"""Anytime comparison: ONE reader model against a bank of per-deadline retrained specialists.

The reader emits a legal decision at every token boundary (128 ms on 2a/2b, 15.625 ms on SD-SSVEP); its accuracy at deadline d is read
from the stored per-deadline curve of the SAME trained model. Each baseline is a DIFFERENT model,
trained from scratch on trials truncated to d (deadline-specialist mode), which is what a deployed
per-deadline system would have to do. Paired by subject and seed.
"""
import argparse, gzip, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS  # noqa: E402
from reader import subject_stats as st  # noqa: E402

# Wall-clock deadlines per corpus: 2a/2b are 4 s trials on a 128 ms token grid; SD-SSVEP is a 1 s trial
# on a 15.625 ms grid, so the same wall-clock deadlines do not exist there. The last entry is the full trial.
DEADLINES = {"2a": "1.0,2.0,3.0,4.0", "2b": "1.0,2.0,3.0,4.0", "hgd": "1.0,2.0,3.0,4.0",
             "sdssvep": "0.25,0.5,0.75,1.0"}


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
    ap.add_argument("--cohort", type=Path, default=ROOT / "runs/cohort")
    ap.add_argument("--bank", type=Path, default=ROOT / "runs/bank")
    ap.add_argument("--reader", type=Path, default=None,
                    help="Reader arm directory. Default <cohort>/reader, which is the "
                         "ENDPOINT-supervised arm. The headline comparison uses the "
                         "PREFIX-SUPERVISED arm (--prefix-weight 0.3), which trains as its own arm; "
                         "point this at it. --cohort still supplies the bank's 4 s row.")
    ap.add_argument("--dataset", default="2a")
    ap.add_argument("--pool-ms", type=float, default=None, help="token duration; default from the corpus spec")
    ap.add_argument("--deadlines", default=None, help="comma-separated seconds; default per corpus, last = full trial")
    ap.add_argument("--out", type=Path, default=None, help="default results/anytime_<dataset>.md")
    args = ap.parse_args()
    spec = SPECS[args.dataset]
    pool_ms = args.pool_ms or 1000.0 * spec["pool"] / spec["fs"]
    deadlines = [float(d) for d in (args.deadlines or DEADLINES[args.dataset]).split(",")]
    full_trial = deadlines[-1]                  # the last deadline IS the full trial (the cohort run)
    assert abs(full_trial - spec["samples"] / spec["fs"]) < 0.01, "last deadline must be the full trial"
    out_path = args.out or ROOT / "results" / f"anytime_{args.dataset}.md"

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
            idx = deadline_index(d, pool_ms, len(curve))
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
    # the full-trial specialist is the cohort run itself
    for family in {k[0] for k in bank}:
        for run in (args.cohort / family).glob(f"{args.dataset}_S*"):
            if (run / "summary.json").exists():
                s = json.loads((run / "summary.json").read_text())
                bank[(family, s["subject"], s["seed"])][full_trial] = 100 * s["final_test"]["acc"]

    families = sorted({k[0] for k in bank})
    rng = np.random.default_rng(st.RNG_SEED)
    lines = [f"# Anytime: one reader vs per-deadline specialist banks ({args.dataset})", "",
             "Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials "
             "truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its "
             "within-subject cohort run.", "",
             f"Reader arm: `{reader_dir.name}`. Prefix supervision moves the early deadlines by "
             "several points, so which arm this is forms part of the result, not metadata.", "",
             "The comparison of record is against the STRONGEST bank at each deadline (the most accurate "
             "retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the "
             "difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired "
             "bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.", "",
             "| deadline | reader (1 model) | " + " | ".join(f"{f} bank" for f in families) +
             " | reader minus STRONGEST bank, subject-level | " + " | ".join(f"vs {f} (runs)" for f in families) + " |",
             "|---|---|" + "---|" * (2 * len(families) + 1)]
    detail, family_tests = {}, {}
    for d in deadlines:
        r = {k: v[d] for k, v in reader.items() if d in v}
        cells, deltas, bank_means, stats_by = [], {}, {}, {}
        for f in families:
            b = {(s, sd): v[d] for (fam, s, sd), v in bank.items() if fam == f and d in v}
            shared = sorted(set(r) & set(b))
            cells.append(f"{np.mean([b[k] for k in shared]):.2f} ({len(shared)})" if shared else "-")
            if shared:
                deltas[f] = float(np.mean([r[k] - b[k] for k in shared]))
                bank_means[f] = float(np.mean([b[k] for k in shared]))
                stats_by[f] = st.summarise(st.subject_diffs(r, b), rng)
        # strongest = the most accurate specialist family at this deadline (smallest READER advantage).
        # An earlier version reported max(delta), i.e. the WEAKEST bank; that overstated the anytime gain.
        strongest = max(bank_means, key=bank_means.get) if bank_means else None
        if strongest and d != full_trial:
            family_tests[d] = stats_by[strongest]
        detail[d] = {"reader_mean": float(np.mean(list(r.values()))) if r else None,
                     "reader_n": len(r), "paired_delta_vs": deltas, "bank_means": bank_means,
                     "strongest_bank": strongest,
                     "paired_delta_vs_strongest": deltas.get(strongest) if strongest else None,
                     "subject_level_vs": stats_by}
        detail[d]["_row"] = (cells, strongest)
    holm_ok = st.holm(family_tests) if family_tests else False
    for d in deadlines:
        cells, strongest = detail[d].pop("_row")
        label = f"{d:g} s" + (" (full trial)" if d == full_trial else "")
        vs = detail[d]["subject_level_vs"]
        lines.append(f"| {label} | {detail[d]['reader_mean']:.2f} ({detail[d]['reader_n']}) | " + " | ".join(cells) +
                     f" | {(strongest + ': ' + st.fmt(vs[strongest])) if strongest else 'no bank'} | " +
                     " | ".join(f"{detail[d]['paired_delta_vs'][f]:+.2f}" if f in detail[d]["paired_delta_vs"] else "-"
                                for f in families) + " |")
    if not families:
        lines += ["", "No specialist bank exists for this corpus yet."]
    out_path.write_text("\n".join(lines) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps(
        {"dataset": args.dataset, "reader_arm": reader_dir.name, "pool_ms": pool_ms, "full_trial_s": full_trial,
         "holm_over_early_deadlines": holm_ok, "deadlines": detail}, indent=1, default=str))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
