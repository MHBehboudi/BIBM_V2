#!/usr/bin/env python3
"""Score a run tree into the paper's tables.

Writes results/<name>.json and prints the table. Subject means are formed FIRST (seeds are not
independent observations), then summarised across subjects. Paired comparisons are per
(dataset, subject, seed) against the named reference arm -- never a difference of two means.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["2a", "2b", "hgd", "sdssvep"]
PUBLISHED_WITHIN = {"2a": 85.34, "2b": 88.37, "hgd": 95.93, "sdssvep": 94.16}
BEST_PUBLISHED_WITHIN = {"2a": 85.34, "2b": 88.37, "hgd": 95.93, "sdssvep": 95.50}  # SD: DeepConvNet
PUBLISHED_LOSO = {"2a": 64.56, "2b": 76.44, "sdssvep": 79.72}                        # BiTE, no HGD row


def load(tree: Path):
    cells = collections.defaultdict(dict)
    for f in sorted(tree.glob("*/*/summary.json")):
        d = json.loads(f.read_text())
        if d.get("status") != "complete":
            continue
        cells[(d["dataset"], d["subject"], d["seed"])][f.parts[-3]] = d["final_test"]["acc"] * 100
    return cells


def subject_means(cells, arm):
    by = collections.defaultdict(list)
    for (ds, sub, _), arms in cells.items():
        if arm in arms:
            by[(ds, sub)].append(arms[arm])
    out = collections.defaultdict(list)
    for (ds, _), v in by.items():
        out[ds].append(float(np.mean(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", default="bite", help="arm to pair against")
    ap.add_argument("--protocol", choices=("within", "loso"), default="within")
    args = ap.parse_args()

    cells = load(args.runs)
    arms = sorted({a for v in cells.values() for a in v})
    bars = PUBLISHED_LOSO if args.protocol == "loso" else BEST_PUBLISHED_WITHIN
    out = {"protocol": args.protocol, "n_cells": len(cells), "arms": {}, "paired": {}}

    print(f"\n{'arm':10s} " + " ".join(f"{d:>9s}" for d in ORDER) + f" {'corpus-bal':>11s}")
    for arm in arms:
        m = subject_means(cells, arm)
        row = {ds: float(np.mean(m[ds])) for ds in ORDER if m.get(ds)}
        bal = float(np.mean(list(row.values()))) if len(row) == len(ORDER) else None
        out["arms"][arm] = {"per_corpus": row, "corpus_balanced": bal,
                            "n_subjects": {ds: len(m[ds]) for ds in row}}
        print(f"{arm:10s} " + " ".join(f"{row[d]:9.2f}" if d in row else f"{'-':>9s}" for d in ORDER)
              + (f" {bal:11.2f}" if bal else f" {'-':>11s}"))
    print(f"{'BAR':10s} " + " ".join(f"{bars[d]:9.2f}" if d in bars else f"{'-':>9s}" for d in ORDER))

    if args.reference in arms:
        print(f"\npaired vs {args.reference} (per subject, seed):")
        for arm in arms:
            if arm == args.reference:
                continue
            per = collections.defaultdict(list)
            for (ds, _, _), a in cells.items():
                if arm in a and args.reference in a:
                    per[ds].append(a[arm] - a[args.reference])
            allv = [x for v in per.values() for x in v]
            if not allv:
                continue
            bal = float(np.mean([np.mean(per[ds]) for ds in ORDER if per.get(ds)]))
            out["paired"][arm] = {"corpus_balanced": bal, "n": len(allv),
                                  "per_corpus": {ds: float(np.mean(v)) for ds, v in per.items()},
                                  "se": float(np.std(allv, ddof=1) / np.sqrt(len(allv)))}
            print(f"  {arm:10s} n={len(allv):3d} corpus-bal {bal:+6.2f} se {out['paired'][arm]['se']:.2f}  "
                  + " ".join(f"{ds} {np.mean(per[ds]):+.2f}" for ds in ORDER if per.get(ds)))

    dest = ROOT / "results" / f"{args.name}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
