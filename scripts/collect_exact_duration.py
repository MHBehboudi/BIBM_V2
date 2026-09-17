#!/usr/bin/env python3
"""Collect reader/exact_duration.py outputs (runs/exact_duration/<ds>/<arm>/*.json) into results/exact_duration.json.

Only records whose run is still the run of record are kept: a record computed from a checkpoint that has since
been replaced (a MIG original superseded by its full-GPU re-run) is dropped, and exact_duration.py recomputes it.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ("Exact-duration, same-checkpoint accuracy: each trained checkpoint given ONLY the first n samples "
               "(partial last pooling window kept, endpoint of the truncated input). Gate interventions g in {0,.5,1} "
               "for two-branch arms. Causality on trained weights: max |logit| discrepancy truncated-vs-curve at every "
               "pooling boundary, and under noise after the boundary. Produced by reader/exact_duration.py.")
# exact_duration/<ds>/<arm> -> the run-of-record directory it was computed from
SOURCES = {"reader": "cohort/reader", "compact": "cohort/compact", "bite": "cohort/bite",
           "ff_control": "controls/ff_control", "compact_mean": "controls/compact_mean",
           "reader_anytime": "anytime/reader_anytime", "compact_anytime": "anytime/compact_anytime"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", type=Path, default=ROOT / "runs/exact_duration")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/exact_duration.json")
    args = ap.parse_args()
    records, stale, causality = [], [], defaultdict(lambda: {"checkpoints": 0, "boundaries": 0,
                                                              "trunc_max": 0.0, "future_max": 0.0})
    for p in sorted(args.exact.glob("*/*/*.json")):
        arm = p.parent.name
        r = json.loads(p.read_text())
        r["run"] = Path(r["run"]).name          # harness records store the full run path
        run = args.runs / SOURCES.get(arm, arm) / r["run"] / "summary.json"
        if not run.exists():
            stale.append(str(p.relative_to(args.exact)))
            continue
        s = json.loads(run.read_text())
        if r["device_trained"] != s["device"] or abs(float(r["final_test_acc_logged"]) - s["final_test"]["acc"]) > 1e-9:
            stale.append(str(p.relative_to(args.exact)))
            continue
        r.pop("seconds", None)
        r["arm"] = arm
        records.append(r)
        if r.get("causality"):
            c = causality[f"{r['dataset']}|{arm}"]
            c["checkpoints"] += 1
            c["boundaries"] += r["causality"]["boundaries_tested"]
            c["trunc_max"] = max(c["trunc_max"], r["causality"]["trunc_max_abs_logit_diff"])
            c["future_max"] = max(c["future_max"], r["causality"]["future_max_abs_logit_diff"])
    out = {"description": DESCRIPTION, "records": records, "causality_summary": dict(sorted(causality.items())),
           "all_full_length_match_logged": all(r["full_length_matches_logged"] for r in records)}
    args.out.write_text(json.dumps(out, indent=1))
    from collections import Counter
    print(Counter((r["dataset"], r["arm"]) for r in records))
    print(f"{len(records)} records -> {args.out.relative_to(ROOT)}; stale (run of record replaced, recompute): {len(stale)}")
    for s_ in stale[:10]:
        print("  stale:", s_)


if __name__ == "__main__":
    main()
