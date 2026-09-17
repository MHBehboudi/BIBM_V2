#!/usr/bin/env python3
"""Export every run of record to flat CSV files, the ground truth every table in results/ is computed from.

results/raw/runs.csv            one row per run: arm, cell, protocol, deadline window, loss, parameters,
                                final-epoch test accuracy / balanced accuracy / Cohen's kappa / NLL / ECE,
                                final training accuracy, the test-curve PEAK (diagnostic only, never used for
                                selection), GPU, wall time, and which record rule selected it
results/raw/anytime_curves.csv  READER-family runs: test accuracy at every token boundary of the SAME trained
                                model (the anytime curve), one row per (run, token)

Kappa is computed from the saved final-epoch test logits (test_logits.npz), the same predictions the accuracy
comes from. Reads the record layout written by scripts/link_record_runs.py (or by the reproduce scripts).
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from reader.data import SPECS  # noqa: E402


def cohen_kappa(labels, predictions, n_classes):
    confusion = np.zeros((n_classes, n_classes))
    np.add.at(confusion, (labels, predictions), 1)
    total = confusion.sum()
    observed = np.trace(confusion) / total
    expected = (confusion.sum(0) * confusion.sum(1)).sum() / total ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def record_status(runs: Path):
    p = runs / "RECORD.json"
    if not p.exists():
        return {}
    return {(r["view"], r["cell"]): r for r in json.loads(p.read_text())["runs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "raw")
    args = ap.parse_args()
    status = record_status(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)

    views = []
    for group in ("cohort", "controls", "anytime", "bank", "loso", "zoo", "superseded", "mig"):
        base = args.runs / group
        if not base.exists():
            continue
        if group == "mig":                       # runs/mig/<group>/<arm>/<cell>
            views += [(f"mig/{g.name}/{a.name}", a) for g in sorted(base.iterdir()) for a in sorted(g.iterdir())]
        else:
            views += [(f"{group}/{a.name}", a) for a in sorted(base.iterdir()) if a.is_dir()]

    rows, curves = [], []
    for view, arm_dir in views:
        for run in sorted(arm_dir.glob("*_S*_seed*")):
            sp = run / "summary.json"
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            if s.get("status") != "complete":
                continue
            kappa = ""
            lp = run / "test_logits.npz"
            if lp.exists():
                z = np.load(lp)
                kappa = cohen_kappa(z["labels"], z["logits"].argmax(1), SPECS[s["dataset"]]["classes"])
            rec = status.get((view, run.name), {})
            if view.startswith("mig/"):
                rec = {"status": "MIG original (replaced in the record)", "source": ""}
            peak = s.get("test_peak_diagnostic") or {}
            rows.append({
                "group": view.split("/")[0], "arm": view.split("/", 1)[1], "dataset": s["dataset"],
                "subject": s["subject"], "seed": s["seed"],
                "protocol": (s.get("protocol") or {}).get("name", "within"),
                "window_seconds": s.get("window_seconds") or "", "prefix_weight": s.get("prefix_weight", 0.0),
                "model": s["model"], "parameters": s["parameters"], "epochs": s["epochs"],
                "test_acc": round(100 * s["final_test"]["acc"], 4),
                "test_balanced_acc": round(100 * s["final_test"]["balanced_acc"], 4),
                "test_kappa": round(kappa, 4) if kappa != "" else "",
                "test_nll": round(s["final_test"]["nll"], 4), "test_ece": round(s["final_test"]["ece"], 4),
                "train_acc": round(100 * s["final_train"]["acc"], 4) if s.get("final_train") else "",
                "test_peak_acc_diagnostic": round(100 * peak["acc"], 4) if peak else "",
                "test_peak_epoch_diagnostic": peak.get("epoch", ""),
                "device": s.get("device", "").replace("NVIDIA ", ""),
                "hours": round(s.get("elapsed_seconds", float("nan")) / 3600, 3),
                "record_status": rec.get("status", "original"), "harness_source": rec.get("source", ""),
            })
            dp = run / "diag.jsonl.gz"
            if s["model"] == "reader" and dp.exists() and not view.startswith("mig/"):
                last = None
                for line in gzip.open(dp, "rt"):
                    last = line
                curve = (json.loads(last).get("mechanism") or {}).get("anytime_test_token_acc") if last else None
                if curve:
                    spec = SPECS[s["dataset"]]
                    ms = 1000 * spec["pool"] / spec["fs"]
                    for i, a in enumerate(curve):
                        curves.append({"arm": view.split("/", 1)[1], "group": view.split("/")[0],
                                       "dataset": s["dataset"], "subject": s["subject"], "seed": s["seed"],
                                       "protocol": rows[-1]["protocol"], "token": i + 1,
                                       "ends_at_ms": round((i + 1) * ms, 3), "test_acc": round(100 * a, 4)})

    for name, data in (("runs.csv", rows), ("anytime_curves.csv", curves)):
        with (args.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"{len(data):6d} rows -> {(args.out / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
