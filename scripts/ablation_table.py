#!/usr/bin/env python3
"""One ablation table for the paper, computed only from committed files:
results/raw/runs.csv, results/exact_duration.json and results/gate_intervention.json.

Three kinds of ablation, kept apart because they answer different questions:
  A  ARCHITECTURE, retrained from scratch with the same endpoint loss:
       Compact        the prefix-reversed branch removed (READER's parent)
       FF-Control     the second branch kept, capacity- and init-matched, but reading the prefix FORWARD
       Compact-Mean   no second branch; the classifier reads the causal running mean (a readout change)
  B  INFERENCE-TIME interventions on the SAME trained READER weights (no retraining):
       g = 0 forward route only, g = 1 reversed route only, g = 0.5 the unlearned initial average
  C  TRAINING OBJECTIVE: prefix supervision (cross-entropy on every intermediate decision, weight 0.3,
     the only weight ever trained), on READER and on Compact. READER+PS minus Compact+PS isolates the
     architecture under the identical loss.

Endpoint = full trial. Early = the same checkpoint given only the first quarter of the trial (2a/2b 1 s,
SD-SSVEP 0.25 s) and half of it (2 s / 0.5 s), exact truncated input. Differences are variant minus READER,
subject-level (seeds averaged per subject), 95% paired bootstrap CI, subject W/T/L.
Inference interventions at the endpoint are reported per run (mean, se, runs made worse) as recorded in training.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from reader import subject_stats as st  # noqa: E402

CORPORA = ["2a", "2b", "hgd", "sdssvep"]
EARLY = {"2a": (250, 500), "2b": (250, 500), "sdssvep": (64, 128)}
FS = {"2a": 250.0, "2b": 250.0, "sdssvep": 256.0}
NAMES = {"2a": "2a", "2b": "2b", "hgd": "HGD", "sdssvep": "SD-SSVEP"}
ARCH = [("cohort", "reader", "reader", "READER (endpoint loss)"),
        ("cohort", "compact", "compact", "A1  − reversed branch (Compact)"),
        ("controls", "ff_control", "ff_control", "A2  reversed → forward second branch (FF-Control)"),
        ("controls", "compact_mean", "compact_mean", "A3  second branch → running-mean readout (Compact-Mean)"),
        ("anytime", "reader_anytime", "reader_anytime", "C1  READER + prefix supervision"),
        ("anytime", "compact_anytime", "compact_anytime", "C2  Compact + prefix supervision")]
GATES = [("0.0", "gate_forward_only", "B1  g = 0: forward route only (inference)"),
         ("1.0", "gate_backward_only", "B2  g = 1: reversed route only (inference)"),
         ("0.5", "gate_half", "B3  g = 0.5: unlearned equal average (inference)")]


def fmt_d(r):
    if r is None:
        return "pending"
    return f"{r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] {r['wins']}/{r['ties']}/{r['losses']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    rng = np.random.default_rng(st.RNG_SEED)
    rows = list(csv.DictReader(open(args.results / "raw/runs.csv")))
    endpoint = defaultdict(dict)                 # (arm, ds) -> {(sub, seed): acc}
    for r in rows:
        if r["group"] in ("cohort", "controls", "anytime"):
            endpoint[(r["arm"], r["dataset"])][(int(r["subject"]), int(r["seed"]))] = float(r["test_acc"])
    exact = defaultdict(dict)                    # (arm, ds, n) -> {(sub, seed): acc}
    gate_exact = defaultdict(dict)               # (g, ds, n) -> {(sub, seed): acc}
    for rec in json.loads((args.results / "exact_duration.json").read_text())["records"]:
        k = (int(rec["subject"]), int(rec["seed"]))
        for n, a in rec["acc"].items():
            exact[(rec["arm"], rec["dataset"], int(n))][k] = 100 * float(a)
        if rec["arm"] == "reader":
            for g, curve in (rec.get("gate") or {}).items():
                for n, a in curve.items():
                    gate_exact[(g, rec["dataset"], int(n))][k] = 100 * float(a)
    gate_end = json.loads((args.results / "gate_intervention.json").read_text())["corpora"]

    columns = [("end", ds) for ds in CORPORA] + [("early", ds, i) for ds in EARLY for i in (0, 1)]

    def label(c):
        if c[0] == "end":
            return f"{NAMES[c[1]]} full"
        n = EARLY[c[1]][c[2]]
        return f"{NAMES[c[1]]} {n / FS[c[1]]:g} s"

    def values(arm, c):
        return endpoint.get((arm, c[1]), {}) if c[0] == "end" else exact.get((arm, c[1], EARLY[c[1]][c[2]]), {})

    def mean_of(acc):
        sm = st.subject_means(acc)
        return float(np.mean(list(sm.values()))) if sm else None

    report = {"absolute": {}, "minus_reader": {}, "reader_ps_minus_compact_ps": {}}
    head = "| variant | " + " | ".join(label(c) for c in columns) + " |"
    sep = "|---|" + "---:|" * len(columns)
    abs_lines = ["## Accuracy (%, subject-level mean over 3 seeds)", "", head, sep]
    delta_lines = ["## Variant minus READER (pp): mean [95% CI] subject W/T/L", "", head, sep]
    reader_vals = {c: values("reader", c) for c in columns}
    for group, arm, key, name in ARCH:
        cells_a, cells_d = [], []
        for c in columns:
            acc = values(key, c)
            m = mean_of(acc)
            planned = group == "anytime" and c[1] != "hgd"
            cells_a.append(f"{m:.2f}" if m is not None else ("pending" if planned else "—"))
            if key == "reader":
                cells_d.append("—")
                continue
            d = st.subject_diffs(acc, reader_vals[c]) if acc else {}
            r = st.summarise(d, rng) if len(d) > 1 else None
            cells_d.append(fmt_d(r) if r else ("pending" if planned else "—"))
            report["minus_reader"][f"{key}|{label(c)}"] = r
            report["absolute"][f"{key}|{label(c)}"] = m
        abs_lines.append(f"| {name} | " + " | ".join(cells_a) + " |")
        if key != "reader":
            delta_lines.append(f"| {name} | " + " | ".join(cells_d) + " |")
    for g, gkey, name in GATES:
        cells_a, cells_d = [], []
        for c in columns:
            if c[0] == "end":
                gi = gate_end.get(c[1], {}).get("interventions", {}).get(gkey)
                learned = gate_end.get(c[1], {}).get("learned_acc")
                cells_a.append(f"{learned + gi['mean']:.2f}" if gi else "—")
                cells_d.append(f"{gi['mean']:+.2f} (se {gi['se']:.2f}, {gi['worse']}/{gi['n']} runs worse)" if gi else "—")
                continue
            acc = gate_exact.get((g, c[1], EARLY[c[1]][c[2]]), {})
            m = mean_of(acc)
            cells_a.append(f"{m:.2f}" if m is not None else "—")
            r = st.summarise(st.subject_diffs(acc, reader_vals[c]), rng) if acc else None
            cells_d.append(fmt_d(r) if r else "—")
            report["minus_reader"][f"gate{g}|{label(c)}"] = r
        abs_lines.append(f"| {name} | " + " | ".join(cells_a) + " |")
        delta_lines.append(f"| {name} | " + " | ".join(cells_d) + " |")

    matched = ["## Architecture under the identical loss: (READER + PS) minus (Compact + PS)", "",
               "Both arms trained with prefix supervision at weight 0.3. A positive interval means the prefix-reversed "
               "branch helps beyond what the loss alone buys.", "", head, sep]
    cells = []
    for c in columns:
        d = st.subject_diffs(values("reader_anytime", c), values("compact_anytime", c))
        r = st.summarise(d, rng) if len(d) > 1 else None
        cells.append(fmt_d(r) if r else ("—" if c[1] == "hgd" else "pending"))
        report["reader_ps_minus_compact_ps"][label(c)] = r
    matched.append("| READER+PS − Compact+PS | " + " | ".join(cells) + " |")

    lines = ["# Ablations", "", (__doc__ or "").split("\n\n", 1)[1].strip(), "",
             *abs_lines, "", *delta_lines, "", *matched, "",
             "HGD has no exact-duration evaluation and no FF-Control / Compact-Mean / prefix-supervised runs "
             "(its 14 subjects make every READER run ~1 GPU-hour); its column is the endpoint only.", ""]
    (args.results / "ABLATION.md").write_text("\n".join(lines))
    (args.results / "ablation.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
