#!/usr/bin/env python3
"""Write one line of train.py arguments per (arm, cell, seed)."""
import argparse
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SUBJECTS = {"2a": 9, "2b": 9, "hgd": 14, "sdssvep": 10}
# Hardest and easiest subject per corpus by fixed-final reference result: the 8-cell screen.
SCREEN = [("2a", 2), ("2a", 9), ("2b", 9), ("2b", 4), ("hgd", 11), ("hgd", 14), ("sdssvep", 1), ("sdssvep", 5)]


def cell_group(name):
    """screen | all | all_no_hgd | <corpus> | explicit '2a:1;2b:3'."""
    if name == "screen":
        return SCREEN
    if name in ("all", "all_no_hgd"):
        corpora = [c for c in SUBJECTS if not (name == "all_no_hgd" and c == "hgd")]
        return [(c, s) for c in corpora for s in range(1, SUBJECTS[c] + 1)]
    if name in SUBJECTS:
        return [(name, s) for s in range(1, SUBJECTS[name] + 1)]
    return [(c.split(":")[0], int(c.split(":")[1])) for c in name.split(";")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="label:model[:extra args separated by ,] e.g. compact_pw03:compact:--prefix-weight,0.3")
    ap.add_argument("--seeds", default="2025,2026,2027")
    ap.add_argument("--cells", default="screen")
    ap.add_argument("--study", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cells = cell_group(args.cells)
    lines = []
    for arm, (ds, sub), seed in itertools.product(args.arm, cells, [int(s) for s in args.seeds.split(",")]):
        label, model, *extra = arm.split(":")
        extra = extra[0].split(",") if extra else []
        out = RUNS / args.study / label / f"{ds}_S{sub}_seed{seed}"
        lines.append(" ".join(["--model", model, "--dataset", ds, "--subject", str(sub), "--seed", str(seed),
                               *extra, "--out", str(out)]))
    # interleave heavy HGD runs so a pack never holds only HGD
    lines.sort(key=lambda l: ("hgd" in l, l))
    heavy = [l for l in lines if "--dataset hgd" in l]
    light = [l for l in lines if "--dataset hgd" not in l]
    mixed = []
    while heavy or light:
        if heavy:
            mixed.append(heavy.pop(0))
        for _ in range(3):
            if light:
                mixed.append(light.pop(0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(mixed) + "\n")
    print(len(mixed), "lines ->", args.output)


if __name__ == "__main__":
    main()
