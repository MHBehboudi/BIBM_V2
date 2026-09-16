#!/usr/bin/env python3
"""Prefix supervision: one predeclared setting against the endpoint-only loss.

READER computes a decision at every deadline whether or not anything supervises those decisions.
The endpoint-only loss (`--prefix-weight 0`) trains the architecture and reads the intermediate
decisions for free; prefix supervision (`--prefix-weight 0.3`) adds a cross-entropy on them.

This is a LOSS ablation, not an architectural one, and it is reported separately for that reason.
Exactly ONE prefix weight has ever been trained on this model -- 0.3, declared before the runs --
so nothing here is selected on the test set. The question it answers is the one worth asking of any
deep-supervision scheme: does supervising the intermediate predictions improve EARLY decoding, and
what does it cost at the ENDPOINT?

Both arms must be identical apart from the loss. That is checked, not assumed: `_matched()` aborts
if the two arms differ in model, epochs, parameter count, input mode, protocol or window.
"""
import argparse, gzip, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from reader.data import SPECS  # noqa: E402

# Deadlines per corpus. 2a/2b are 4 s trials on a 128 ms grid; SD-SSVEP is a 1 s trial on a
# 15.625 ms grid, so the same wall-clock deadlines do not exist there.
DEADLINES = {"2a": [1.0, 2.0, 3.0, 4.0], "2b": [1.0, 2.0, 3.0, 4.0],
             "hgd": [1.0, 2.0, 3.0, 4.0], "sdssvep": [0.25, 0.5, 0.75, 1.0]}
MATCH_ON = ("model", "epochs", "parameters", "input_mode", "protocol", "window_seconds")


def deadline_index(deadline_s: float, pool_ms: float, n_tokens: int) -> int:
    """ceil: the reported token must END AT OR AFTER the deadline (see anytime.py)."""
    return min(int(math.ceil(deadline_s * 1000 / pool_ms - 1e-9)) - 1, n_tokens - 1)


def load_arm(arm_dir: Path):
    """(dataset, subject, seed) -> (summary, endpoint acc, per-deadline curve)."""
    out = {}
    for summary_path in sorted(arm_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        rows = [json.loads(l) for l in gzip.open(summary_path.parent / "diag.jsonl.gz", "rt")]
        curve = (rows[-1].get("mechanism") or {}).get("anytime_test_token_acc")
        out[(s["dataset"], s["subject"], s["seed"])] = (s, 100 * s["final_test"]["acc"], curve)
    return out


def _matched(a, b, keys):
    """The ablation is only an ablation if one thing differs. Verify it."""
    sa, sb = a[keys[0]][0], b[keys[0]][0]
    diffs = {k: (sa.get(k), sb.get(k)) for k in MATCH_ON if sa.get(k) != sb.get(k)}
    if diffs:
        raise SystemExit(f"arms differ in more than the loss: {diffs}")
    wa, wb = sa.get("prefix_weight"), sb.get("prefix_weight")
    if wa == wb:
        raise SystemExit(f"both arms have prefix_weight={wa}; this is not an ablation")
    return wa, wb


def paired(deltas):
    d = np.asarray(deltas, float)
    return (float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
            int((d > 0).sum()), int((d < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=Path, required=True, help="arm trained with --prefix-weight 0")
    ap.add_argument("--prefix", type=Path, required=True, help="arm trained with --prefix-weight 0.3")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "results/ablation_prefix_supervision.md")
    args = ap.parse_args()

    a, b = load_arm(args.endpoint), load_arm(args.prefix)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no (dataset, subject, seed) cells in common")
    w_end, w_pre = _matched(a, b, shared)

    by_corpus = defaultdict(list)
    for k in shared:
        by_corpus[k[0]].append(k)

    lines = ["# Prefix supervision: deep supervision of the intermediate decisions", "",
             f"`--prefix-weight {w_end}` (endpoint-only) vs `--prefix-weight {w_pre}`. One predeclared "
             "weight, no sweep. Same model, seeds, subjects, epochs and protocol; the loss is the only "
             "difference, checked by `_matched()`. Paired per (subject, seed); se is of the paired delta.", "",
             "Accuracy at each deadline is the SAME trained model read at that deadline -- no retraining, "
             "no separate models. The endpoint row is the full-trial accuracy reported in the "
             "within-subject table.", ""]
    detail = {}
    for ds, keys in sorted(by_corpus.items()):
        pool_ms = 1000 * SPECS[ds]["pool"] / SPECS[ds]["fs"]
        n_tokens = len(a[keys[0]][2])
        lines += [f"## {ds}  (n={len(keys)} cells, {pool_ms:.3f} ms grid, {n_tokens} tokens)", "",
                  "| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |",
                  "|---|---:|---:|---:|---:|---:|"]
        rows = {}
        for d in DEADLINES[ds]:
            i = deadline_index(d, pool_ms, n_tokens)
            last = i == n_tokens - 1
            x = [100 * a[k][2][i] for k in keys]
            y = [100 * b[k][2][i] for k in keys]
            mean, se, wins, losses = paired([q - p for p, q in zip(x, y)])
            label = f"{d:g} s" + (" (endpoint)" if last else "")
            lines.append(f"| {label} | {np.mean(x):.2f} | {np.mean(y):.2f} | "
                         f"{mean:+.2f} | {se:.2f} | {wins}/{losses} |")
            rows[f"{d:g}"] = {"token_index": i, "token_ends_ms": pool_ms * (i + 1),
                              "is_endpoint": last, "endpoint_only": float(np.mean(x)),
                              "prefix_supervised": float(np.mean(y)), "paired_delta": mean,
                              "se": se, "wins": wins, "losses": losses}
        lines.append("")
        detail[ds] = {"n_cells": len(keys), "pool_ms": pool_ms, "deadlines": rows}

    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"endpoint_arm_prefix_weight": w_end, "prefix_arm_prefix_weight": w_pre,
         "corpora": detail}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
