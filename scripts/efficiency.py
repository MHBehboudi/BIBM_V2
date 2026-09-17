#!/usr/bin/env python3
"""Parameters and CPU inference latency: what one READER costs against one full-trial model and against the
bank of per-deadline specialists a non-anytime model needs.

Measured, not estimated. Single decision, batch 1, ONE CPU thread, float32, eval mode, median of `--reps`
timed calls after `--warmup` untimed calls. BiTE's latency includes computing its STFT input from the raw
trial (it cannot decide without it). READER's latency is reported for a single decision at each deadline
(its reversed branch reads the observed prefix, O(t) tokens) and for the full anytime curve (every 128 ms
decision of a trial recomputed from scratch, O(T^2)); a streaming deployment computes one decision per new
token, whose cost is the single-decision column at that deadline.

Writes results/EFFICIENCY.md and results/efficiency.json.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS  # noqa: E402
from reader.models import registry  # noqa: E402
from reader.models.zoo import MODELS as ZOO  # noqa: E402

DEADLINES = {"2a": [1.0, 2.0, 3.0, 4.0], "2b": [1.0, 2.0, 3.0, 4.0], "hgd": [1.0, 2.0, 3.0, 4.0],
             "sdssvep": [0.25, 0.5, 0.75, 1.0]}


def timed(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        t.append(1000 * (time.perf_counter() - start))
    return statistics.median(t)


def reader_decision(model, x):
    """ONE READER decision on the observed input x: forward TCN over the prefix, ONE reversed pass over it.

    `ReaderDecoder.forward` also returns the decision at every earlier token (O(T^2)), which a deployed
    single decision does not need. This computes only the last one; `main` asserts it equals
    `model(x)["logits"]` before timing it.
    """
    tokens = model.tokens(model.spatial_features(model.carriers(x)))
    positioned = tokens + model.pos[:, :tokens.shape[1]]
    forward_end = model.tcn(positioned)[:, -1]
    backward_end = model.backward_tcn(positioned.flip(1))[:, -1]
    g = model._gate()
    return model._readout((1 - g) * forward_end + g * backward_end)


def build(name, ds, samples=None):
    options = {"samples": samples} if samples else {}
    model, _, _, needs_spectral = registry.build(name, ds, **options)
    return model.eval(), needs_spectral


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--corpora", default="2a,2b,hgd,sdssvep")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--note", default="", help="caveat printed under the header (e.g. measurement conditions)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
               platform.processor())
    report = {"cpu": cpu, "threads": 1, "torch": torch.__version__, "reps": args.reps, "corpora": {}}
    lines = ["# Efficiency: parameters and CPU latency per decision", "",
             f"CPU: {cpu}, 1 thread, float32, batch 1, median of {args.reps} calls. Latency is milliseconds of "
             "compute per decision, not including acquisition. BiTE includes its STFT.", ""]
    if args.note:
        lines += [args.note, ""]

    for ds in args.corpora.split(","):
        spec = SPECS[ds]
        n_tokens = -(-spec["samples"] // spec["pool"])
        x_full = torch.randn(1, spec["channels"], spec["samples"])
        entry = {}

        def latency(name, samples):
            model, needs_spectral = build(name, ds, samples if name in ("bite",) or name.startswith("zoo_") else None)
            x = x_full[..., :samples]
            if needs_spectral:
                from reader.models.baseline import stft_exact
                fn = lambda: model(x, stft_exact(x.numpy(), ds, torch.device("cpu")))  # noqa: E731
            elif name == "reader":
                model.classify(model.tokens(model.spatial_features(model.carriers(x))))   # apply the head max-norm once
                ref = model(x)["logits"]
                assert torch.allclose(reader_decision(model, x), ref, atol=1e-5), "single-decision path != forward"
                fn = lambda: reader_decision(model, x)  # noqa: E731
            else:
                fn = lambda: model(x)  # noqa: E731
            return model, timed(fn, args.reps, args.warmup)

        rows = []
        reader, reader_full_ms = latency("reader", spec["samples"])
        compact, compact_ms = latency("compact", spec["samples"])
        bite, bite_ms = latency("bite", spec["samples"])
        params = {m: int(sum(p.numel() for p in mod.parameters()))
                  for m, mod in (("reader", reader), ("compact", compact), ("bite", bite))}
        curve_ms = timed(lambda: reader.anytime_logits(x_full), max(5, args.reps // 5), 2)
        per_deadline = {}
        for d in DEADLINES[ds]:
            samples = int(round(d * spec["fs"]))
            _, r_ms = latency("reader", samples)
            b_model, b_ms = latency("bite", samples)
            per_deadline[d] = {"reader_single_decision_ms": r_ms, "bite_specialist_ms": b_ms,
                               "bite_specialist_parameters": int(sum(p.numel() for p in b_model.parameters()))}
        entry.update({"parameters": params, "full_trial_ms": {"reader": reader_full_ms, "compact": compact_ms,
                                                              "bite": bite_ms},
                      "reader_full_anytime_curve_ms": curve_ms, "tokens": n_tokens, "per_deadline": per_deadline})
        zoo = {}
        for m in ZOO:
            try:
                mod, ms = latency(f"zoo_{m}", spec["samples"])
                zoo[m] = {"parameters": int(sum(p.numel() for p in mod.parameters())), "full_trial_ms": ms}
            except Exception as e:                      # FACTNet cannot run on SD-SSVEP as released
                zoo[m] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        entry["zoo"] = zoo
        report["corpora"][ds] = entry

        k = len(DEADLINES[ds])
        lines += [f"## {ds} ({spec['channels']} channels, {spec['samples']} samples, {n_tokens} tokens of "
                  f"{1000 * spec['pool'] / spec['fs']:g} ms)", "",
                  "| model | parameters | full-trial decision (ms) |", "|---|---:|---:|"]
        for m, label in (("reader", "**READER**"), ("compact", "Compact"), ("bite", "BiTE")):
            lines.append(f"| {label} | {params[m]:,} | {entry['full_trial_ms'][m]:.2f} |")
        for m in sorted(zoo, key=lambda m: zoo[m].get("parameters", 0)):
            z = zoo[m]
            lines.append(f"| {m} | {z['parameters']:,} | {z['full_trial_ms']:.2f} |" if "parameters" in z
                         else f"| {m} | cannot run as released | — |")
        lines += ["", f"Anytime deployment with decisions at {', '.join(f'{d:g} s' for d in DEADLINES[ds])}:", "",
                  "| deadline | READER single decision (ms) | BiTE specialist at that deadline (ms) |",
                  "|---|---:|---:|"]
        for d, v in per_deadline.items():
            lines.append(f"| {d:g} s | {v['reader_single_decision_ms']:.2f} | {v['bite_specialist_ms']:.2f} |")
        bank_params = sum(v["bite_specialist_parameters"] for v in per_deadline.values())
        entry["bite_bank_parameters"] = bank_params
        lines += ["", f"- Models to store: READER **1** ({params['reader']:,} parameters) vs a BiTE bank of **{k}** "
                  f"specialists ({bank_params:,} parameters in total).",
                  "- Timings are single-thread medians on a shared compute node; small steps between deadlines "
                  "(e.g. a jump at the same length for both models) come from the convolution backend, not the models.",
                  f"- READER's full anytime curve (all {n_tokens} decisions recomputed from scratch): "
                  f"{curve_ms:.1f} ms; a streaming system computes only the newest decision.", ""]
        print("\n".join(lines[-25:]))

    (args.out / "EFFICIENCY.md").write_text("\n".join(lines) + "\n")
    (args.out / "efficiency.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
