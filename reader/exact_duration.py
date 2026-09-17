#!/usr/bin/env python3
"""Exact-duration, same-checkpoint evaluation (co-author priority 1, 2026-09-16). No training.

For every trained checkpoint, the model is given ONLY the first n samples of each test trial and its
ENDPOINT prediction on that truncated input is scored. Nothing is read off a full-trial curve: at
n = 250 on a 32-sample pool the last pooling window is PARTIAL (26 samples) and the model's own
ceil_mean_pool rescales it, exactly as at deployment. The same checkpoint is used at every n.

Also, for causal arms, the co-author's strengthened causality verification ON TRAINED WEIGHTS:
  trunc_max  = max |logits(x[:u*pool]) - anytime_curve(x)[u-1]| over every pooling boundary u, all trials
  future_max = max |anytime_curve(x')[:u] - anytime_curve(x)[:u]| where x' replaces every sample after
               u*pool by N(0, 1e3^2) noise, at u in {1, N/4, N/2, 3N/4}
READER additionally reports the gate interventions g in {0, 0.5, 1} at every n.

BiTE (released code) is evaluated on the same truncated inputs with its STFT computed from the
truncated trial only. Documented adaptation: none to the architecture; its AvgPool FLOORS, so the
partial final window (e.g. 26 samples at n = 250) is discarded by BiTE's own design.

Normalisation: the pointwise (channel x sample) StandardScaler is fit on the full-length TRAINING
role; truncating after scaling equals scaling the truncated trial, since each sample position has
its own statistics. No test statistic is used.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS, load_roles  # noqa: E402
from reader.models import registry  # noqa: E402
GRID = {"2a": [125, 250, 375, 500, 625, 750, 875, 1000], "2b": [125, 250, 375, 500, 625, 750, 875, 1000],
        "sdssvep": [32, 64, 96, 128, 160, 192, 224, 256]}


@torch.no_grad()
def logits_of(model, tensors, bs=64):
    model.eval()
    return torch.cat([model(*[t[i:i + bs] for t in tensors])["logits"].float() for i in range(0, len(tensors[0]), bs)])


@torch.no_grad()
def curve_of(model, x, bs=64):
    model.eval()
    return torch.cat([model.anytime_logits(x[i:i + bs]).float() for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, help="registry arm name: reader, compact, ff_control, compact_mean, bite")
    ap.add_argument("--runs", type=Path, required=True, help="arm directory holding <ds>_S<sub>_seed<seed>/final.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/exact_duration")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--accuracy-only", action="store_true",
                    help="skip the gate interventions and the trained-weight causality checks (already established "
                         "on every endpoint-trained checkpoint); used for the loss-ablation arms")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ds, spec = args.dataset, SPECS[args.dataset]
    grid = GRID[ds]
    out_dir = args.out / ds / args.runs.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in sorted(args.runs.glob(f"{ds}_S*_seed*")):
        target = out_dir / f"{run.name}.json"
        if not (run / "final.pt").exists():
            continue
        summary = json.loads((run / "summary.json").read_text())
        if target.exists():
            # Skip only if the record was computed from THIS run. A run of record can be replaced (e.g. a MIG
            # original by its full-GPU re-run) under the same name; its GPU or logged accuracy then differs.
            old = json.loads(target.read_text())
            if old.get("device_trained") == summary.get("device") and \
                    abs(float(old.get("final_test_acc_logged", -1)) - summary["final_test"]["acc"]) < 1e-9:
                continue
            print(f"recomputing {target.name}: its run of record changed", flush=True)
        started = time.time()
        sub, seed = summary["subject"], summary["seed"]
        _, _, xte, yte = load_roles(ds, sub, "bite")
        model, _, _, needs_spectral = registry.build(args.arm, ds)
        model.load_state_dict(torch.load(run / "final.pt", map_location="cpu", weights_only=True))
        model.eval()
        x = torch.from_numpy(xte)
        y = torch.from_numpy(yte)

        def tensors(n):
            t = [x[..., :n].contiguous()]
            if needs_spectral:
                from reader.models.baseline import stft_exact
                t.append(stft_exact(xte[..., :n], ds, "cpu"))
            return t

        rec = {"dataset": ds, "subject": sub, "seed": seed, "arm": args.arm, "run": run.name,
               "device_trained": summary.get("device"), "final_test_acc_logged": summary["final_test"]["acc"],
               "grid_samples": grid, "grid_seconds": [n / spec["fs"] for n in grid], "acc": {}, "gate": {}}
        saved = {}
        for n in grid:
            lg = logits_of(model, tensors(n))
            rec["acc"][str(n)] = float((lg.argmax(1) == y).float().mean())
            saved[f"n{n}"] = lg.numpy()
            if getattr(model, "reader_mode", None) in ("bidir", "ff") and not args.accuracy_only:
                for g in (0.0, 0.5, 1.0):
                    model._intervention = {"gate": g}
                    lg2 = logits_of(model, tensors(n))
                    rec["gate"].setdefault(str(g), {})[str(n)] = float((lg2.argmax(1) == y).float().mean())
                model._intervention = {}
        full = spec["samples"]
        rec["full_length_matches_logged"] = abs(rec["acc"].get(str(full), -1) - summary["final_test"]["acc"]) < 1e-9

        if hasattr(model, "anytime_logits") and not args.accuracy_only:
            pool = spec["pool"]
            curve = curve_of(model, x)
            n_tokens = curve.shape[1]
            trunc = 0.0
            boundaries = [u for u in range(1, n_tokens + 1) if u * pool <= full]
            for u in boundaries:
                trunc = max(trunc, (logits_of(model, [x[..., :u * pool].contiguous()]) - curve[:, u - 1]).abs().max().item())
            g = torch.Generator().manual_seed(seed)
            future, tested = 0.0, []
            for u in sorted({1, n_tokens // 4, n_tokens // 2, 3 * n_tokens // 4}):
                if u < 1 or u * pool >= full:
                    continue
                x2 = x.clone()
                x2[..., u * pool:] = 1e3 * torch.randn(x2[..., u * pool:].shape, generator=g)
                future = max(future, (curve_of(model, x2)[:, :u] - curve[:, :u]).abs().max().item())
                tested.append(u)
            rec["causality"] = {"trunc_max_abs_logit_diff": trunc, "boundaries_tested": len(boundaries),
                                "future_max_abs_logit_diff": future, "future_boundaries": tested,
                                "trials": int(len(y)), "bn_track_running_stats": all(
                                    m.track_running_stats for m in model.modules()
                                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))}
        rec["seconds"] = time.time() - started
        np.savez_compressed(out_dir / f"{run.name}.npz", labels=yte, **saved)
        target.write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ("subject", "seed", "acc", "full_length_matches_logged")}),
              rec.get("causality"), f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
