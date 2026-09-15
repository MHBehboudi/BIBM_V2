#!/usr/bin/env python3
"""Instrumented trainer (official roles, fixed final epoch).

Recipe (compact-family contract, applied identically to every arm):
Adam lr 2e-3 wd 2e-3, cosine to zero over the horizon, label smoothing .1, batch 64,
gradient clip 5, float32, 600 epochs, no model selection. The reported number is
the final epoch. The official test role is evaluated every epoch for the
train-vs-test DIAGNOSTIC trajectory only; nothing is ever selected on it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reader.data import CHANCE, SPECS, load_roles, loso_roles, seed_everything  # noqa: E402
from reader.diagnostics import Instrument, calibration, diag_epochs, input_information  # noqa: E402
from reader.models import registry  # noqa: E402

TELEMETRY_BUDGET = 10 * 1024 * 1024


def task_loss(out, y, prefix_weight, smoothing=.1):
    loss = F.cross_entropy(out["logits"], y, label_smoothing=smoothing)
    parts = {"endpoint_ce": loss.detach()}
    seq = out.get("sequence")
    if prefix_weight and seq is not None and seq.shape[1] > 1:
        early = seq[:, :-1]
        prefix = F.cross_entropy(early.reshape(-1, early.shape[-1]),
                                 y[:, None].expand(-1, early.shape[1]).reshape(-1),
                                 label_smoothing=smoothing)
        loss = loss + prefix_weight * prefix
        parts["prefix_ce"] = prefix.detach()
    aux = out.get("aux_loss")
    if aux is not None:
        loss = loss + aux
        parts["aux_loss"] = aux.detach()
    return loss, parts


@torch.no_grad()
def evaluate(model, tensors, y, device, batch_size=64):
    model.eval()
    logits = []
    for start in range(0, len(y), batch_size):
        out = model(*[t[start:start + batch_size].to(device) for t in tensors])
        logits.append(out["logits"].float().cpu())
    model.train()
    return torch.cat(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=tuple(SPECS), required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--prefix-weight", type=float, default=0.0)
    ap.add_argument("--input-mode", choices=("bite", "quotient"), default=None)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--option", action="append", default=[], help="key=value model option")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--protocol", choices=("within", "loso"), default="within",
                    help="within: official train/test session roles of one subject. "
                         "loso: BiTE cross_subject — every other subject pooled as the training "
                         "role, --subject held out entirely as the test role.")
    ap.add_argument("--no-ea", action="store_true",
                    help="loso only: drop the Euclidean-alignment whitening BiTE enables for "
                         "cross_subject (is_EA: true). Control arm, not the protocol-matched one.")
    ap.add_argument("--window-seconds", type=float, default=None,
                    help="DEADLINE-SPECIALIST MODE: truncate both roles to the first N seconds and "
                         "train from scratch there, exactly as a per-deadline baseline would be built.")
    args = ap.parse_args()

    options = {}
    for item in args.option:
        k, v = item.split("=", 1)
        try:
            options[k] = json.loads(v)
        except json.JSONDecodeError:
            options[k] = v
    spec = SPECS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        print("complete, skipping", run_dir)
        return

    window_samples = None
    if args.window_seconds is not None:
        window_samples = int(round(args.window_seconds * spec["fs"]))
        if not 1 <= window_samples <= spec["samples"]:
            raise ValueError(f"window {args.window_seconds}s is outside the trial")
        options["samples"] = window_samples

    seed_everything(args.seed)
    model, key_points, caps, needs_spectral = registry.build(args.model, args.dataset, **options)
    input_mode = args.input_mode or getattr(model, "input_mode", "bite")
    model = model.to(device)
    parameters = int(sum(p.numel() for p in model.parameters()))

    if args.protocol == "loso":
        train_x, train_y, test_x, test_y = loso_roles(
            args.dataset, args.subject, input_mode, euclidean_alignment=not args.no_ea)
    else:
        train_x, train_y, test_x, test_y = load_roles(args.dataset, args.subject, input_mode)
    if window_samples is not None:
        # Truncate the raw trial first, then everything downstream (scaler, STFT) sees only
        # what a system with this deadline would ever have seen.
        train_x, test_x = train_x[..., :window_samples], test_x[..., :window_samples]
    info = input_information(args.dataset, train_x, train_y, test_x, test_y, spec["fs"])
    tr_tensors = [torch.from_numpy(train_x)]
    te_tensors = [torch.from_numpy(test_x)]
    if needs_spectral:
        from reader.models.baseline import stft_exact
        tr_tensors.append(stft_exact(train_x, args.dataset, device))
        te_tensors.append(stft_exact(test_x, args.dataset, device))
    ytr, yte = torch.from_numpy(train_y), torch.from_numpy(test_y)
    if hasattr(model, "fit_statistics"):
        model.fit_statistics(tr_tensors[0].to(device))

    training_seed = args.seed + 9000
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    order = torch.Generator().manual_seed(training_seed)
    decay = [p for p in model.parameters() if not getattr(p, "_no_weight_decay", False)]
    no_decay = [p for p in model.parameters() if getattr(p, "_no_weight_decay", False)]
    groups = [{"params": decay, "weight_decay": 2e-3}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = torch.optim.Adam(groups, lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    instrument = Instrument(model, key_points, caps)
    diag_at = set() if args.no_diag else diag_epochs(args.epochs)
    curves, diag_rows = [], []
    diag_path = run_dir / "diag.jsonl.gz"
    diag_file = gzip.open(diag_path, "wt")

    def forward_eval(*tensors):
        return model(*tensors)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            instrument.reset_epoch()
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch, args.epochs)
            perm = torch.randperm(len(ytr), generator=order)
            loss_sum, parts_sum, correct, seen = 0.0, {}, 0, 0
            for start in range(0, len(perm), 64):
                idx = perm[start:start + 64]
                if len(idx) < 2:
                    continue
                batch = [t[idx].to(device, non_blocking=True) for t in tr_tensors]
                y = ytr[idx].to(device)
                out = model(*batch)
                loss, parts = task_loss(out, y, args.prefix_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                instrument.after_backward(args.clip or None)
                if args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                n = len(idx)
                loss_sum += loss.detach() * n
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v * n
                correct += (out["logits"].detach().argmax(1) == y).sum()
                seen += n
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            tr_logits = evaluate(model, tr_tensors, ytr, device)
            te_logits = evaluate(model, te_tensors, yte, device)
            row = {"epoch": epoch, "lr": lr,
                   "train_loss_dropout": float(loss_sum) / seen,
                   "train_acc_dropout": float(correct) / seen,
                   **{k: float(v) / seen for k, v in parts_sum.items()},
                   "train": calibration(tr_logits, ytr), "test": calibration(te_logits, yte),
                   **instrument.epoch_scalars()}
            if hasattr(model, "epoch_telemetry"):
                row["model"] = model.epoch_telemetry()
            curves.append(row)
            if epoch in diag_at:
                report = {"epoch": epoch, **instrument.parameter_report(epoch, args.clip or None),
                          **instrument.probe(forward_eval, (tr_tensors, ytr), (te_tensors, yte), device)}
                if hasattr(model, "diagnostic_report"):
                    report["mechanism"] = model.diagnostic_report(tr_tensors, te_tensors, ytr, yte, device)
                diag_file.write(json.dumps(report) + "\n")
                diag_file.flush()
                t, e = row["train"], row["test"]
                print(f"ep {epoch:4d} lr {lr:.2e} loss {row['train_loss_dropout']:.4f} "
                      f"train {t['acc']:.4f} test {e['acc']:.4f} nll {e['nll']:.3f} "
                      f"ece {e['ece']:.3f} gn {row['grad_norm_mean']:.3f}", flush=True)
    finally:
        diag_file.close()

    final_logits = evaluate(model, te_tensors, yte, device)
    final = calibration(final_logits, yte)
    size = diag_path.stat().st_size
    torch.save(model.state_dict(), run_dir / "final.pt")
    np.savez_compressed(run_dir / "test_logits.npz", logits=final_logits.numpy(), labels=test_y)
    test_curve = [c["test"]["acc"] for c in curves]
    summary = {
        "status": "complete", "model": args.model, "options": options, "dataset": args.dataset,
        "subject": args.subject, "seed": args.seed, "epochs": args.epochs,
        "prefix_weight": args.prefix_weight, "input_mode": input_mode, "parameters": parameters,
        "window_seconds": args.window_seconds, "window_samples": window_samples,
        "protocol": ({"name": "loso", "train_role": "all other subjects, both sessions, pooled",
                      "test_role": f"subject {args.subject}, both sessions, never seen",
                      "euclidean_alignment": not args.no_ea,
                      "reported": "fixed final epoch", "test_curve_use": "diagnostic only, never selection"}
                     if args.protocol == "loso" else
                     {"name": "within", "train_role": "100% official training", "test_role": "official test",
                      "reported": "fixed final epoch", "test_curve_use": "diagnostic only, never selection"}),
        "final_test": final, "final_train": curves[-1]["train"],
        "chance": CHANCE[args.dataset],
        "test_peak_diagnostic": {"acc": float(max(test_curve)), "epoch": int(np.argmax(test_curve)) + 1},
        "input_information": info,
        "telemetry_bytes": size, "telemetry_within_budget": size <= TELEMETRY_BUDGET,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "elapsed_seconds": time.time() - started,
    }
    (run_dir / "curves.json").write_text(json.dumps(curves))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("model", "dataset", "subject", "seed", "parameters")}
                     | {"final_acc": final["acc"], "elapsed": round(summary["elapsed_seconds"])}))


if __name__ == "__main__":
    main()
