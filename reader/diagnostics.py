"""Per-layer / per-epoch diagnostic instrument.

Everything here is observational: it never changes parameters, buffers, modes or the
training RNG streams (probe passes run under ``torch.no_grad`` in eval mode and the
caller restores ``model.train()``; CPU/CUDA RNG states are saved and restored around
each probe so dropout order in training is unchanged).

What is recorded, and what each quantity separates
--------------------------------------------------
parameters (every tensor)
    norm, rms, epoch-mean gradient rms (pre-clip), grad/weight ratio, update ratio
    ||w_t - w_prev|| / ||w_prev|| per elapsed epoch, cosine to initialisation,
    fraction of rows sitting at a max-norm cap.            -> dead / frozen / runaway groups
modules (every leaf with an output tensor)
    mean, std, rms, positive fraction, dead units (<1% positive over the probe set),
    saturated fraction for bounded activations, per-unit train->test mean shift in
    train-std units and log std ratio.                     -> saturation, collapse, session shift
batch norms
    running statistics vs the actual train / test input statistics.  -> BN mismatch
key points (model-declared representations)
    effective rank (Roy & Vetterli), participation ratio, NC1 = tr(Sw)/tr(Sb),
    ridge probe fit on TRAIN scored on TEST, ridge probe 5-fold CV WITHIN TEST,
    class-mean direction cosine train vs test.
        train->test probe low but within-test probe high     -> shift / misalignment (overfitting to session)
        within-test probe near chance                        -> no usable signal at that depth
        train probe low / train fit < ~95%                   -> capacity bottleneck
logits
    NLL, accuracy, balanced accuracy, ECE (15 bins), confidence, over-confidence,
    Brier, logit rms, per-token accuracy curve on train and test.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU)
BOUNDED = (nn.Sigmoid, nn.Tanh, nn.Softmax)


def diag_epochs(total: int) -> set[int]:
    base = {1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100}
    base |= set(range(125, total + 1, 25))
    base.add(total)
    return {e for e in base if e <= total}


# --------------------------------------------------------------------------- statistics
def effective_rank(features: np.ndarray) -> tuple[float, float]:
    x = features - features.mean(0, keepdims=True)
    if x.shape[0] < 2 or not np.isfinite(x).all():
        return float("nan"), float("nan")
    s = np.linalg.svd(x.astype(np.float64), compute_uv=False)
    if s.sum() <= 0:
        return 0.0, 0.0
    p = s / s.sum()
    p = p[p > 0]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    ev = s ** 2
    participation = float(ev.sum() ** 2 / (ev ** 2).sum())
    return erank, participation


def nc1(features: np.ndarray, labels: np.ndarray) -> float:
    mu = features.mean(0)
    sw = sb = 0.0
    for c in np.unique(labels):
        fc = features[labels == c]
        mc = fc.mean(0)
        sw += ((fc - mc) ** 2).sum()
        sb += len(fc) * ((mc - mu) ** 2).sum()
    return float(sw / max(sb, 1e-12))


def class_direction_cosine(ftr, ytr, fte, yte) -> float:
    cos = []
    mtr, mte = ftr.mean(0), fte.mean(0)
    for c in np.intersect1d(np.unique(ytr), np.unique(yte)):
        a = ftr[ytr == c].mean(0) - mtr
        b = fte[yte == c].mean(0) - mte
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(cos)) if cos else float("nan")


def ridge_probes(ftr, ytr, fte, yte, seed=0) -> dict:
    out = {}
    scaler = StandardScaler().fit(ftr)
    clf = RidgeClassifier(alpha=10.0).fit(scaler.transform(ftr), ytr)
    out["probe_train_fit"] = float((clf.predict(scaler.transform(ftr)) == ytr).mean())
    out["probe_train_to_test"] = float((clf.predict(scaler.transform(fte)) == yte).mean())
    counts = np.bincount(yte)
    folds = int(min(5, counts[counts > 0].min())) if len(counts) else 0
    if folds >= 2:
        accs = []
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(fte, yte):
            s = StandardScaler().fit(fte[tr])
            c = RidgeClassifier(alpha=10.0).fit(s.transform(fte[tr]), yte[tr])
            accs.append((c.predict(s.transform(fte[te])) == yte[te]).mean())
        out["probe_within_test_cv"] = float(np.mean(accs))
    return out


def calibration(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> dict:
    logits = logits.float()
    prob = logits.softmax(-1)
    conf, pred = prob.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    onehot = F.one_hot(labels, logits.shape[-1]).float()
    classes = torch.unique(labels)
    balanced = torch.stack([correct[labels == c].mean() for c in classes]).mean().item()
    return {
        "nll": F.cross_entropy(logits, labels).item(),
        "acc": correct.mean().item(),
        "balanced_acc": balanced,
        "ece": ece,
        "confidence": conf.mean().item(),
        "overconfidence": conf.mean().item() - correct.mean().item(),
        "brier": ((prob - onehot) ** 2).sum(-1).mean().item(),
        "logit_rms": logits.pow(2).mean().sqrt().item(),
    }


def _reduce(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    """Reduce a representation to [B, d] without discarding its class-bearing statistic."""
    t = tensor.detach().float()
    if kind == "vector":
        return t.flatten(1)
    if kind == "channels_time":        # [B,F,T] or [B,F,1,T]: per-feature mean and log-variance over time
        t = t.flatten(1, -2) if t.ndim > 3 else t
        return torch.cat([t.mean(-1), torch.log(t.var(-1) + 1e-6)], 1)
    if kind == "tokens":               # [B,N,D]: last token and token mean
        return torch.cat([t[:, -1], t.mean(1)], 1)
    raise ValueError(kind)


# --------------------------------------------------------------------------- instrument
class Instrument:
    def __init__(self, model: nn.Module, key_points: dict[str, tuple[str, str]] | None = None,
                 max_norm_caps: dict[str, float] | None = None):
        self.model = model
        self.key_points = key_points or {}
        self.max_norm_caps = max_norm_caps or {}
        self.init = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.prev = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.prev_epoch = 0
        self.grad_sq = {}
        self.grad_batches = 0
        self.total_grad_norms = []
        self.modules = {n: m for n, m in model.named_modules()
                        if n and len(list(m.children())) == 0}

    # ------------------------------------------------------------------ training-side
    @torch.no_grad()
    def after_backward(self, clip_norm: float | None):
        """Accumulate pre-clip gradient energy on device; synchronised once per epoch."""
        total = None
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach().float().pow(2).sum()
                self.grad_sq[n] = self.grad_sq.get(n, 0.0) + g / p.numel()
                total = g if total is None else total + g
        self.grad_batches += 1
        if total is not None:
            self.total_grad_norms.append(total.sqrt())

    def _norms(self):
        if not self.total_grad_norms:
            return np.asarray([np.nan])
        return torch.stack(self.total_grad_norms).float().cpu().numpy()

    def epoch_scalars(self) -> dict:
        norms = self._norms()
        return {"grad_norm_mean": float(norms.mean()), "grad_norm_max": float(norms.max())}

    def reset_epoch(self):
        self.grad_sq = {}
        self.grad_batches = 0
        self.total_grad_norms = []

    @torch.no_grad()
    def parameter_report(self, epoch: int, clip_norm: float | None) -> dict:
        rows = {}
        elapsed = max(epoch - self.prev_epoch, 1)
        for n, p in self.model.named_parameters():
            w = p.detach().float()
            norm = w.norm().item()
            rms = w.pow(2).mean().sqrt().item()
            grad_rms = math.sqrt(float(self.grad_sq.get(n, 0.0)) / max(self.grad_batches, 1))
            prev = self.prev[n].float()
            init = self.init[n].float()
            row = {
                "norm": norm, "rms": rms, "grad_rms": grad_rms,
                "grad_to_weight": grad_rms / (rms + 1e-12),
                "update_per_epoch": (w - prev).norm().item() / (prev.norm().item() + 1e-12) / elapsed,
                "cos_to_init": float(F.cosine_similarity(w.flatten(), init.flatten(), 0, 1e-12)),
            }
            cap = self.max_norm_caps.get(n)
            if cap is not None and w.ndim >= 2:
                row_norms = w.flatten(1).norm(dim=1)
                row["frac_rows_at_cap"] = float((row_norms >= 0.999 * cap).float().mean())
            rows[n] = row
            self.prev[n] = p.detach().clone()
        self.prev_epoch = epoch
        clip_fraction = None
        if clip_norm is not None and self.total_grad_norms:
            clip_fraction = float(np.mean(self._norms() > clip_norm))
        return {"parameters": rows, "clip_fraction": clip_fraction}

    # ------------------------------------------------------------------ probe-side
    @torch.no_grad()
    def _pass(self, forward, x, batch_size, device):
        """Forward the whole split in eval mode; collect module stats and key-point features."""
        sums = defaultdict(lambda: defaultdict(float))
        unit_stats = {}
        bn_inputs = {}
        keys = {name: [] for name in self.key_points}
        by_module = defaultdict(list)
        for kp, spec in self.key_points.items():
            module_name, kind = spec[0], spec[1]
            which = spec[2] if len(spec) > 2 else "output"
            by_module[module_name].append((kp, kind, which))
        handles = []

        def out_hook(name, module):
            def hook(_m, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                if not torch.is_tensor(output) or not output.is_floating_point():
                    return
                o = output.detach().float()
                s = sums[name]
                s["n"] += o.numel()
                s["sum"] += o.sum().item()
                s["sq"] += o.pow(2).sum().item()
                s["pos"] += (o > 0).sum().item()
                if isinstance(module, BOUNDED):
                    lo, hi = (-1.0, 1.0) if isinstance(module, nn.Tanh) else (0.0, 1.0)
                    span = hi - lo
                    s["sat"] += ((o < lo + .03 * span) | (o > hi - .03 * span)).sum().item()
                if o.ndim >= 2:
                    dims = [d for d in range(o.ndim) if d != 1]
                    count = o.numel() / o.shape[1]
                    ukey = name if unit_stats.get(name, {}).get("width", o.shape[1]) == o.shape[1] else f"{name}@{o.shape[1]}"
                    u = unit_stats.setdefault(ukey, {"n": 0.0, "sum": 0.0, "sq": 0.0, "pos": 0.0, "width": o.shape[1]})
                    u["n"] += count
                    u["sum"] = u["sum"] + o.sum(dims)
                    u["sq"] = u["sq"] + o.pow(2).sum(dims)
                    u["pos"] = u["pos"] + (o > 0).float().sum(dims)
                for kp, kind, which in by_module.get(name, []):
                    source = inputs[0] if which == "input" else output
                    keys[kp].append(_reduce(source, kind).cpu())
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    i = inputs[0].detach().float()
                    dims = [d for d in range(i.ndim) if d != 1]
                    b = bn_inputs.setdefault(name, {"n": 0.0, "sum": 0.0, "sq": 0.0, "width": i.shape[1]})
                    if b["width"] != i.shape[1]:
                        return
                    c = i.numel() / i.shape[1]
                    b["n"] += c
                    b["sum"] = b["sum"] + i.sum(dims)
                    b["sq"] = b["sq"] + i.pow(2).sum(dims)
            return hook

        for name, module in self.modules.items():
            handles.append(module.register_forward_hook(out_hook(name, module)))
        logits, sequences = [], []
        try:
            tensors = x if isinstance(x, (tuple, list)) else (x,)
            for start in range(0, len(tensors[0]), batch_size):
                out = forward(*[t[start:start + batch_size].to(device) for t in tensors])
                logits.append(out["logits"].detach().float().cpu())
                if out.get("sequence") is not None:
                    sequences.append(out["sequence"].detach().float().cpu())
        finally:
            for h in handles:
                h.remove()
        return (sums, unit_stats, bn_inputs,
                {k: torch.cat(v).numpy() for k, v in keys.items() if v},
                torch.cat(logits), torch.cat(sequences) if sequences else None)

    @torch.no_grad()
    def probe(self, forward, train, test, device, batch_size=64) -> dict:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = self.model.training
        self.model.eval()
        try:
            tr = self._pass(forward, train[0], batch_size, device)
            te = self._pass(forward, test[0], batch_size, device)
        finally:
            self.model.train(was_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        ytr, yte = train[1], test[1]
        report = {"logits": {"train": calibration(tr[4], ytr), "test": calibration(te[4], yte)}}
        if tr[5] is not None:
            report["token_accuracy"] = {
                "train": (tr[5].argmax(-1) == ytr[:, None]).float().mean(0).tolist(),
                "test": (te[5].argmax(-1) == yte[:, None]).float().mean(0).tolist(),
            }
        modules = {}
        for name, s in tr[0].items():
            t = te[0].get(name)
            if not s["n"]:
                continue
            mean = s["sum"] / s["n"]
            row = {"mean": mean, "std": math.sqrt(max(s["sq"] / s["n"] - mean ** 2, 0.0)),
                   "rms": math.sqrt(s["sq"] / s["n"]), "pos_frac": s["pos"] / s["n"]}
            if "sat" in s:
                row["saturated_frac"] = s["sat"] / s["n"]
            if t and t["n"]:
                tmean = t["sum"] / t["n"]
                row["test_rms"] = math.sqrt(t["sq"] / t["n"])
                row["test_std"] = math.sqrt(max(t["sq"] / t["n"] - tmean ** 2, 0.0))
            ut, ue = tr[1].get(name), te[1].get(name)
            if ut is not None and ue is not None and ut["width"] != ue["width"]:
                ue = None
            if ut is not None:
                mu = ut["sum"] / ut["n"]
                sd = (ut["sq"] / ut["n"] - mu ** 2).clamp_min(0).sqrt()
                pos = ut["pos"] / ut["n"]
                row["units"] = int(mu.numel())
                if isinstance(self.modules[name], ACTIVATIONS):
                    row["dead_unit_frac"] = float((pos < .01).float().mean())
                    row["always_on_unit_frac"] = float((pos > .99).float().mean())
                row["zero_var_unit_frac"] = float((sd < 1e-6).float().mean())
                if ue is not None:
                    mue = ue["sum"] / ue["n"]
                    sde = (ue["sq"] / ue["n"] - mue ** 2).clamp_min(0).sqrt()
                    row["unit_mean_shift"] = float(((mue - mu).abs() / (sd + 1e-6)).mean())
                    row["unit_log_std_ratio"] = float(torch.log((sde + 1e-6) / (sd + 1e-6)).mean())
            modules[name] = row
        report["modules"] = modules
        bns = {}
        for name, b in tr[2].items():
            m = self.modules[name]
            if m.running_mean is None:
                continue
            rm, rv = m.running_mean.float().cpu(), m.running_var.float().cpu()
            row = {}
            for split, stats in (("train", b), ("test", te[2].get(name))):
                if stats is None:
                    continue
                mu = stats["sum"].cpu() / stats["n"]
                var = (stats["sq"].cpu() / stats["n"] - mu ** 2).clamp_min(0)
                row[f"{split}_mean_shift"] = float(((mu - rm).abs() / (rv.sqrt() + 1e-5)).mean())
                row[f"{split}_log_var_ratio"] = float(torch.log((var + 1e-5) / (rv + 1e-5)).mean())
            bns[name] = row
        report["batchnorm"] = bns
        keyrep = {}
        ytr_np, yte_np = ytr.numpy(), yte.numpy()
        for kp in self.key_points:
            if kp not in tr[3] or kp not in te[3]:
                continue
            ftr, fte = tr[3][kp], te[3][kp]
            if not (np.isfinite(ftr).all() and np.isfinite(fte).all()):
                keyrep[kp] = {"nonfinite": True}
                continue
            er_tr, pr_tr = effective_rank(ftr)
            er_te, pr_te = effective_rank(fte)
            row = {"dim": int(ftr.shape[1]), "erank_train": er_tr, "erank_test": er_te,
                   "participation_train": pr_tr, "participation_test": pr_te,
                   "nc1_train": nc1(ftr, ytr_np), "nc1_test": nc1(fte, yte_np),
                   "class_direction_cos": class_direction_cosine(ftr, ytr_np, fte, yte_np)}
            row.update(ridge_probes(ftr, ytr_np, fte, yte_np))
            keyrep[kp] = row
        report["key_points"] = keyrep
        return report


# --------------------------------------------------------------------------- model-free baseline
def filterbank_logpower(x: np.ndarray, fs: float, bands) -> np.ndarray:
    """Model-free information baseline: log band power per channel and band."""
    from scipy.signal import butter, sosfiltfilt
    feats = []
    for lo, hi in bands:
        sos = butter(4, [lo, min(hi, fs / 2 - 1)], btype="band", fs=fs, output="sos")
        feats.append(np.log(sosfiltfilt(sos, x, axis=-1).var(-1) + 1e-8))
    return np.concatenate(feats, 1)


def complex_spectrum(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x, axis=-1) / x.shape[-1]
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    keep = (freqs >= lo) & (freqs <= hi)
    s = spec[..., keep]
    return np.concatenate([s.real, s.imag], -1).reshape(len(x), -1)


def input_information(dataset, train_x, train_y, test_x, test_y, fs) -> dict:
    out = {}
    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
    ftr = filterbank_logpower(train_x, fs, bands)
    fte = filterbank_logpower(test_x, fs, bands)
    out["logpower"] = ridge_probes(ftr, train_y, fte, test_y)
    if dataset == "sdssvep":
        ctr = complex_spectrum(train_x, fs, 8, 64)
        cte = complex_spectrum(test_x, fs, 8, 64)
        out["complex_spectrum"] = ridge_probes(ctr, train_y, cte, test_y)
    # Share of variance on the flat (all-ones) spatial direction, projected onto the UNIT vector.
    c = train_x.shape[1]
    u = np.ones(c) / math.sqrt(c)
    for split, x in (("train", train_x), ("test", test_x)):
        proj = np.einsum("c,nct->nt", u, x)
        out[f"flat_mode_share_{split}"] = float((proj ** 2).sum() / (x ** 2).sum())
    # Line-noise share (>45 Hz) of total power, the S5 artifact signature.
    if fs >= 200:
        spec = np.abs(np.fft.rfft(test_x, axis=-1)) ** 2
        freqs = np.fft.rfftfreq(test_x.shape[-1], 1 / fs)
        out["test_power_share_45_55hz"] = float(spec[..., (freqs > 45) & (freqs < 55)].sum() / spec.sum())
        spec = np.abs(np.fft.rfft(train_x, axis=-1)) ** 2
        out["train_power_share_45_55hz"] = float(spec[..., (freqs > 45) & (freqs < 55)].sum() / spec.sum())
    return out
