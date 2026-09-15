#!/usr/bin/env python3
"""Where is our 2b deficit? Static band power (what the compact/reader carrier computes) versus
TIME-RESOLVED spectral structure (what BiTE's STFT branch computes), model-free, same ridge protocol.

Our reader trails BiTE by -0.92 pp on 2b and beats it on the other three corpora; BiTE's advantage
over our compact parent is +2.02 pp on 2b, its largest. 2b has 3 bipolar channels, so almost no
spatial information is available and the decoding must come from time-frequency structure. This asks
whether the information BiTE's STFT branch can see is present at the INPUT and missing from ours.

  bandpow      log band power over the whole trial          (our carrier's statistic)
  stft         log |STFT| in the same bands, 8 time windows (BiTE's branch, time-resolved)
  both         concatenation
  stft_res     stft after bandpow is ridge-regressed out of it (fit on train only)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS, raw_roles  # noqa: E402
from reader.diagnostics import filterbank_logpower, ridge_probes  # noqa: E402

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]


def windowed_logpower(x, fs, bands, windows=8):
    """Log band power in `windows` consecutive time windows: the time-resolved spectrogram."""
    edges = np.linspace(0, x.shape[-1], windows + 1).astype(int)
    return np.concatenate([filterbank_logpower(x[..., a:b], fs, bands) for a, b in zip(edges[:-1], edges[1:])], 1)


def residualize(ftr, fte, gtr, gte):
    mu, sd = gtr.mean(0, keepdims=True), gtr.std(0, keepdims=True) + 1e-8
    r = Ridge(alpha=10.0).fit((gtr - mu) / sd, ftr)
    return ftr - r.predict((gtr - mu) / sd), fte - r.predict((gte - mu) / sd)


def main():
    out = {}
    for ds in sys.argv[1:] or ["2b", "2a"]:
        spec = SPECS[ds]
        rows = []
        for sub in range(1, spec["subjects"] + 1):
            xtr, ytr, xte, yte = raw_roles(ds, sub)
            btr, bte = filterbank_logpower(xtr, spec["fs"], BANDS), filterbank_logpower(xte, spec["fs"], BANDS)
            str_, ste = windowed_logpower(xtr, spec["fs"], BANDS), windowed_logpower(xte, spec["fs"], BANDS)
            rtr, rte = residualize(str_, ste, btr, bte)
            row = {"subject": sub, "chance": 1.0 / len(np.unique(ytr)),
                   "bandpow": ridge_probes(btr, ytr, bte, yte),
                   "stft": ridge_probes(str_, ytr, ste, yte),
                   "both": ridge_probes(np.hstack([btr, str_]), ytr, np.hstack([bte, ste]), yte),
                   "stft_res": ridge_probes(rtr, ytr, rte, yte)}
            rows.append(row)
            print(ds, sub, {k: round(row[k]["probe_train_to_test"], 3)
                            for k in ("bandpow", "stft", "both", "stft_res")}, flush=True)
        m = {k: float(np.mean([r[k]["probe_train_to_test"] for r in rows]))
             for k in ("bandpow", "stft", "both", "stft_res")}
        out[ds] = {"chance": rows[0]["chance"], "means": m,
                   "per_subject": {k: [r[k]["probe_train_to_test"] for r in rows]
                                   for k in ("bandpow", "stft", "both", "stft_res")}}
        print(f"== {ds}: chance {out[ds]['chance']:.3f} | bandpow {m['bandpow']:.4f} stft {m['stft']:.4f} "
              f"(stft-bandpow {m['stft']-m['bandpow']:+.4f}) both {m['both']:.4f} "
              f"(both-bandpow {m['both']-m['bandpow']:+.4f}) stft_res {m['stft_res']:.4f}", flush=True)
    Path(__file__).with_name("spectral_headroom_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
