#!/usr/bin/env python3
"""Model-free: which reference-quotient operator makes class information TRANSFER train->test?

For every subject: raw input, CAR, and geometry-local quotients x - R x with R a row-stochastic
Gaussian neighbourhood of width sigma (sigma -> inf is CAR). Features: log band power (MI) or
complex spectrum (SD-SSVEP). Ridge fit on the training role, scored on the test role, plus the
within-test CV ceiling. Also the variance share left in the smoothest spatial modes after CAR.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS, electrode_positions, raw_roles  # noqa: E402
from reader.diagnostics import complex_spectrum, filterbank_logpower, ridge_probes  # noqa: E402

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]


def quotient_matrix(pos, sigma):
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    if sigma is None:
        r = np.ones_like(d2)
    else:
        r = np.exp(-d2 / (2 * sigma ** 2))
        np.fill_diagonal(r, 0.0)   # local average of the OTHER electrodes (Hjorth-like)
    return np.eye(len(pos)) - r / r.sum(1, keepdims=True)


def standardize(train, test):
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def main():
    out = {}
    for ds in sys.argv[1:] or ["2a", "hgd", "sdssvep"]:
        spec = SPECS[ds]
        pos = electrode_positions(ds).numpy().astype(np.float64)
        ops = {"raw": np.eye(len(pos)), "car": quotient_matrix(pos, None)}
        for s in (0.15, 0.25, 0.4, 0.7):
            ops[f"local_{s}"] = quotient_matrix(pos, s)
        rows = {k: [] for k in ops}
        smooth = []
        for sub in range(1, spec["subjects"] + 1):
            xtr, ytr, xte, yte = raw_roles(ds, sub)
            for name, q in ops.items():
                atr = np.einsum("dc,nct->ndt", q, xtr)
                ate = np.einsum("dc,nct->ndt", q, xte)
                if ds == "sdssvep":
                    ftr, fte = complex_spectrum(atr, spec["fs"], 8, 64), complex_spectrum(ate, spec["fs"], 8, 64)
                else:
                    ftr, fte = filterbank_logpower(atr, spec["fs"], BANDS), filterbank_logpower(ate, spec["fs"], BANDS)
                rows[name].append(ridge_probes(ftr, ytr, fte, yte))
            # variance share of the 3 smoothest (lowest graph-Laplacian eigen) non-flat spatial modes after CAR
            d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
            w = np.exp(-d2 / (2 * 0.25 ** 2)); np.fill_diagonal(w, 0)
            lap = np.diag(w.sum(1)) - w
            vals, vecs = np.linalg.eigh(lap)
            car = np.einsum("dc,nct->ndt", ops["car"], xtr)
            total = (car ** 2).sum()
            smooth.append(float(sum(((np.einsum("c,nct->nt", vecs[:, i], car)) ** 2).sum() for i in (1, 2, 3)) / total))
            print(ds, sub, {k: round(v[-1]["probe_train_to_test"], 3) for k, v in rows.items()}, flush=True)
        out[ds] = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
        out[ds]["smooth3_share_after_car"] = float(np.mean(smooth))
        out[ds]["per_subject_train_to_test"] = {k: [r["probe_train_to_test"] for r in v] for k, v in rows.items()}
    path = Path(__file__).with_name("quotient_operator_probe.json")
    path.write_text(json.dumps(out, indent=1))
    for ds, r in out.items():
        print(ds, {k: (round(v["probe_train_to_test"], 4), round(v["probe_within_test_cv"], 4))
                   for k, v in r.items() if isinstance(v, dict) and "probe_train_to_test" in v},
              "smooth3", round(r["smooth3_share_after_car"], 3))


if __name__ == "__main__":
    main()
