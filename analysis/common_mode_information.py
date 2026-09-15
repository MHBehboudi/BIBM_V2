#!/usr/bin/env python3
"""Model-free: how much CLASS information does the common (all-ones) mode carry, per corpus?

Motivates the decomposition arm: CAR discards the common mode entirely. If the common mode is
class-informative (expected on SSVEP, where all occipital electrodes see the flicker in phase),
discarding it must cost accuracy; if it is nuisance (expected on motor imagery), removing it helps.
Features: log band power (MI) / complex spectrum (SSVEP). Ridge fit on train, scored on test.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS, raw_roles           # noqa: E402
from reader.diagnostics import complex_spectrum, filterbank_logpower, ridge_probes  # noqa: E402

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]


def feats(ds, x, fs):
    if ds == "sdssvep":
        return complex_spectrum(x, fs, 8, 64)
    return filterbank_logpower(x, fs, BANDS)


def main():
    out = {}
    for ds in sys.argv[1:] or ["2a", "2b", "hgd", "sdssvep"]:
        spec = SPECS[ds]
        rows = {"full": [], "field_only": [], "common_only": [], "field_plus_common": []}
        for sub in range(1, spec["subjects"] + 1):
            xtr, ytr, xte, yte = raw_roles(ds, sub)
            parts = {}
            for name, xs in (("train", xtr), ("test", xte)):
                m = xs.mean(1, keepdims=True)
                parts[name] = {"full": xs, "field_only": xs - m, "common_only": m,
                               "field_plus_common": np.concatenate([xs - m, m], 1)}
            for k in rows:
                rows[k].append(ridge_probes(feats(ds, parts["train"][k], spec["fs"]), ytr,
                                            feats(ds, parts["test"][k], spec["fs"]), yte))
            print(ds, sub, {k: round(v[-1]["probe_train_to_test"], 3) for k, v in rows.items()}, flush=True)
        out[ds] = {k: {"mean_train_to_test": float(np.mean([r["probe_train_to_test"] for r in v])),
                       "per_subject": [round(r["probe_train_to_test"], 4) for r in v]} for k, v in rows.items()}
        out[ds]["chance"] = 1.0 / spec["classes"]
    Path(__file__).with_name("common_mode_information.json").write_text(json.dumps(out, indent=1))
    print()
    for ds, r in out.items():
        print(ds, "chance", round(r["chance"], 3),
              {k: round(v["mean_train_to_test"], 4) for k, v in r.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
