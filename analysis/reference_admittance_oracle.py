#!/usr/bin/env python3
"""Is the reference decision a PER-TRIAL decision, or a per-corpus one?

CAR is preprocessing: it applies one fixed projection x -> x - mean(x) to every trial of every
subject. Our own measurements say the right amount of common-mode removal is corpus-split (2a hard
subjects want it, SD-SSVEP's common mode IS the flicker). A per-trial reference would be a genuine
architectural mechanism -- a fixed linear spatial filter CANNOT express it, because a trial-dependent
projection is not a fixed linear map -- but only if the headroom exists.

We sweep the admittance a in x_a = x - (1-a) * mean_channels(x):  a=0 is CAR, a=1 is raw.
For each subject we fit one ridge probe per a on the training role and then score the test role
under four selection rules, each strictly stronger than the last:

  fixed_car / fixed_raw   the two endpoints preprocessing actually offers
  best_global_a           one a for the whole corpus, chosen on TEST (already optimistic)
  best_subject_a          one a per subject, chosen on TEST (what a per-subject mechanism could win)
  oracle_trial_a          a per TRIAL, chosen on TEST (the ceiling for any per-trial mechanism)

oracle_trial_a - best_subject_a is the headroom that per-trial adaptation, and nothing weaker,
could reach. If it is small the mechanism is not worth building.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS, raw_roles  # noqa: E402
from reader.diagnostics import complex_spectrum, filterbank_logpower  # noqa: E402

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
ADMIT = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    out = {}
    for ds in sys.argv[1:] or ["2a", "sdssvep"]:
        spec = SPECS[ds]
        per_subject = []
        for sub in range(1, spec["subjects"] + 1):
            xtr, ytr, xte, yte = raw_roles(ds, sub)
            mtr, mte = xtr.mean(1, keepdims=True), xte.mean(1, keepdims=True)
            correct = {}
            for a in ADMIT:
                atr, ate = xtr - (1 - a) * mtr, xte - (1 - a) * mte
                if ds == "sdssvep":
                    ftr, fte = complex_spectrum(atr, spec["fs"], 8, 64), complex_spectrum(ate, spec["fs"], 8, 64)
                else:
                    ftr, fte = filterbank_logpower(atr, spec["fs"], BANDS), filterbank_logpower(ate, spec["fs"], BANDS)
                sc = StandardScaler().fit(ftr)
                clf = RidgeClassifier(alpha=10.0).fit(sc.transform(ftr), ytr)
                correct[a] = (clf.predict(sc.transform(fte)) == yte).astype(float)
            per_subject.append({"subject": sub, "acc": {a: float(c.mean()) for a, c in correct.items()},
                                "correct": {a: c for a, c in correct.items()}})
            print(ds, sub, {a: round(float(c.mean()), 3) for a, c in correct.items()}, flush=True)

        acc = {a: float(np.mean([s["acc"][a] for s in per_subject])) for a in ADMIT}
        best_global = max(ADMIT, key=lambda a: acc[a])
        best_subject = float(np.mean([max(s["acc"].values()) for s in per_subject]))
        oracle_trial = float(np.mean([np.mean(np.max(np.stack([s["correct"][a] for a in ADMIT]), 0))
                                      for s in per_subject]))
        out[ds] = {
            "acc_by_admittance": acc, "best_global_a": best_global,
            "fixed_car": acc[0.0], "fixed_raw": acc[1.0],
            "best_global": acc[best_global], "best_subject_a": best_subject, "oracle_trial_a": oracle_trial,
            "headroom_trial_over_subject": oracle_trial - best_subject,
            "headroom_subject_over_global": best_subject - acc[best_global],
            "per_subject_best_a": [max(s["acc"], key=s["acc"].get) for s in per_subject],
        }
        r = out[ds]
        print(f"== {ds}: CAR {r['fixed_car']:.4f} raw {r['fixed_raw']:.4f} | best global a={best_global} "
              f"{r['best_global']:.4f} | best per-subject {best_subject:.4f} (+{r['headroom_subject_over_global']:.4f}) "
              f"| ORACLE per-trial {oracle_trial:.4f} (+{r['headroom_trial_over_subject']:.4f} over per-subject)",
              flush=True)
        print(f"   per-subject best a: {r['per_subject_best_a']}", flush=True)
    Path(__file__).with_name("reference_admittance_oracle.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
