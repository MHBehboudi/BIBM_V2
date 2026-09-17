#!/usr/bin/env python3
"""Model zoo report: BiTE's released baselines re-trained under this repository's protocol, next to READER,
its controls, and BiTE's own PUBLISHED per-subject numbers.

Reported per corpus:
  1. zoo table      mean +/- SD across subject-level seed means (3 seeds), the seed-2025 mean (BiTE publishes
                    seed 2025 only, so this is the like-for-like column), BiTE's published mean, parameters.
  2. per subject    our 3-seed mean per subject, with BiTE's published value for that subject.
  3. READER minus   subject-level paired difference, 95% bootstrap CI, exact sign-flip Wilcoxon (mid-ranks),
                    subject W/T/L; Holm WITHIN this zoo family only. The pre-declared primary family lives in
                    reader/subject_stats.py and is not mixed with these secondary comparisons.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from reader import subject_stats as st  # noqa: E402

ZOO = ["ATCNet", "DMSANet", "DeepConvNet", "EEGNeX", "EEGNet", "EEGTCNet", "EISATC", "FACTNet",
       "MBCNNEATCFNet", "ShallowConvNet"]
PUBLISHED_NAME = {"DMSANet": "DMSACNN"}          # BiTE's table label for the same model
CORPORA = ["2a", "2b", "sdssvep", "hgd"]
SUBJECTS = {"2a": 9, "2b": 9, "sdssvep": 10, "hgd": 14}


def load(arm_dir, ds):
    acc, params = {}, set()
    for p in arm_dir.glob(f"{ds}_S*_seed*/summary.json"):
        s = json.loads(p.read_text())
        acc[(s["subject"], s["seed"])] = 100 * s["final_test"]["acc"]
        params.add(s["parameters"])
    return acc, (params.pop() if len(params) == 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", type=Path, default=ROOT / "runs/zoo", help="<Model>/<ds>_S<sub>_seed<seed>")
    ap.add_argument("--cohort", type=Path, default=ROOT / "runs/cohort", help="reader/, compact/, bite/")
    ap.add_argument("--controls", type=Path, default=ROOT / "runs/controls", help="ff_control/, compact_mean/")
    ap.add_argument("--published", type=Path, default=ROOT / "results/published_bite_tables.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/model_zoo")
    args = ap.parse_args()
    published = json.loads(args.published.read_text())["tables"]
    rng = np.random.default_rng(st.RNG_SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    arms = {**{m: args.zoo / m for m in ZOO},
            "BiTE": args.cohort / "bite", "Compact": args.cohort / "compact",
            "FF-Control": args.controls / "ff_control", "Compact-Mean": args.controls / "compact_mean",
            "READER": args.cohort / "reader"}
    report, lines = {}, ["# Model zoo: BiTE's baselines under one protocol", "",
                         "Official within-subject roles, 600 epochs, fixed final epoch, no selection, seeds 2025-2027. "
                         "Baselines are BiTE's released code via its own `get_model`. 'published' = BiTE's table "
                         "(seed 2025, their trainer). FACTNet cannot run on SD-SSVEP as released (FA hard-codes "
                         "seq_len=1000).", ""]
    for ds in CORPORA:
        data = {name: load(d, ds) for name, d in arms.items()}
        if not any(a for a, _ in data.values()):
            continue
        lines += [f"## {ds}", "", "| model | params | ours: mean +/- SD (subjects, runs) | ours: seed 2025 | published (seed 2025) |",
                  "|---|---:|---:|---:|---:|"]
        rows = []
        for name, (acc, params) in data.items():
            if not acc:
                pub = published.get(ds, {}).get(PUBLISHED_NAME.get(name, name))
                if pub and name in ZOO:
                    rows.append((name, params, None, None, pub["mean_of_subjects"], None))
                continue
            sm = st.subject_means(acc)
            s25 = [v for (sub, seed), v in acc.items() if seed == 2025]
            full25 = len(s25) == SUBJECTS[ds]
            pub = published.get(ds, {}).get(PUBLISHED_NAME.get(name, name))
            rows.append((name, params, (np.mean(list(sm.values())), np.std(list(sm.values()), ddof=1) if len(sm) > 1 else float("nan"),
                                        len(sm), len(acc)), np.mean(s25) if full25 else None,
                         pub["mean_of_subjects"] if pub else None, sm))
        rows.sort(key=lambda r: -(r[2][0] if r[2] else -1))
        for name, params, ours, s25, pub, _ in rows:
            ours_s = f"{ours[0]:.2f} +/- {ours[1]:.2f} ({ours[2]}, {ours[3]})" if ours else "not run"
            lines.append(f"| {name} | {params or '-'} | {ours_s} | {f'{s25:.2f}' if s25 is not None else '-'} | "
                         f"{f'{pub:.2f}' if pub is not None else '-'} |")
        # per subject
        subs = list(range(1, SUBJECTS[ds] + 1))
        lines += ["", f"Per subject, ours (3-seed mean) / published:", "",
                  "| model | " + " | ".join(f"S{s}" for s in subs) + " |", "|---|" + "---:|" * len(subs)]
        for name, params, ours, s25, pub, sm in rows:
            if not sm:
                continue
            p = published.get(ds, {}).get(PUBLISHED_NAME.get(name, name), {}).get("per_subject", {})   # JSON keys are str
            cells = [(f"{sm[s]:.1f}" + (f" / {p[str(s)]:.1f}" if str(s) in p else "")) if s in sm else "-" for s in subs]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        # READER minus each other model
        reader_acc = data["READER"][0]
        family = {}
        for name, (acc, _) in data.items():
            if name == "READER" or not acc or not reader_acc:
                continue
            family[name] = st.summarise(st.subject_diffs(reader_acc, acc), rng)
        family = {k: v for k, v in family.items() if v is not None}
        st.holm(family)
        if family:
            lines += ["", "READER minus model, subject-level (Holm within this corpus's zoo family):", "",
                      "| model | mean d [95% CI]  W/T/L  p |", "|---|---|"]
            for name in sorted(family, key=lambda k: family[k]["mean"]):
                lines.append(f"| {name} | {st.fmt(family[name])} |")
        report[ds] = {"table": [{"model": r[0], "params": r[1],
                                 "ours_mean": r[2][0] if r[2] else None, "ours_sd": r[2][1] if r[2] else None,
                                 "subjects": r[2][2] if r[2] else 0, "runs": r[2][3] if r[2] else 0,
                                 "ours_seed2025": r[3], "published": r[4],
                                 "per_subject": r[5] or {}} for r in rows],
                      "reader_minus": family}
        lines.append("")
    (args.out / "MODEL_ZOO.md").write_text("\n".join(lines) + "\n")
    (args.out / "model_zoo.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
