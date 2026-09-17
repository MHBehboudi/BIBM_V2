#!/usr/bin/env python3
"""Paper tables computed ONLY from results/raw/runs.csv, so every number can be checked without a GPU.

  results/MAIN_TABLE.md (+ main_table.json)        within-subject: READER vs every baseline RE-RUN here with
                                                    3 seeds (BiTE's 10 released baselines + BiTE), next to
                                                    the published single-seed numbers
  results/CROSS_SUBJECT.md (+ cross_subject_stats.json)   leave-one-subject-out, all seeds present
  results/DEVELOPMENT_DISCLOSURE.md (+ .json)      screen subjects vs subjects the screen never used
  results/REPRODUCIBILITY.md (+ .json)             GPU per arm, MIG vs full-GPU re-runs, peak-to-final decay

Statistics follow reader/subject_stats.py: the unit is the SUBJECT (seeds averaged first), 95% paired
bootstrap CI over subjects, exact sign-flip Wilcoxon with mid-ranks, subject W/T/L, Holm within a declared
family. A model whose runs are incomplete for a corpus is shown with its run count and is excluded from
"strongest baseline" and from Holm families until complete.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from reader import subject_stats as st  # noqa: E402

CORPORA = ["2a", "2b", "hgd", "sdssvep"]
NAMES = {"2a": "BCI IV-2a", "2b": "BCI IV-2b", "hgd": "HGD", "sdssvep": "SD-SSVEP"}
SUBJECTS = {"2a": 9, "2b": 9, "hgd": 14, "sdssvep": 10}
SEEDS = 3
ZOO = ["ATCNet", "DMSANet", "DeepConvNet", "EEGNeX", "EEGNet", "EEGTCNet", "EISATC", "FACTNet",
       "MBCNNEATCFNet", "ShallowConvNet"]
PUBLISHED_NAME = {"DMSANet": "DMSACNN"}
PUBLISHED_LOSO_BITE = {"2a": 64.56, "2b": 76.44, "sdssvep": 79.72}
# BiTE's paper Table I; its released spreadsheet rounds HGD to 95.94. Other baselines use the spreadsheet means.
BITE_PAPER = {"2a": 85.34, "2b": 88.37, "hgd": 95.93, "sdssvep": 94.16}
# the 8-cell development screen (scripts/make_manifest.py SCREEN): READER was chosen among 8 arms on these
SCREEN = {"2a": {2, 9}, "2b": {4, 9}, "hgd": {11, 14}, "sdssvep": {1, 5}}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["subject"], r["seed"] = int(r["subject"]), int(r["seed"])
        for k in ("test_acc", "test_kappa", "test_peak_acc_diagnostic"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
    return rows


def acc_map(rows, group, arm, ds, key="test_acc"):
    return {(r["subject"], r["seed"]): r[key] for r in rows
            if r["group"] == group and r["arm"] == arm and r["dataset"] == ds}


def describe(acc, kappa=None):
    sm = st.subject_means(acc)
    v = list(sm.values())
    out = {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
           "subjects": len(v), "runs": len(acc), "seeds": len({k[1] for k in acc})}
    if kappa:
        km = list(st.subject_means(kappa).values())
        out["kappa"] = float(np.nanmean(km)) if km else None
    return out


def cell(d, complete):
    if d["mean"] is None:
        return "not run"
    s = f"{d['mean']:.2f} ± {d['sd']:.2f}" if d["sd"] is not None else f"{d['mean']:.2f}"
    if d.get("kappa") is not None:
        s += f" / {d['kappa']:.2f}"
    return s if complete else f"{s} (partial: {d['runs']} runs)"


def main_table(rows, published, rng):
    models = [("zoo", m, m) for m in ZOO] + [("cohort", "bite", "BiTE"), ("cohort", "compact", "Compact (READER without the reversed branch)"),
                                              ("cohort", "reader", "**READER**")]
    report = {"corpora": {}, "models": {}}
    table = {}
    for group, arm, label in models:
        table[label] = {}
        params = sorted({int(r["parameters"]) for r in rows if r["group"] == group and r["arm"] == arm})
        for ds in CORPORA:
            acc = acc_map(rows, group, arm, ds)
            kap = acc_map(rows, group, arm, ds, "test_kappa")
            d = describe(acc, kap)
            d["complete"] = d["runs"] == SUBJECTS[ds] * SEEDS
            pub = published.get(ds, {}).get(PUBLISHED_NAME.get(arm, "BiTE" if arm == "bite" else arm))
            d["published"] = (BITE_PAPER[ds] if arm == "bite" else pub["mean_of_subjects"]) \
                if pub and (group == "zoo" or arm == "bite") else None
            d["params"] = sorted({int(r["parameters"]) for r in rows if r["group"] == group and r["arm"] == arm
                                  and r["dataset"] == ds})
            d["acc"] = acc
            table[label][ds] = d
        report["models"][label] = {"group": group, "arm": arm, "parameters": params}

    reader = "**READER**"
    lines = ["# Main table: within-subject accuracy against baselines RE-RUN here", "",
             "Every model below was trained by this repository's harness on the official within-subject roles "
             "(session 1 train, session 2 test), 600 epochs, fixed final epoch, no validation set and no model "
             "selection, **3 seeds (2025-2027)**. Cells are mean ± SD across subjects (seeds averaged per subject) "
             "/ Cohen's kappa. *published* is BiTE's own table (arXiv 2510.10004, one seed, their trainer).", "",
             "Baselines are BiTE's released implementations loaded through BiTE's own `get_model`, unchanged, and "
             "trained with this repository's recipe (Adam 2e-3, cosine, label smoothing .1, batch 64, float32, "
             "gradient clip 5). BiTE itself is trained with `--clip 0`, as released. FACTNet cannot run on "
             "SD-SSVEP as released (its attention hard-codes 1000 samples).", ""]
    header = "| model | params (2a) | " + " | ".join(f"{NAMES[d]} ours | pub." for d in CORPORA) + " | corpus-bal. |"
    lines += [header, "|---|---:|" + "---:|---:|" * len(CORPORA) + "---:|"]
    order = sorted(table, key=lambda k: (k == reader, k.startswith("Compact"), k == "BiTE",
                                         np.mean([table[k][d]["mean"] for d in CORPORA if table[k][d]["mean"] is not None])))
    for label in order:
        t = table[label]
        bal = [t[d]["mean"] for d in CORPORA if t[d]["complete"]]
        balanced = f"{np.mean(bal):.2f}" if len(bal) == len(CORPORA) else "—"
        p2a = t["2a"]["params"]
        lines.append(f"| {label} | {p2a[0] if len(p2a) == 1 else ('-'.join(map(str, p2a)) if p2a else '—')} | " +
                     " | ".join(f"{cell(t[d], t[d]['complete'])} | {t[d]['published']:.2f}" if t[d]["published"] is not None
                                else f"{cell(t[d], t[d]['complete'])} | —" for d in CORPORA) + f" | {balanced} |")
        report["models"][label]["corpus_balanced"] = float(np.mean(bal)) if len(bal) == len(CORPORA) else None

    lines += ["", "## READER against the strongest re-run baseline on each corpus", "",
              "Strongest = highest 3-seed mean among COMPLETE re-run baselines (choosing the maximum of eleven "
              "noisy means favours the baseline). Subject-level paired difference, 95% bootstrap CI, subject "
              "W/T/L, exact Wilcoxon p; Holm over READER vs all eleven baselines on that corpus once complete. "
              "Full per-subject detail: `results/model_zoo/MODEL_ZOO.md`.", "",
              "| corpus | READER | strongest re-run baseline | READER minus it | READER minus BiTE (re-run) | READER rank |",
              "|---|---:|---|---|---|---:|"]
    for ds in CORPORA:
        baselines = {k: table[k][ds] for k in table if k not in (reader,) and not k.startswith("Compact")}
        family = {k: (st.summarise(st.subject_diffs(table[reader][ds]["acc"], v["acc"]), rng) if v["complete"] else None)
                  for k, v in baselines.items() if v["mean"] is not None}
        holm_done = st.holm(family)
        complete = {k: v for k, v in baselines.items() if v["complete"]}
        strongest = max(complete, key=lambda k: complete[k]["mean"]) if complete else None
        means = sorted([table[k][ds]["mean"] for k in table
                        if table[k][ds]["mean"] is not None and not k.startswith("Compact")], reverse=True)
        rank = means.index(table[reader][ds]["mean"]) + 1 if table[reader][ds]["mean"] in means else None
        n_incomplete = sum(1 for v in baselines.values() if v["mean"] is not None and not v["complete"])
        note = f" ({n_incomplete} baselines still incomplete)" if n_incomplete else ""
        lines.append(f"| {NAMES[ds]} | {table[reader][ds]['mean']:.2f} | "
                     f"{strongest} {complete[strongest]['mean']:.2f}{note} | {st.fmt(family.get(strongest))} | "
                     f"{st.fmt(family.get('BiTE'))} | {rank} of {len(means)} |" if strongest else
                     f"| {NAMES[ds]} | {table[reader][ds]['mean']:.2f} | pending | pending | pending | — |")
        report["corpora"][ds] = {"strongest_rerun_baseline": strongest, "holm_complete": holm_done,
                                 "reader_rank": rank, "models_ranked": len(means),
                                 "reader_minus": {k: v for k, v in family.items() if v is not None}}

    lines += ["", "## How closely the re-runs track the published numbers", "",
              "BiTE's table is one seed (2025) with BiTE's own trainer. Our seed-2025 run of the same model, subject means "
              "averaged, minus the published value. Large gaps would mean the harness disadvantages a baseline.", "",
              "| corpus | models compared | mean (ours − published) | mean absolute gap | largest gaps |", "|---|---:|---:|---:|---|"]
    tracking = {}
    for ds in CORPORA:
        gaps = {}
        for label, t in table.items():
            d = t[ds]
            if d["published"] is None:
                continue
            s25 = [r["test_acc"] for r in rows if r["dataset"] == ds and r["seed"] == 2025 and
                   (r["group"], r["arm"]) == (report["models"][label]["group"], report["models"][label]["arm"])]
            if len(s25) == SUBJECTS[ds]:
                gaps[label] = float(np.mean(s25)) - d["published"]
        if gaps:
            v = np.array(list(gaps.values()))
            worst = sorted(gaps.items(), key=lambda kv: -abs(kv[1]))[:3]
            lines.append(f"| {NAMES[ds]} | {len(v)} | {v.mean():+.2f} | {np.abs(v).mean():.2f} | " +
                         ", ".join(f"{k} {g:+.2f}" for k, g in worst) + " |")
            tracking[ds] = gaps
    report["rerun_minus_published_seed2025"] = tracking

    lines += ["", "## Published bars (single seed) for reference", "",
              "| corpus | best published | held by | READER (3-seed) | READER minus bar |", "|---|---:|---|---:|---:|"]
    for ds in CORPORA:
        pub = {m: (BITE_PAPER[ds] if m == "BiTE" else v["mean_of_subjects"]) for m, v in published[ds].items()}
        best = max(pub, key=pub.get)
        lines.append(f"| {NAMES[ds]} | {pub[best]:.2f} | {'BiTE' if best == 'BiTE' else best} | "
                     f"{table[reader][ds]['mean']:.2f} | {table[reader][ds]['mean'] - pub[best]:+.2f} |")
    lines += ["", "A published single-seed mean and a 3-seed mean are not a paired comparison; the re-run table above "
              "is the comparison of record."]
    for label in table:
        for ds in CORPORA:
            table[label][ds].pop("acc")
    report["table"] = table
    return lines, report


def cross_subject(rows, rng):
    arms = [("bite", "BiTE (--clip 0, as released)"), ("compact", "Compact"), ("reader", "**READER**")]
    lines = ["# Cross-subject (leave-one-subject-out)", "",
             "BiTE's cross-subject protocol: every other subject's sessions pooled as the training role, the held "
             "subject never seen; Euclidean alignment as BiTE enables it; 600 epochs, fixed final epoch. BiTE "
             "publishes no HGD row, so none is run. Mean ± SD across held subjects (seeds averaged per subject).", "",
             "| model | " + " | ".join(f"{NAMES[d]} | seeds (runs)" for d in PUBLISHED_LOSO_BITE) + " |",
             "|---|" + "---:|---:|" * len(PUBLISHED_LOSO_BITE)]
    out = {"arms": {}, "reader_minus": {}}
    data = {}
    for arm, label in arms:
        data[arm] = {}
        cells = []
        for ds in PUBLISHED_LOSO_BITE:
            acc = acc_map(rows, "loso", arm, ds)
            d = describe(acc)
            data[arm][ds] = acc
            out["arms"].setdefault(arm, {})[ds] = d
            cells.append(f"{d['mean']:.2f} ± {d['sd']:.2f} | {d['seeds']} ({d['runs']})" if d["mean"] is not None
                         else "pending | 0")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| BiTE published (one seed) | " + " | ".join(f"{v:.2f} | 1" for v in PUBLISHED_LOSO_BITE.values()) + " |")
    old = {ds: describe(acc_map(rows, "superseded", "loso_bite_clip5", ds)) for ds in PUBLISHED_LOSO_BITE}
    if any(v["mean"] is not None for v in old.values()):
        lines.append("| *superseded: BiTE seed 2025 at clip 5* | " + " | ".join(
            f"*{v['mean']:.2f}* | {v['seeds']} ({v['runs']})" if v["mean"] is not None else "— | 0" for v in old.values()) + " |")
        out["superseded_bite_clip5"] = old
    lines += ["", "READER minus comparator, subject-level (seeds present for BOTH arms averaged per subject):", "",
              "| corpus | vs BiTE | vs Compact |", "|---|---|---|"]
    for ds in PUBLISHED_LOSO_BITE:
        res = {}
        for other in ("bite", "compact"):
            d = st.subject_diffs(data["reader"][ds], data[other][ds])
            res[other] = st.summarise(d, rng) if d else None
        out["reader_minus"][ds] = res
        lines.append(f"| {NAMES[ds]} | {st.fmt(res['bite'])} | {st.fmt(res['compact'])} |")
    lines += ["", "The seed-2025 BiTE runs first reported here were trained at the trainer's default gradient clip "
              "of 5 by mistake; BiTE's release does not clip. They are shown in italics and are not the record."]
    return lines, out


def disclosure(rows, rng):
    lines = ["# Development disclosure: screen subjects vs subjects the screen never used", "",
             "READER was selected among eight architecture arms on an 8-cell screen scored on the OFFICIAL TEST "
             "session (the hardest and easiest subject per corpus: " +
             ", ".join(f"{NAMES[d]} S{'/S'.join(map(str, sorted(s)))}" for d, s in SCREEN.items()) +
             "). The within-subject cohort then added the other 34 subjects. The table separates the two, so the "
             "selection bias can be bounded rather than assumed away.", "",
             "| comparison | subjects | per corpus (2a / 2b / HGD / SD) | corpus-balanced |", "|---|---|---|---:|"]
    out = {}
    for other in ("compact", "bite"):
        for part in ("screen", "fresh"):
            per = {}
            for ds in CORPORA:
                d = st.subject_diffs(acc_map(rows, "cohort", "reader", ds), acc_map(rows, "cohort", other, ds))
                d = {s: v for s, v in d.items() if (s in SCREEN[ds]) == (part == "screen")}
                per[ds] = (float(np.mean(list(d.values()))), len(d))
            bal = float(np.mean([v[0] for v in per.values()]))
            out[f"reader_minus_{other}_{part}"] = {"per_corpus": per, "corpus_balanced": bal}
            lines.append(f"| READER − {other} | {part} ({sum(v[1] for v in per.values())}) | " +
                         " / ".join(f"{v[0]:+.2f}" for v in per.values()) + f" | {bal:+.2f} |")
    lines += ["", "If the gain were created by selecting on the screen subjects, it would vanish on the fresh ones."]
    return lines, out


def reproducibility(rows, rng):
    lines = ["# Reproducibility", "", "## GPU of every run of record", "",
             "H100 80GB and H200 NVL reproduce each other bit-for-bit (45/45 identical READER re-runs). MIG-partitioned "
             "H100 NVL slices do not, so every MIG run of a comparison of record is re-run on a full GPU and the "
             "re-run is the record.", "", "| runs | arm | GPU | runs | record status |", "|---|---|---|---:|---|"]
    counts = defaultdict(int)
    for r in rows:
        if r["group"] in ("cohort", "controls", "anytime", "bank", "loso"):
            counts[(r["group"], r["arm"], r["device"], r["record_status"])] += 1
    for (g, a, dev, stat), n in sorted(counts.items()):
        lines.append(f"| {g} | {a} | {dev} | {n} | {stat} |")
    pending = sum(n for (g, a, dev, stat), n in counts.items() if stat == "mig_pending")

    lines += ["", "## MIG original minus full-GPU re-run (identical arguments)", ""]
    mig = [r for r in rows if r["group"] == "mig"]
    out = {"mig_pending_runs": pending, "mig_vs_full": {}}
    if not mig:
        lines.append(f"Pending: {pending} MIG runs are queued for full-GPU re-runs (jobs 409509/409510).")
    else:
        rec = {(r["group"], r["arm"], r["dataset"], r["subject"], r["seed"]): r["test_acc"] for r in rows if r["group"] != "mig"}
        by = defaultdict(list)
        for r in mig:
            _, g, a = r["arm"].split("/", 2) if r["arm"].count("/") >= 2 else ("", *r["arm"].split("/", 1))
            key = (g, a, r["dataset"], r["subject"], r["seed"])
            if key in rec:
                by[(g, a)].append(r["test_acc"] - rec[key])
        lines += ["| runs | arm | pairs | mean (MIG − full) | se | pairs that differ | mean abs difference |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for (g, a), d in sorted(by.items()):
            d = np.array(d)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            lines.append(f"| {g} | {a} | {len(d)} | {d.mean():+.2f} | {se:.2f} | {(np.abs(d) > 1e-6).sum()} | {np.abs(d).mean():.2f} |")
            out["mig_vs_full"][f"{g}/{a}"] = {"pairs": len(d), "mean": float(d.mean()), "se": float(se),
                                              "differ": int((np.abs(d) > 1e-6).sum())}
        if pending:
            lines.append(f"\n{pending} MIG runs are still waiting for their re-runs.")

    lines += ["", "## Test-curve peak minus reported final epoch (diagnostic; never used for selection)", "",
              "BiTE's protocol trains 600 epochs and reports the final epoch with no validation set. Every arm loses "
              "accuracy between its own test-curve peak and that final epoch. The loss is a property of the recipe, "
              "not of READER, if it is similar across arms.", "",
              "| runs | arm | runs | mean peak − final (pp) | se |", "|---|---|---:|---:|---:|"]
    decay = defaultdict(list)
    for r in rows:
        if r["group"] in ("cohort", "zoo", "controls") and not np.isnan(r["test_peak_acc_diagnostic"]):
            decay[(r["group"], r["arm"])].append(r["test_peak_acc_diagnostic"] - r["test_acc"])
    out["peak_minus_final"] = {}
    for (g, a), d in sorted(decay.items(), key=lambda kv: np.mean(kv[1])):
        d = np.array(d)
        lines.append(f"| {g} | {a} | {len(d)} | {d.mean():.2f} | {d.std(ddof=1) / np.sqrt(len(d)):.2f} |")
        out["peak_minus_final"][f"{g}/{a}"] = {"runs": len(d), "mean": float(d.mean())}
    return lines, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-csv", type=Path, default=ROOT / "results/raw/runs.csv")
    ap.add_argument("--published", type=Path, default=ROOT / "results/published_bite_tables.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    rows = load(args.runs_csv)
    published = json.loads(args.published.read_text())["tables"]
    for name, fn in (("MAIN_TABLE", lambda: main_table(rows, published, np.random.default_rng(st.RNG_SEED))),
                     ("CROSS_SUBJECT", lambda: cross_subject(rows, np.random.default_rng(st.RNG_SEED))),
                     ("DEVELOPMENT_DISCLOSURE", lambda: disclosure(rows, np.random.default_rng(st.RNG_SEED))),
                     ("REPRODUCIBILITY", lambda: reproducibility(rows, np.random.default_rng(st.RNG_SEED)))):
        lines, data = fn()
        (args.out / f"{name}.md").write_text("\n".join(lines) + "\n")
        (args.out / f"{name.lower()}.json").write_text(json.dumps(data, indent=1, default=str))
        print(f"-> results/{name}.md")


if __name__ == "__main__":
    main()
