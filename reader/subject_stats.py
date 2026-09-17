#!/usr/bin/env python3
"""Subject-level uncertainty for the READER paper (co-author priorities 1, 3 and 6; 2026-09-16).

Unit of inference = SUBJECT. Seeds are nested within subjects and are averaged first:
    d_i = (1/S_i) sum_s (A_{i,s}^READER - A_{i,s}^M)     over the seeds present for BOTH models.
Reported per corpus and comparator: mean d, 95% percentile bootstrap CI (10,000 resamples of the
subject-level d_i, i.e. paired resampling; fixed generator seed), two-sided Wilcoxon signed-rank on d_i
(zero differences discarded and counted; EXACT conditional sign-flip null over all 2^n patterns with
mid-ranks, valid under ties), subject W/T/L,
and Holm-adjusted p within a family declared in docs/CONTROLS.md:
    ENDPOINT family (12): READER vs {Compact, FF-Control, Compact-Mean, BiTE} x {2a, 2b, SD-SSVEP}
    PREFIX family  (12): READER vs Compact, same checkpoint, at the 4 exact durations x 3 corpora
Holm is applied only when every member of a family is present; otherwise the family is marked
incomplete and only unadjusted p is shown.
Exact-duration curves: every bootstrap draw resamples WHOLE subject curves, so pointwise intervals
at different durations come from the same subject draws. No duration is singled out.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# Set by main() from the command line; defaults follow scripts/reproduce_*.sh.
RUNS = ROOT / "runs"
EXACT = ROOT / "runs/exact_duration"
OUT = ROOT / "results/subject_stats"
BANK = ROOT / "runs/bank"
CORPORA = ["2a", "2b", "sdssvep"]
ENDPOINT_ARMS = {}                       # filled in main(): arm -> directory of <ds>_S<sub>_seed<seed> runs
COMPARATORS = ["compact", "ff_control", "compact_mean", "bite"]
PRIMARY_N = {"2a": [250, 500, 750, 1000], "2b": [250, 500, 750, 1000], "sdssvep": [64, 128, 192, 256]}
FS = {"2a": 250.0, "2b": 250.0, "sdssvep": 256.0}
B = 10_000
RNG_SEED = 20260916


def endpoint_acc(arm_dir, ds):
    out = {}
    for p in arm_dir.glob(f"{ds}_S*_seed*/summary.json"):
        s = json.loads(p.read_text())
        out[(s["subject"], s["seed"])] = 100 * s["final_test"]["acc"]
    return out


def subject_diffs(a, b):
    """{(sub, seed): acc} x2 -> {sub: mean over shared seeds of a - b}."""
    by = defaultdict(list)
    for k in set(a) & set(b):
        by[k[0]].append(a[k] - b[k])
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def subject_means(a):
    by = defaultdict(list)
    for (sub, _), v in a.items():
        by[sub].append(v)
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def boot_ci(d, rng):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank on subject-level differences.

    Zero differences are discarded (zero_method='wilcox') and counted. Ranks of |d| use MID-RANKS for
    ties. The null distribution is the EXACT conditional sign-flip distribution of T+ given those
    ranks, enumerated over all 2^n sign patterns (n <= 20 here), so it stays exact with ties -- unlike
    scipy's 'exact' (tie-free assumption) or the normal approximation (poor at n ~ 9).
    scipy's untied exact p is also reported as a cross-check when there are no ties or zeros.
    """
    d = np.asarray(d, float)
    zeros = int((np.abs(d) < 1e-9).sum())
    nz = d[np.abs(d) >= 1e-9]
    n = len(nz)
    if n == 0:
        return {"p": 1.0, "zeros": zeros, "ties": False, "method": "all differences zero", "n_used": 0}
    ranks = stats.rankdata(np.abs(nz))                 # average ranks for ties
    ties = len(np.unique(ranks)) < n
    t_obs = ranks[nz > 0].sum()
    expected = ranks.sum() / 2
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    t_all = (signs * ranks).sum(1)
    p = float(np.mean(np.abs(t_all - expected) >= abs(t_obs - expected) - 1e-9))
    out = {"p": p, "zeros": zeros, "ties": bool(ties), "method": "exact sign-flip (mid-ranks)", "n_used": int(n),
           "T_plus": float(t_obs)}
    if not ties and not zeros:
        out["scipy_exact_p"] = float(stats.wilcoxon(nz, method="exact").pvalue)
    return out


def summarise(d_by_subject, rng):
    d = list(d_by_subject.values())
    if len(d) < 2:
        return None
    lo, hi = boot_ci(d, rng)
    w = wilcoxon(d)
    return {"n_subjects": len(d), "mean": float(np.mean(d)), "ci95": [lo, hi],
            "wins": int(sum(x > 1e-9 for x in d)), "ties": int(sum(abs(x) <= 1e-9 for x in d)),
            "losses": int(sum(x < -1e-9 for x in d)), "wilcoxon": w, "per_subject": d_by_subject}


def holm(family):
    """family: {label: result or None}. Adds p_holm when complete."""
    present = {k: v for k, v in family.items() if v is not None}
    if len(present) != len(family):
        return False
    order = sorted(present, key=lambda k: present[k]["wilcoxon"]["p"])
    m, running = len(order), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * present[k]["wilcoxon"]["p"]))
        present[k]["p_holm"] = running
    return True


def fmt(r):
    if r is None:
        return "pending"
    w = r["wilcoxon"]
    ph = f", Holm p {r['p_holm']:.3f}" if "p_holm" in r else ""
    return (f"{r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  {r['wins']}/{r['ties']}/{r['losses']}  "
            f"p {w['p']:.3f} ({w['method']}, zeros {w['zeros']}){ph}")


def exact_curves(ds, arm_dir_name):
    """{(sub, seed): {n: acc%}}, plus gate interventions {(sub, seed): {g: {n: acc%}}}."""
    acc, gate = {}, {}
    for p in (EXACT / ds / arm_dir_name).glob("*.json"):
        r = json.loads(p.read_text())
        k = (r["subject"], r["seed"])
        acc[k] = {int(n): 100 * v for n, v in r["acc"].items()}
        if r.get("gate"):
            gate[k] = {float(g): {int(n): 100 * v for n, v in c.items()} for g, c in r["gate"].items()}
    return acc, gate


def curve_boot(diff_by_subject_n, grid, rng):
    """diff_by_subject_n: {sub: {n: d}} -> mean and pointwise CI per n, whole curves resampled together."""
    subs = sorted(diff_by_subject_n)
    M = np.array([[diff_by_subject_n[s][n] for n in grid] for s in subs])
    idx = rng.integers(0, len(subs), size=(B, len(subs)))
    boots = M[idx].mean(1)
    return {int(n): {"mean": float(M[:, j].mean()), "ci95": [float(np.percentile(boots[:, j], 2.5)),
                                                             float(np.percentile(boots[:, j], 97.5))]}
            for j, n in enumerate(grid)}


def main():
    global EXACT, OUT, BANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, default=ROOT / "runs/cohort", help="reader/, compact/, bite/")
    ap.add_argument("--controls", type=Path, default=ROOT / "runs/controls", help="ff_control/, compact_mean/")
    ap.add_argument("--bank", type=Path, default=ROOT / "runs/bank", help="per-deadline specialists")
    ap.add_argument("--exact", type=Path, default=ROOT / "runs/exact_duration",
                    help="reader/exact_duration.py output: <ds>/<arm>/*.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/subject_stats")
    args = ap.parse_args()
    EXACT, OUT, BANK = args.exact, args.out, args.bank
    ENDPOINT_ARMS.update({"reader": args.cohort / "reader", "compact": args.cohort / "compact",
                          "bite": args.cohort / "bite", "ff_control": args.controls / "ff_control",
                          "compact_mean": args.controls / "compact_mean"})
    rng = np.random.default_rng(RNG_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    report, lines = {"endpoint": {}, "exact_duration": {}}, ["# Subject-level statistics (READER paper)", ""]

    # ------------------------------------------------------------------ endpoint
    lines += ["## Endpoint accuracy: mean +/- SD across subject-level seed means", "",
              "| corpus | " + " | ".join(ENDPOINT_ARMS) + " |", "|---|" + "---:|" * len(ENDPOINT_ARMS)]
    accs = {ds: {arm: endpoint_acc(d, ds) for arm, d in ENDPOINT_ARMS.items()} for ds in CORPORA}
    for ds in CORPORA:
        cells = []
        for arm in ENDPOINT_ARMS:
            sm = subject_means(accs[ds][arm])
            cells.append(f"{np.mean(list(sm.values())):.2f} +/- {np.std(list(sm.values()), ddof=1):.2f} (n={len(sm)}, runs={len(accs[ds][arm])})"
                         if len(sm) > 1 else "pending")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    family = {}
    for ds in CORPORA:
        for comp in COMPARATORS:
            family[(ds, comp)] = summarise(subject_diffs(accs[ds]["reader"], accs[ds][comp]), rng) \
                if accs[ds][comp] else None
    complete = holm(family)
    lines += ["", f"## Endpoint: READER minus comparator, subject-level (family of 12, Holm {'applied' if complete else 'NOT applied: family incomplete'})",
              "", "mean d [95% bootstrap CI]  subject W/T/L  Wilcoxon p", "",
              "| corpus | " + " | ".join(f"vs {c}" for c in COMPARATORS) + " |", "|---|" + "---|" * len(COMPARATORS)]
    for ds in CORPORA:
        lines.append(f"| {ds} | " + " | ".join(fmt(family[(ds, c)]) for c in COMPARATORS) + " |")
    report["endpoint"] = {f"{ds}|{c}": v for (ds, c), v in family.items()}

    # ------------------------------------------------------------------ exact duration
    prefix_family = {}
    for ds in CORPORA:
        r_acc, r_gate = exact_curves(ds, "reader")
        if not r_acc:
            lines += ["", f"## Exact duration {ds}: pending"]
            continue
        grid = sorted(next(iter(r_acc.values())))
        others = {name: exact_curves(ds, name)[0] for name in ("compact", "ff_control", "compact_mean", "bite")}
        lines += ["", f"## Exact-duration, same-checkpoint accuracy: {ds} (runs: READER {len(r_acc)}, " +
                  ", ".join(f"{k} {len(v)}" for k, v in others.items()) + ")", "",
                  "| samples | seconds | READER | " + " | ".join(others) + " |", "|---:|---:|---:|" + "---:|" * len(others)]
        for n in grid:
            row = [f"{np.mean([c[n] for c in r_acc.values()]):.2f}"]
            for name, o in others.items():
                row.append(f"{np.mean([c[n] for c in o.values()]):.2f}" if o else "pending")
            lines.append(f"| {n} | {n / FS[ds]:.3f} | " + " | ".join(row) + " |")
        rep = {}
        for name, o in others.items():
            if not o:
                continue
            by = defaultdict(lambda: defaultdict(list))
            for k in set(r_acc) & set(o):
                for n in grid:
                    by[k[0]][n].append(r_acc[k][n] - o[k][n])
            diff = {s: {n: float(np.mean(v[n])) for n in grid} for s, v in by.items()}
            curve = curve_boot(diff, grid, rng)
            rep[name] = curve
            lines += ["", f"READER minus {name} (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):", "",
                      "| samples | mean d | 95% CI |", "|---:|---:|---|"]
            for n in grid:
                lines.append(f"| {n} | {curve[n]['mean']:+.2f} | [{curve[n]['ci95'][0]:+.2f}, {curve[n]['ci95'][1]:+.2f}] |")
            if name == "compact":
                for n in PRIMARY_N[ds]:
                    prefix_family[(ds, n)] = summarise({s: diff[s][n] for s in diff}, rng)
        # gate interventions at exact durations
        if r_gate:
            lines += ["", f"Gate interventions on trained READER ({ds}): accuracy at forced g minus learned-gate accuracy, subject-level mean [95% CI]", "",
                      "| samples | g=0 (forward only) | g=0.5 | g=1 (reversed only) |", "|---:|---|---|---|"]
            rep["gate"] = {}
            for n in grid:
                cells = []
                for g in (0.0, 0.5, 1.0):
                    by = defaultdict(list)
                    for k in r_gate:
                        by[k[0]].append(r_gate[k][g][n] - r_acc[k][n])
                    d = [float(np.mean(v)) for v in by.values()]
                    lo, hi = boot_ci(d, rng)
                    cells.append(f"{np.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]")
                    rep["gate"].setdefault(str(g), {})[n] = {"mean": float(np.mean(d)), "ci95": [lo, hi]}
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
        # specialists (separately retrained per duration) at the primary durations
        bank = BANK
        spec_rows = []
        for fam in ("bite", "compact"):
            tags = ("0.25s", "0.5s", "0.75s") if ds == "sdssvep" else ("1s", "2s", "3s")
            for n, tag in zip(PRIMARY_N[ds][:3], tags):
                arm_dir = bank / f"{fam}_{tag}"
                acc = endpoint_acc(arm_dir, ds) if arm_dir.exists() else {}
                if not acc:
                    continue
                if next(iter(json.loads(p.read_text())["window_samples"] for p in arm_dir.glob(f"{ds}_S*/summary.json"))) != n:
                    raise SystemExit(f"specialist {fam}_{tag} window does not equal {n} samples")
                r = summarise(subject_diffs({k: v[n] for k, v in r_acc.items()}, acc), rng)
                spec_rows.append((fam, n, np.mean(list(subject_means(acc).values())), r))
            full = endpoint_acc(ENDPOINT_ARMS[fam], ds)
            if spec_rows and any(f == fam for f, *_ in spec_rows):
                r = summarise(subject_diffs({k: v[PRIMARY_N[ds][3]] for k, v in r_acc.items()}, full), rng)
                spec_rows.append((fam, PRIMARY_N[ds][3], np.mean(list(subject_means(full).values())), r))
        if spec_rows:
            lines += ["", f"READER (one endpoint-trained checkpoint, exact duration) minus separately retrained specialists ({ds}); "
                      "the full-duration specialist is the cohort run", "", "| family | samples | specialist mean | READER - specialist |", "|---|---:|---:|---|"]
            for fam, n, m, r in spec_rows:
                lines.append(f"| {fam} | {n} | {m:.2f} | {fmt(r)} |")
        report["exact_duration"][ds] = rep
    complete_p = holm(prefix_family) if prefix_family else False
    if prefix_family:
        lines += ["", f"## PREFIX family: READER vs Compact, same checkpoint, primary durations (Holm {'applied' if complete_p else 'NOT applied: family incomplete'})", "",
                  "| corpus | samples | READER - Compact |", "|---|---:|---|"]
        for (ds, n), r in sorted(prefix_family.items()):
            lines.append(f"| {ds} | {n} | {fmt(r)} |")
        report["prefix_family"] = {f"{ds}|{n}": v for (ds, n), v in prefix_family.items()}
    (OUT / "SUBJECT_STATS.md").write_text("\n".join(lines) + "\n")
    (OUT / "subject_stats.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
