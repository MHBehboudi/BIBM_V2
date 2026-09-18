#!/usr/bin/env python3
"""A single self-contained HTML page over every committed result: results/dashboard.html.

The paper's tables answer questions one at a time. This answers them together -- pick a corpus and
see where READER sits among twelve models, which subjects that mean is made of, what it does before
the trial ends, and which trials it gets wrong -- without a server, a build step or a network call.

It is GENERATED, never hand-edited: everything it draws is read from results/ at build time and
embedded as one JSON blob, so the page cannot drift from the tables. Re-run after any rebuild:

    python scripts/make_dashboard.py

Colour follows the model, never its rank (READER blue, Compact orange, BiTE aqua); magnitude uses a
single-hue blue ramp; the light and dark palettes are the validated steps of the same hues. Every
chart has a table view behind a toggle, because three of the light-surface hues sit below 3:1.
"""
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CORPORA = ["2a", "2b", "hgd", "sdssvep"]
TITLE = {"2a": "BCI IV-2a", "2b": "BCI IV-2b", "hgd": "HGD", "sdssvep": "SD-SSVEP"}
NAMED = {"reader": "READER", "compact": "Compact", "bite": "BiTE"}


def read_json(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def build_payload():
    runs = [r for r in csv.DictReader(open(RES / "raw/runs.csv")) if r["protocol"] == "within"]

    # model -> corpus -> {subjects: {id: acc}, mean, params}
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    params = {}
    for r in runs:
        if r["group"] == "cohort":
            name = NAMED[r["arm"]]
        elif r["group"] == "zoo":
            name = r["arm"]
        else:
            continue
        acc[name][r["dataset"]][int(r["subject"])].append(float(r["test_acc"]))
        params[(name, r["dataset"])] = int(r["parameters"])

    models = {}
    for name, per_ds in acc.items():
        models[name] = {}
        for ds, subs in per_ds.items():
            per_subject = {str(s): round(sum(v) / len(v), 4) for s, v in sorted(subs.items())}
            vals = list(per_subject.values())
            models[name][ds] = {"subjects": per_subject, "mean": round(sum(vals) / len(vals), 4),
                                "params": params[(name, ds)], "seeds": len(next(iter(subs.values())))}

    main = read_json("main_table.json") or {}
    deltas = {ds: {k: {"mean": v["mean"], "ci95": v["ci95"], "w": v["wins"], "t": v["ties"],
                       "l": v["losses"], "p": v["wilcoxon"]["p"], "p_holm": v.get("p_holm")}
                   for k, v in main.get("corpora", {}).get(ds, {}).get("reader_minus", {}).items()}
              for ds in CORPORA if ds in main.get("corpora", {})}

    anytime = {}
    for ds in ["2a", "2b", "sdssvep"]:
        d = read_json(f"anytime_{ds}.json")
        if not d:
            continue
        anytime[ds] = {"arm": d.get("reader_arm"), "full_trial_s": d.get("full_trial_s"), "deadlines": {
            k: {"reader": v["reader_mean"], "banks": v["bank_means"],
                "strongest": v["strongest_bank"], "delta": v["paired_delta_vs_strongest"],
                "ci95": (v.get("subject_level_vs", {}).get(v["strongest_bank"]) or {}).get("ci95")}
            for k, v in d["deadlines"].items()}}

    err = read_json("error_analysis.json") or {"corpora": {}}
    errors = {}
    for ds, e in err["corpora"].items():
        errors[ds] = {
            "n_classes": e["n_classes"], "class_names": e["class_names"],
            "confusion": {a: v["confusion_rownorm"] for a, v in e["arms"].items()},
            "recall": {a: v["per_class_recall"] for a, v in e["arms"].items()},
            "reliability": {a: [[b["confidence"], b["accuracy"], b["n"]] for b in v["reliability_bins"]
                                if b["n"] >= 25] for a, v in e["arms"].items()},
            "ece": {a: v["ece"] for a, v in e["arms"].items()},
            "stability": {a: v["stability"] for a, v in e["arms"].items()},
            "complementarity": (e.get("complementarity_reader_vs_bite") or {}).get("pooled"),
        }

    abl = read_json("ablation.json") or {}
    prov = list(csv.DictReader(open(RES / "raw/record_provenance.csv")))
    return {
        "built": date.today().isoformat(),
        "n_runs": len(prov),
        "corpora": CORPORA,
        "titles": TITLE,
        "models": models,
        "reader_minus": deltas,
        "strongest": {ds: main["corpora"][ds].get("strongest_rerun_baseline")
                      for ds in CORPORA if ds in main.get("corpora", {})},
        "rank": {ds: main["corpora"][ds].get("reader_rank") for ds in CORPORA if ds in main.get("corpora", {})},
        "published": {ds: {k: v[ds].get("published") for k, v in main.get("table", {}).items()
                           if ds in v and v[ds].get("published") is not None} for ds in CORPORA},
        "anytime": anytime,
        "errors": errors,
        "ablation": {"minus_reader": abl.get("minus_reader", {}),
                     "ps": abl.get("reader_ps_minus_compact_ps", {})},
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>READER Results</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,.10);
  --reader: #2a78d6; --compact: #eb6834; --bite: #1baf7a; --other: #898781;
  --seq-1: #cde2fb; --seq-2: #9ec5f4; --seq-3: #3987e5; --seq-4: #184f95; --seq-5: #0d366b;
  --pos: #2a78d6; --neg: #d03b3b; --mid: #f0efec; --fill-soft: #9ec5f4; --fill-none: #d0cfca;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #fff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
  --reader: #3987e5; --compact: #d95926; --bite: #199e70;
  --seq-1: #104281; --seq-2: #1c5cab; --seq-3: #2a78d6; --seq-4: #6da7ec; --seq-5: #b7d3f6;
  --pos: #3987e5; --neg: #e66767; --mid: #383835; --fill-soft: #1c5cab; --fill-none: #4a4a46;
} }
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #fff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
  --reader: #3987e5; --compact: #d95926; --bite: #199e70;
  --seq-1: #104281; --seq-2: #1c5cab; --seq-3: #2a78d6; --seq-4: #6da7ec; --seq-5: #b7d3f6;
  --pos: #3987e5; --neg: #e66767; --mid: #383835; --fill-soft: #1c5cab; --fill-none: #4a4a46;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 32px 16px 80px; }
header h1 { font-size: 1.6rem; margin: 0 0 6px; letter-spacing: -.01em; }
header p { color: var(--ink-2); margin: 0 0 4px; max-width: 70ch; }
.meta { color: var(--muted); font-size: .8rem; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 24px 0 20px;
  position: sticky; top: 0; z-index: 20; background: var(--page); padding: 10px 0; border-bottom: 1px solid var(--grid); }
.bar .lab { color: var(--muted); font-size: .8rem; margin-right: 2px; }
button.chip { font: inherit; font-size: .85rem; padding: 5px 12px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--ring); background: var(--surface); color: var(--ink-2); }
button.chip[aria-pressed="true"] { background: var(--reader); border-color: var(--reader); color: #fff; }
button.chip:focus-visible { outline: 2px solid var(--reader); outline-offset: 2px; }
.theme { margin-left: auto; }
section { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 20px 20px 16px; margin-bottom: 20px; }
section h2 { font-size: 1.02rem; margin: 0 0 4px; }
section .sub { color: var(--ink-2); font-size: .85rem; margin: 0 0 14px; max-width: 78ch; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }
.tile .k { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.tile .v { font-size: 1.7rem; font-weight: 650; letter-spacing: -.02em; margin: 2px 0; font-variant-numeric: tabular-nums; }
.tile .d { font-size: .82rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--ink-2); font-size: 11px; }
.axis line, .axis path { stroke: var(--axis); }
.gridline { stroke: var(--grid); stroke-width: 1; }
.tt { position: fixed; pointer-events: none; z-index: 60; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 7px 10px; font-size: .8rem; opacity: 0;
  transition: opacity .1s; box-shadow: 0 6px 20px rgba(0,0,0,.14); max-width: 280px; font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 2px; font-size: .82rem; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
details { margin-top: 12px; }
details summary { cursor: pointer; font-size: .82rem; color: var(--ink-2); }
table { border-collapse: collapse; width: 100%; font-size: .8rem; margin-top: 10px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
.note { font-size: .78rem; color: var(--muted); margin-top: 10px; max-width: 80ch; }
.scroll { overflow-x: auto; }
@media (max-width: 640px) { .wrap { padding: 20px 16px 60px; } header h1 { font-size: 1.3rem; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>READER — prefix-bidirectional reading for EEG decoding</h1>
  <p>One model that emits a legal decision at every token boundary, against BiTE and its ten released
     baselines re-run under one protocol: official roles, 600 epochs, fixed final epoch, no validation
     set, no model selection, three seeds.</p>
  <p class="meta" id="meta"></p>
</header>

<div class="bar">
  <span class="lab">Corpus</span><span id="corpusChips"></span>
  <button class="chip theme" id="themeBtn" aria-pressed="false">Dark</button>
</div>

<section>
  <h2>Where READER stands</h2>
  <p class="sub">Subject means over three seeds. The delta is paired at the subject level against the
     strongest baseline re-run here, with a 95% bootstrap CI.</p>
  <div class="tiles" id="tiles"></div>
</section>

<section>
  <h2>Accuracy against size</h2>
  <p class="sub">Up is more accurate, left is smaller. Hollow markers are BiTE's released baselines.
     Hover any point for its numbers.</p>
  <svg id="landscape" viewBox="0 0 900 380" role="img" aria-label="Accuracy against parameter count"></svg>
  <div class="legend" id="landLegend"></div>
  <details><summary>Table view</summary><div class="scroll" id="landTable"></div></details>
</section>

<section>
  <h2>Who the mean is made of</h2>
  <p class="sub">One row per subject: BiTE's accuracy and READER's, joined by a line coloured by which
     one won. A corpus mean can hide a decoder that helps four subjects and hurts five.</p>
  <svg id="dumbbell" viewBox="0 0 900 420" role="img" aria-label="Per-subject accuracy"></svg>
  <div class="legend" id="dumbLegend"></div>
  <details><summary>Table view</summary><div class="scroll" id="dumbTable"></div></details>
</section>

<section>
  <h2>Before the trial ends</h2>
  <p class="sub">One READER read at each deadline, against banks of specialists retrained from scratch
     for that deadline. Positive means one model beats the whole bank.</p>
  <svg id="anytime" viewBox="0 0 900 360" role="img" aria-label="Accuracy at each decision deadline"></svg>
  <div class="legend" id="anyLegend"></div>
  <p class="note" id="anyNote"></p>
</section>

<section>
  <h2>Which trials, and whose</h2>
  <p class="sub">READER and BiTE see the same trials in the same order. The oracle bar is the ceiling a
     perfect selector between the two would reach — it is a diagnostic, not a proposed system.</p>
  <svg id="comp" viewBox="0 0 900 150" role="img" aria-label="Error overlap between READER and BiTE"></svg>
  <div class="legend" id="compLegend"></div>
  <h2 style="margin-top:22px">Which classes</h2>
  <p class="sub">READER's confusion, row-normalised (%), pooled over subjects and seeds.</p>
  <svg id="conf" viewBox="0 0 900 330" role="img" aria-label="Confusion matrix"></svg>
  <p class="note" id="confNote"></p>
</section>

<section>
  <h2>Every model, every subject</h2>
  <p class="sub">Darker is more accurate. The pale columns are the subjects every model finds hard —
     they are the same subjects across architectures, which is the part a mean never shows.</p>
  <div class="scroll"><svg id="heat" viewBox="0 0 900 460" role="img" aria-label="Model by subject accuracy"></svg></div>
</section>

<footer class="meta">Generated by <code>scripts/make_dashboard.py</code> from <code>results/</code>.
  Every number here has a table in that directory.</footer>
</div>
<div class="tt" id="tt" role="status"></div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const SVGNS = 'http://www.w3.org/2000/svg';
let CORPUS = D.corpora[0];

const el = (n, a = {}, parent = null) => {
  const e = document.createElementNS(SVGNS, n);
  for (const k in a) e.setAttribute(k, a[k]);
  if (parent) parent.appendChild(e);
  return e;
};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const clear = id => { const s = document.getElementById(id); while (s.firstChild) s.removeChild(s.firstChild); return s; };
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : x.toFixed(d);
const sign = (x, d = 2) => (x >= 0 ? '+' : '') + fmt(x, d);

const tt = document.getElementById('tt');
function hover(node, html) {
  node.style.cursor = 'default';
  node.addEventListener('pointerenter', e => { tt.innerHTML = html; tt.style.opacity = 1; move(e); });
  node.addEventListener('pointermove', move);
  node.addEventListener('pointerleave', () => { tt.style.opacity = 0; });
  function move(e) {
    const p = 14, w = tt.offsetWidth, h = tt.offsetHeight;
    tt.style.left = Math.min(e.clientX + p, innerWidth - w - 8) + 'px';
    tt.style.top = Math.max(8, Math.min(e.clientY + p, innerHeight - h - 8)) + 'px';
  }
}
function table(host, head, rows) {
  const t = document.createElement('table');
  t.innerHTML = '<thead><tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('') + '</tbody>';
  host.replaceChildren(t);
}
function legend(host, items) {
  document.getElementById(host).innerHTML = items.map(([c, l]) =>
    `<span><i class="sw" style="background:${c}"></i>${l}</span>`).join('');
}
const colourOf = m => m === 'READER' ? css('--reader') : m === 'BiTE' ? css('--bite')
  : m === 'Compact' ? css('--compact') : css('--other');

/* ---- headline tiles ---------------------------------------------------- */
function tiles() {
  const host = document.getElementById('tiles');
  host.innerHTML = D.corpora.map(ds => {
    const m = D.models['READER'][ds];
    if (!m) return '';
    const strongest = D.strongest[ds], d = (D.reader_minus[ds] || {})[strongest];
    const delta = d ? `${sign(d.mean)} pp vs ${strongest} [${sign(d.ci95[0])}, ${sign(d.ci95[1])}]`
                    : 'baseline pending';
    const cur = ds === CORPUS ? `border-color:${css('--reader')}` : '';
    return `<div class="tile" style="${cur}"><div class="k">${D.titles[ds]}</div>
      <div class="v">${fmt(m.mean)}%</div><div class="d">${delta}</div>
      <div class="d" style="color:var(--muted)">rank ${D.rank[ds] || '—'} · ${Object.keys(m.subjects).length} subjects × ${m.seeds} seeds</div></div>`;
  }).join('');
}

/* ---- accuracy vs parameters ------------------------------------------- */
function landscape() {
  const svg = clear('landscape'), W = 900, H = 380, L = 56, R = 18, T = 16, B = 44;
  const pts = Object.entries(D.models).filter(([, v]) => v[CORPUS])
    .map(([name, v]) => ({ name, ...v[CORPUS] }));
  if (!pts.length) return;
  const xs = pts.map(p => Math.log10(p.params)), ys = pts.map(p => p.mean);
  const x0 = Math.min(...xs) - .12, x1 = Math.max(...xs) + .12;
  const y0 = Math.min(...ys) - 2.5, y1 = Math.max(...ys) + 2.5;
  const X = v => L + (Math.log10(v) - x0) / (x1 - x0) * (W - L - R);
  const Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B);

  const g = el('g', {}, svg);
  for (let t = Math.ceil(y0 / 5) * 5; t <= y1; t += 5) {
    el('line', { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: 'gridline' }, g);
    el('text', { x: L - 8, y: Y(t) + 4, 'text-anchor': 'end' }, g).textContent = t;
  }
  for (let e = Math.ceil(x0); e <= x1; e++) {
    const v = Math.pow(10, e);
    el('line', { x1: X(v), x2: X(v), y1: T, y2: H - B, class: 'gridline' }, g);
    el('text', { x: X(v), y: H - B + 18, 'text-anchor': 'middle' }, g).textContent =
      v >= 1e6 ? (v / 1e6) + 'M' : v >= 1e3 ? (v / 1e3) + 'K' : v;
  }
  el('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle' }, svg).textContent = 'parameters (log scale)';
  el('text', { x: 14, y: (T + H - B) / 2, 'text-anchor': 'middle',
    transform: `rotate(-90 14 ${(T + H - B) / 2})` }, svg).textContent = 'accuracy (%)';

  pts.sort((a, b) => (a.name === 'READER') - (b.name === 'READER'));
  for (const p of pts) {
    const named = ['READER', 'BiTE', 'Compact'].includes(p.name), c = colourOf(p.name);
    const node = el('circle', { cx: X(p.params), cy: Y(p.mean), r: named ? 7 : 5,
      fill: named ? c : 'none', stroke: named ? css('--surface') : c,
      'stroke-width': named ? 2 : 1.4 }, svg);
    el('text', { x: X(p.params), y: Y(p.mean) - (named ? 13 : 10), 'text-anchor': 'middle',
      'font-size': named ? 12 : 10, 'font-weight': named ? 650 : 400,
      fill: named ? c : css('--muted') }, svg).textContent = p.name;
    const pub = D.published[CORPUS][p.name];
    hover(node, `<b>${p.name}</b><br>${fmt(p.mean)}% · ${p.params.toLocaleString()} params`
      + (pub ? `<br>published ${fmt(pub)}%` : '') + `<br>${p.seeds} seeds`);
  }
  table(document.getElementById('landTable'), ['model', 'accuracy (%)', 'parameters', 'published'],
    pts.slice().sort((a, b) => b.mean - a.mean).map(p =>
      [p.name, fmt(p.mean), p.params.toLocaleString(), fmt(D.published[CORPUS][p.name])]));
  legend('landLegend', [[css('--reader'), 'READER'], [css('--bite'), 'BiTE'],
    [css('--compact'), 'Compact'], [css('--other'), "BiTE's released baselines, re-run here"]]);
}

/* ---- per-subject dumbbell --------------------------------------------- */
function dumbbell() {
  const svg = clear('dumbbell'), W = 900, L = 52, R = 70, T = 12, B = 42;
  const A = D.models['READER'][CORPUS], Bm = D.models['BiTE'][CORPUS];
  if (!A || !Bm) return;
  const subs = Object.keys(A.subjects).filter(s => s in Bm.subjects);
  const H = Math.max(180, T + B + subs.length * 26);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const vals = subs.flatMap(s => [A.subjects[s], Bm.subjects[s]]);
  const v0 = Math.max(0, Math.min(...vals) - 5), v1 = Math.min(100, Math.max(...vals) + 5);
  const X = v => L + (v - v0) / (v1 - v0) * (W - L - R);
  const Yi = i => T + 13 + i * 26;

  for (let t = Math.ceil(v0 / 10) * 10; t <= v1; t += 10) {
    el('line', { x1: X(t), x2: X(t), y1: T, y2: H - B, class: 'gridline' }, svg);
    el('text', { x: X(t), y: H - B + 18, 'text-anchor': 'middle' }, svg).textContent = t;
  }
  el('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle' }, svg).textContent = 'accuracy (%)';
  subs.forEach((s, i) => {
    const a = A.subjects[s], b = Bm.subjects[s], y = Yi(i), win = a >= b;
    el('text', { x: L - 10, y: y + 4, 'text-anchor': 'end' }, svg).textContent = 'S' + s;
    el('line', { x1: X(b), x2: X(a), y1: y, y2: y, stroke: win ? css('--reader') : css('--bite'),
      'stroke-width': 3, 'stroke-linecap': 'round', opacity: .45 }, svg);
    for (const [v, c, nm] of [[b, css('--bite'), 'BiTE'], [a, css('--reader'), 'READER']]) {
      const n = el('circle', { cx: X(v), cy: y, r: 6, fill: c, stroke: css('--surface'), 'stroke-width': 2 }, svg);
      hover(n, `<b>Subject ${s}</b><br>${nm} ${fmt(v)}%<br>READER − BiTE ${sign(a - b)} pp`);
    }
    el('text', { x: W - R + 10, y: y + 4, fill: win ? css('--reader') : css('--bite') },
      svg).textContent = sign(a - b, 1);
  });
  legend('dumbLegend', [[css('--reader'), 'READER (and subjects it wins)'],
    [css('--bite'), 'BiTE (and subjects it wins)']]);
  table(document.getElementById('dumbTable'), ['subject', 'READER', 'BiTE', 'difference'],
    subs.map(s => ['S' + s, fmt(A.subjects[s]), fmt(Bm.subjects[s]), sign(A.subjects[s] - Bm.subjects[s])]));
}

/* ---- anytime ----------------------------------------------------------- */
function anytime() {
  const svg = clear('anytime'), W = 900, H = 360, L = 56, R = 20, T = 16, B = 46;
  const a = D.anytime[CORPUS];
  const note = document.getElementById('anyNote');
  if (!a) {
    el('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', fill: css('--muted') }, svg)
      .textContent = 'No anytime bank was trained for ' + D.titles[CORPUS] + '.';
    note.textContent = ''; legend('anyLegend', []); return;
  }
  const ks = Object.keys(a.deadlines).map(Number).sort((p, q) => p - q);
  const series = { reader: ks.map(k => a.deadlines[k].reader) };
  for (const fam of ['bite', 'compact'])
    series[fam] = ks.map(k => a.deadlines[k].banks[fam]);
  const vals = Object.values(series).flat().filter(v => v != null);
  const y0 = Math.min(...vals) - 4, y1 = Math.max(...vals) + 4;
  const X = v => L + (v - ks[0]) / (ks[ks.length - 1] - ks[0] || 1) * (W - L - R);
  const Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B);

  for (let t = Math.ceil(y0 / 5) * 5; t <= y1; t += 5) {
    el('line', { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: 'gridline' }, svg);
    el('text', { x: L - 8, y: Y(t) + 4, 'text-anchor': 'end' }, svg).textContent = t;
  }
  ks.forEach(k => el('text', { x: X(k), y: H - B + 18, 'text-anchor': 'middle' }, svg).textContent = k + ' s');
  el('text', { x: (L + W - R) / 2, y: H - 10, 'text-anchor': 'middle' }, svg).textContent = 'decision deadline';
  el('text', { x: 14, y: (T + H - B) / 2, 'text-anchor': 'middle',
    transform: `rotate(-90 14 ${(T + H - B) / 2})` }, svg).textContent = 'accuracy (%)';

  const spec = [['compact', css('--compact'), 'Compact specialists (one model per deadline)'],
                ['bite', css('--bite'), 'BiTE specialists (one model per deadline)'],
                ['reader', css('--reader'), 'READER (one model, read at every deadline)']];
  for (const [key, colour, label] of spec) {
    const pts = ks.map((k, i) => [X(k), series[key][i]]).filter(p => p[1] != null);
    if (pts.length < 1) continue;
    el('path', { d: pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + Y(p[1])).join(' '), fill: 'none',
      stroke: colour, 'stroke-width': key === 'reader' ? 3 : 2,
      'stroke-dasharray': key === 'reader' ? '' : '5 4' }, svg);
    pts.forEach((p, i) => {
      const n = el('circle', { cx: p[0], cy: Y(p[1]), r: key === 'reader' ? 6 : 5, fill: colour,
        stroke: css('--surface'), 'stroke-width': 2 }, svg);
      const dl = a.deadlines[ks[i]];
      hover(n, `<b>${label.split(' (')[0]}</b> at ${ks[i]} s<br>${fmt(p[1])}%`
        + (key === 'reader' ? `<br>vs strongest bank (${dl.strongest}) ${sign(dl.delta)} pp`
          + (dl.ci95 ? `<br>95% CI [${sign(dl.ci95[0])}, ${sign(dl.ci95[1])}]` : '') : ''));
    });
  }
  legend('anyLegend', spec.map(([, c, l]) => [c, l]).reverse());
  note.textContent = `READER arm: ${a.arm}. Solid line = one model. Dashed = a separate model trained `
    + `from scratch for each deadline. Deltas at the endpoint are the within-subject table's.`;
}

/* ---- complementarity + confusion --------------------------------------- */
function errors() {
  const e = D.errors[CORPUS];
  const svg = clear('comp'), W = 900;
  const parts = [['both_correct', 'both right', css('--fill-soft')],
                 ['only_a', 'only READER', css('--reader')],
                 ['only_b', 'only BiTE', css('--bite')],
                 ['both_wrong', 'both wrong', css('--fill-none')]];
  if (e && e.complementarity) {
    const p = e.complementarity, L = 10, R = 150, y = 40, h = 46;
    svg.setAttribute('viewBox', '0 0 900 130');
    let x = L;
    for (const [k, label, colour] of parts) {
      const w = p[k] / 100 * (W - L - R);
      const n = el('rect', { x, y, width: Math.max(w - 2, 0), height: h, rx: 4, fill: colour }, svg);
      hover(n, `<b>${label}</b><br>${fmt(p[k])}% of test trials`);
      if (w > 46) el('text', { x: x + w / 2 - 1, y: y + h / 2 + 4, 'text-anchor': 'middle',
        fill: [css('--reader'), css('--bite')].includes(colour) ? '#fff' : css('--ink') },
        svg).textContent = fmt(p[k], 1);
      x += w;
    }
    el('text', { x: W - R + 14, y: y + h / 2 + 4, 'font-weight': 600, fill: css('--ink') },
      svg).textContent = `oracle ${fmt(p.oracle, 1)}%`;
    el('text', { x: L, y: 22, fill: css('--muted') }, svg).textContent =
      `${D.titles[CORPUS]} — every test trial, READER against BiTE`;
    legend('compLegend', parts.map(([, l, c]) => [c, l]));
  } else {
    svg.setAttribute('viewBox', '0 0 900 60');
    el('text', { x: 10, y: 30, fill: css('--muted') }, svg).textContent = 'Not available for this corpus.';
    legend('compLegend', []);
  }

  const cs = clear('conf');
  document.getElementById('confNote').textContent = '';
  if (!e) { cs.setAttribute('viewBox', '0 0 900 40'); return; }
  const n = e.n_classes, names = e.class_names || Array.from({ length: n }, (_, i) => String(i + 1));
  const cell = Math.min(46, 340 / n), L2 = 120, T2 = 74, size = cell * n;
  cs.setAttribute('viewBox', `0 0 900 ${T2 + size + 40}`);
  const M = e.confusion['reader'];
  const ramp = [css('--seq-1'), css('--seq-2'), css('--seq-3'), css('--seq-4'), css('--seq-5')];
  const colour = v => { const t = Math.max(0, Math.min(1, v / 100)) * (ramp.length - 1);
    return ramp[Math.round(t)]; };
  for (let i = 0; i < n; i++) {
    el('text', { x: L2 - 10, y: T2 + i * cell + cell / 2 + 4, 'text-anchor': 'end',
      'font-size': n > 6 ? 9 : 11 }, cs).textContent = names[i];
    el('text', { x: L2 + i * cell + cell / 2, y: T2 - 10, 'text-anchor': 'middle',
      'font-size': n > 6 ? 9 : 11 }, cs).textContent = names[i];
    for (let j = 0; j < n; j++) {
      const v = 100 * M[i][j];
      const r = el('rect', { x: L2 + j * cell + 1, y: T2 + i * cell + 1, width: cell - 2,
        height: cell - 2, rx: 3, fill: colour(v) }, cs);
      hover(r, `true <b>${names[i]}</b> → predicted <b>${names[j]}</b><br>${fmt(v, 1)}%`);
      if (v >= 0.5) el('text', { x: L2 + j * cell + cell / 2, y: T2 + i * cell + cell / 2 + 4,
        'text-anchor': 'middle', 'font-size': n > 6 ? 8 : 10,
        fill: v > 55 ? '#fff' : css('--ink') }, cs).textContent = Math.round(v);
    }
  }
  el('text', { x: L2, y: 28, fill: css('--muted') }, cs).textContent = 'predicted class →';
  el('text', { x: 14, y: T2 + size / 2, 'text-anchor': 'middle', fill: css('--muted'),
    transform: `rotate(-90 14 ${T2 + size / 2})` }, cs).textContent = 'true class';
  document.getElementById('confNote').textContent = e.class_names
    ? "Class names are BiTE's own preprocessing."
    : 'SD-SSVEP stimulus classes are numbered: the released labels carry no frequency table.';
}

/* ---- model x subject heatmap ------------------------------------------- */
function heat() {
  const svg = clear('heat'), W = 900, L = 132, T = 30;
  const models = Object.entries(D.models).filter(([, v]) => v[CORPUS])
    .sort((a, b) => b[1][CORPUS].mean - a[1][CORPUS].mean);
  if (!models.length) return;
  const subs = Object.keys(models[0][1][CORPUS].subjects);
  const cw = Math.min(52, (W - L - 20) / subs.length), rh = 24;
  const H = T + models.length * rh + 40;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const all = models.flatMap(([, v]) => Object.values(v[CORPUS].subjects));
  const lo = Math.min(...all), hi = Math.max(...all);
  const ramp = [css('--seq-1'), css('--seq-2'), css('--seq-3'), css('--seq-4'), css('--seq-5')];
  subs.forEach((s, j) => el('text', { x: L + j * cw + cw / 2, y: T - 10, 'text-anchor': 'middle',
    'font-size': 10 }, svg).textContent = s);
  el('text', { x: L - 10, y: T - 10, 'text-anchor': 'end', 'font-size': 10, fill: css('--muted') },
    svg).textContent = 'subject';
  models.forEach(([name, v], i) => {
    const isR = name === 'READER';
    el('text', { x: L - 10, y: T + i * rh + rh / 2 + 4, 'text-anchor': 'end', 'font-size': 11,
      'font-weight': isR ? 650 : 400, fill: isR ? css('--reader') : css('--ink-2') },
      svg).textContent = name;
    subs.forEach((s, j) => {
      const a = v[CORPUS].subjects[s];
      if (a == null) return;
      const t = (a - lo) / Math.max(hi - lo, 1e-9);
      const r = el('rect', { x: L + j * cw + 1, y: T + i * rh + 1, width: cw - 2, height: rh - 2,
        rx: 3, fill: ramp[Math.round(t * (ramp.length - 1))] }, svg);
      hover(r, `<b>${name}</b> · subject ${s}<br>${fmt(a)}%`);
      if (cw > 30) el('text', { x: L + j * cw + cw / 2, y: T + i * rh + rh / 2 + 4,
        'text-anchor': 'middle', 'font-size': 9, fill: t > .55 ? '#fff' : css('--ink') },
        svg).textContent = Math.round(a);
    });
  });
}

/* ---- wiring ------------------------------------------------------------ */
function renderAll() { tiles(); landscape(); dumbbell(); anytime(); errors(); heat(); }

document.getElementById('corpusChips').innerHTML = D.corpora.map(ds =>
  `<button class="chip" data-ds="${ds}" aria-pressed="${ds === CORPUS}">${D.titles[ds]}</button>`).join(' ');
document.getElementById('corpusChips').addEventListener('click', e => {
  const b = e.target.closest('button[data-ds]');
  if (!b) return;
  CORPUS = b.dataset.ds;
  document.querySelectorAll('#corpusChips .chip').forEach(c =>
    c.setAttribute('aria-pressed', c.dataset.ds === CORPUS));
  renderAll();
});

const themeBtn = document.getElementById('themeBtn');
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  themeBtn.textContent = t === 'dark' ? 'Light' : 'Dark';
  themeBtn.setAttribute('aria-pressed', t === 'dark');
  renderAll();
}
let theme = 'light';
try { theme = localStorage.getItem('reader-theme')
  || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch (_) {}
themeBtn.addEventListener('click', () => {
  theme = theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('reader-theme', theme); } catch (_) {}
  applyTheme(theme);
});

document.getElementById('meta').textContent =
  `${D.n_runs.toLocaleString()} runs of record · rebuilt ${D.built} · three seeds per cell`;
applyTheme(theme);
</script>
</body>
</html>
"""


def embed(payload):
    """JSON for a <script type="application/json"> element.

    A literal `</script` anywhere in the blob would close the element early and break the page. `\\/` is a
    legal JSON string escape for `/` and parses back identically, so escaping every `</` is free. No result
    string contains one today -- this keeps that from becoming a silent condition of the page working.
    """
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def main():
    payload = build_payload()
    blob = embed(payload)
    out = RES / "dashboard.html"
    out.write_text(HTML.replace("__DATA__", blob))
    print(f"-> {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload['models'])} models, {payload['n_runs']} runs)")


if __name__ == "__main__":
    main()
