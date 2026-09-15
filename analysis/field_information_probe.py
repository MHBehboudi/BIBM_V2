#!/usr/bin/env python3
"""Model-free: does the DIFFERENTIAL FIELD representation carry class information that the
reference-matched channel representation does not already have?

The co-author's RIFT field branch is built from inter-electrode differences d_e = x_i - x_j on a
k-NN electrode graph. d_e is a LINEAR map of the channels, so any linear spatial filter can already
form it; the only part that is not absorbable by a spatial filter is a NONLINEAR statistic of d_e.
This probe measures exactly that, with the same ridge protocol used for every other input probe:

  chan       log band power of CAR'd channels                      (what our reader already sees)
  edge       log band power of the k-NN inter-electrode differences (the field's non-absorbable part)
  both       concatenation
  edge_res   edge features after the channel features are ridge-regressed OUT of them (fit on train)

If `edge_res` is at chance, the field branch adds no information and no amount of coupling can help.
If `both` > `chan`, there is headroom for a field pathway and the size of the gap bounds it.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS, electrode_positions, raw_roles  # noqa: E402
from reader.diagnostics import complex_spectrum, filterbank_logpower, ridge_probes  # noqa: E402

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]


def car(x):
    return x - x.mean(1, keepdims=True)


def knn_edges(pos, k=3):
    """Symmetric k-NN electrode graph, the co-author's k_nn=3, plus MST edges for connectivity."""
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    n = len(pos)
    edges = set()
    for i in range(n):
        for j in np.argsort(d2[i])[1:k + 1]:
            edges.add((min(i, int(j)), max(i, int(j))))
    # connectivity: union-find over the k-NN edges, then add shortest missing links
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in edges:
        parent[find(i)] = find(j)
    order = np.dstack(np.unravel_index(np.argsort(d2, axis=None), d2.shape))[0]
    for i, j in order:
        if find(int(i)) != find(int(j)):
            edges.add((min(int(i), int(j)), max(int(i), int(j))))
            parent[find(int(i))] = find(int(j))
    return sorted(edges)


def edge_signals(x, edges):
    return np.stack([x[:, i, :] - x[:, j, :] for i, j in edges], 1)


def residualize(ftr, fte, gtr, gte):
    """Remove from f everything a linear map of g can explain; the map is fit on TRAIN only."""
    mu, sd = gtr.mean(0, keepdims=True), gtr.std(0, keepdims=True) + 1e-8
    r = Ridge(alpha=10.0).fit((gtr - mu) / sd, ftr)
    return ftr - r.predict((gtr - mu) / sd), fte - r.predict((gte - mu) / sd)


def main():
    out = {}
    for ds in sys.argv[1:] or ["2a", "2b", "hgd", "sdssvep"]:
        spec = SPECS[ds]
        pos = electrode_positions(ds).numpy().astype(np.float64)
        edges = knn_edges(pos) if ds != "2b" else [(0, 1), (1, 2), (0, 2)]
        rows = []
        for sub in range(1, spec["subjects"] + 1):
            xtr, ytr, xte, yte = raw_roles(ds, sub)
            ctr, cte = (xtr, xte) if ds == "2b" else (car(xtr), car(xte))
            etr, ete = edge_signals(xtr, edges), edge_signals(xte, edges)
            if ds == "sdssvep":
                feat = lambda a: complex_spectrum(a, spec["fs"], 8, 64)
            else:
                feat = lambda a: filterbank_logpower(a, spec["fs"], BANDS)
            fc_tr, fc_te = feat(ctr), feat(cte)
            fe_tr, fe_te = feat(etr), feat(ete)
            rtr, rte = residualize(fe_tr, fe_te, fc_tr, fc_te)
            row = {
                "subject": sub, "n_edges": len(edges), "chance": 1.0 / len(np.unique(ytr)),
                "chan": ridge_probes(fc_tr, ytr, fc_te, yte),
                "edge": ridge_probes(fe_tr, ytr, fe_te, yte),
                "both": ridge_probes(np.hstack([fc_tr, fe_tr]), ytr, np.hstack([fc_te, fe_te]), yte),
                "edge_res": ridge_probes(rtr, ytr, rte, yte),
            }
            rows.append(row)
            print(ds, sub, {k: round(row[k]["probe_train_to_test"], 3)
                            for k in ("chan", "edge", "both", "edge_res")}, flush=True)
        out[ds] = {
            "n_edges": len(edges), "chance": rows[0]["chance"],
            "means": {k: float(np.mean([r[k]["probe_train_to_test"] for r in rows]))
                      for k in ("chan", "edge", "both", "edge_res")},
            "per_subject": {k: [r[k]["probe_train_to_test"] for r in rows]
                            for k in ("chan", "edge", "both", "edge_res")},
        }
        m = out[ds]["means"]
        print(f"== {ds}: chance {out[ds]['chance']:.3f} | chan {m['chan']:.4f} edge {m['edge']:.4f} "
              f"both {m['both']:.4f} (both-chan {m['both']-m['chan']:+.4f}) edge_res {m['edge_res']:.4f}", flush=True)
    Path(__file__).with_name("field_information_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
