"""READER: the compact decoder with a prefix-bidirectional reading head.

Gap this answers
----------------
Bidirectional context in EEG decoders is bought at the cost of deployability. BiTE's BiTCN obtains
its backward context by reversing the COMPLETE trial: its backward branch's last step corresponds
to the trial's first moment, and its STFT front end uses ``center=True``, reading 128 ms into the
future. No output exists until the trial has ended, so a deployed system must train and hold a
separate model for every decision deadline.

Mechanism
---------
A second causal TCN reads the OBSERVED PREFIX in reverse. At deadline t it consumes tokens
t-1, t-2, ..., 0 and its output is fused with the forward reading by a convex per-feature gate.
Reversed over the trial is non-causal; reversed over the prefix is not. One model therefore emits
a legal bidirectional decision at every 128 ms token boundary.

The cost is quadratic: the backward branch is recomputed per prefix, so a full anytime curve is
O(T^2) in the token count. For a single fixed deadline it is O(T) like any causal model.

Ablations reported in the paper are reached through this same class:
  reader="bidir"    the full model
  reader="forward"  the forward-only parent (the mechanism removed)
and, on trained weights, the inference-time route interventions in ``diagnostic_report``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from reader.data import SPECS
from reader.models.compact import CausalTCN, CompactDecoder


class ReaderDecoder(CompactDecoder):
    def __init__(self, dataset: str, reader: str = "bidir", input_mode: str = "bite", **kw):
        spec = SPECS[dataset]
        super().__init__(spec["channels"], spec["classes"], pool=spec["pool"], **kw)
        self.dataset, self.input_mode, self.reader_mode = dataset, input_mode, reader
        embedding = self.proj.out_features
        if reader == "bidir":
            # Construction consumes RNG; isolate it so the parameters shared with the forward-only
            # parent are bit-identical at the same seed and the two arms are exactly paired.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(918273)
                self.backward_tcn = CausalTCN(embedding, depth=len(self.tcn.blocks), dropout=self.drop.p)
            self.backward_gate = nn.Parameter(torch.zeros(embedding))
        elif reader != "forward":
            raise ValueError(reader)
        self._intervention: dict = {}

    # ------------------------------------------------------------------ graph
    def _readout(self, h):
        """Side-effect-free readout. ``classify`` renormalises the head in place, so it must not be
        called inside the forward pass for diagnostics."""
        return F.linear(h, self.head.weight, self.head.bias)

    def _gate(self):
        g = torch.sigmoid(self.backward_gate)
        if "gate" in self._intervention:
            g = torch.full_like(g, self._intervention["gate"])
        return g

    def prefix_states(self, positioned, forward_seq):
        """State at every deadline. Deadline t reads tokens 0..t-1 ONLY -- the trial end is never
        visible, which is the whole point of the mechanism."""
        g = self._gate()
        rows = []
        for t in range(1, positioned.shape[1] + 1):
            b = self.backward_tcn(positioned[:, :t].flip(1))[:, -1]
            rows.append((1 - g) * forward_seq[:, t - 1] + g * b)
        self._last_backward = b
        return torch.stack(rows, 1)

    def forward(self, x):
        tokens = self.tokens(self.spatial_features(self.carriers(x)))
        positioned = tokens + self.pos[:, :tokens.shape[1]]
        forward_seq = self.tcn(positioned)
        if self.reader_mode == "forward":
            sequence = self.classify(forward_seq)
            return {"logits": sequence[:, -1], "sequence": sequence, "route_logits": None}
        states = self.prefix_states(positioned, forward_seq)
        sequence = self.classify(states)
        with torch.no_grad():
            route = {"forward": self._readout(forward_seq[:, -1]),
                     "backward": self._readout(self._last_backward)}
        return {"logits": sequence[:, -1], "sequence": sequence, "route_logits": route}

    @torch.no_grad()
    def anytime_logits(self, x):
        """Logits at every 128 ms deadline. Verified in tests/test_deployment_causality.py to equal
        what the model produces when it is only ever GIVEN that many samples."""
        was = self.training
        self.eval()
        tokens = self.tokens(self.spatial_features(self.carriers(x)))
        positioned = tokens + self.pos[:, :tokens.shape[1]]
        forward_seq = self.tcn(positioned)
        seq = (self.classify(forward_seq) if self.reader_mode == "forward"
               else self.classify(self.prefix_states(positioned, forward_seq)))
        self.train(was)
        return seq

    # ------------------------------------------------------------------ telemetry
    @torch.no_grad()
    def epoch_telemetry(self):
        if self.reader_mode != "bidir":
            return {}
        g = torch.sigmoid(self.backward_gate)
        return {"gate": {"mean": g.mean().item(), "std": g.std().item(),
                         "min": g.min().item(), "max": g.max().item()}}

    @torch.no_grad()
    def diagnostic_report(self, tr_tensors, te_tensors, ytr, yte, device):
        """Same-weight inference interventions: does the mechanism carry the decision?"""
        def acc(tensors, y):
            was = self.training
            self.eval()
            logits, route = [], {"forward": [], "backward": []}
            for s in range(0, len(y), 64):
                out = self(*[t[s:s + 64].to(device) for t in tensors])
                logits.append(out["logits"].cpu())
                if out.get("route_logits"):
                    for k in route:
                        route[k].append(out["route_logits"][k].cpu())
            self.train(was)
            logits = torch.cat(logits)
            res = {"acc": (logits.argmax(1) == y).float().mean().item()}
            if route["forward"]:
                f, b = torch.cat(route["forward"]), torch.cat(route["backward"])
                res["forward_only_acc"] = (f.argmax(1) == y).float().mean().item()
                res["backward_only_acc"] = (b.argmax(1) == y).float().mean().item()
            return res

        report = {"test": acc(te_tensors, yte), "train": acc(tr_tensors, ytr)}
        interventions = {}
        if self.reader_mode == "bidir":
            for label, value in (("gate_forward_only", 0.0), ("gate_backward_only", 1.0), ("gate_half", 0.5)):
                self._intervention = {"gate": value}
                interventions[label] = {"test": acc(te_tensors, yte)["acc"],
                                        "train": acc(tr_tensors, ytr)["acc"]}
        self._intervention = {}
        report["interventions"] = interventions
        anytime = [self.anytime_logits(te_tensors[0][s:s + 64].to(device)).cpu()
                   for s in range(0, len(yte), 64)]
        seq = torch.cat(anytime)
        report["anytime_test_token_acc"] = (seq.argmax(-1) == yte[:, None]).float().mean(0).tolist()
        return report


ARMS = {
    "reader":  dict(reader="bidir",   input_mode="bite"),   # the proposed model
    "compact": dict(reader="forward", input_mode="bite"),   # its forward-only parent (ablation)
}


def build(name, dataset, **options):
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; have {sorted(ARMS)}")
    settings = dict(ARMS[name]); settings.update(options)
    model = ReaderDecoder(dataset, **settings)
    keys = {"carriers": ("tbn", "channels_time"), "spatial": ("act", "channels_time"),
            "tokens": ("proj", "tokens"),
            "reader": ("head", "vector" if settings["reader"] == "bidir" else "tokens", "input")}
    return model, keys, {"spatial.weight": 1.0, "head.weight": .25}, False
