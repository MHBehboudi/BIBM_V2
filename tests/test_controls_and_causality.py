"""FF-Control / Compact-Mean admission gates and STRENGTHENED causality checks.

Why the old causality tests were insufficient (co-author, verified here): every residual TCN block
zero-initialises its second convolution, so at init each TCN is EXACTLY the identity (in eval mode
BatchNorm maps 0 to 0 with default running stats). The reverse branch's last output is then token 0
whatever it reads, so an implementation that reversed the COMPLETE trial at every deadline would
produce the same numbers as the prefix-restricted one and pass. The tests below therefore run on an
ACTIVATED configuration (second convolutions, BatchNorm statistics and gate all randomised), and
include a deliberately future-reading mutant that the procedure must catch, plus a test that pins
the old procedure's blindness to it so the reason for the change stays documented.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS  # noqa: E402
from reader.models import reader as reader_module  # noqa: E402

CAUSAL_ARMS = ["compact", "reader", "ff_control", "compact_mean"]


def _build(arm, ds, seed=3):
    torch.manual_seed(seed)
    model, keys, caps, needs_spectral = reader_module.build(arm, ds)
    assert not needs_spectral
    return model


@torch.no_grad()
def activate(model, seed=0):
    """Move every mechanism away from its degenerate init: random second convolutions, random BN
    running statistics and affine parameters, gate off 0.5."""
    g = torch.Generator().manual_seed(seed)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d) and name.endswith("conv2"):
            m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
            m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
            m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
    if hasattr(model, "backward_gate"):
        model.backward_gate.copy_(torch.randn(model.backward_gate.shape, generator=g))
    return model


class LeakyReader(reader_module.ReaderDecoder):
    """A DELIBERATELY WRONG implementation: the reverse branch reads the COMPLETE sequence at every
    deadline. The causality procedure must detect it."""

    def prefix_states(self, positioned, forward_seq):
        g = self._gate()
        b = self.backward_tcn(positioned.flip(1))[:, -1]          # full-trial reversal: reads the future
        self._last_backward = b
        return (1 - g) * forward_seq + g * b[:, None, :]


def _leaky(ds, seed=3):
    torch.manual_seed(seed)
    return LeakyReader(ds, **reader_module.ARMS["reader"])


@torch.no_grad()
def causality_discrepancies(model, ds, trials=3, seed=1):
    """Max |logit| discrepancy for (a) truncated input vs full-trial curve at every pooling boundary and
    (b) the curve at boundary u when every sample after u is replaced by large noise."""
    spec = SPECS[ds]
    pool = spec["pool"]
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((trials, spec["channels"], spec["samples"]), generator=g)
    curve = model.anytime_logits(x)
    n = curve.shape[1]
    trunc, future = 0.0, 0.0
    for u in range(1, n + 1):
        if u * pool > spec["samples"]:
            break
        endpoint = model(x[..., : u * pool])["logits"]
        trunc = max(trunc, (endpoint - curve[:, u - 1]).abs().max().item())
    for u in sorted({1, 2, n // 3, n // 2, n - 1}):
        if u < 1 or u * pool >= spec["samples"]:
            continue
        x2 = x.clone()
        x2[..., u * pool:] = 1e3 * torch.randn(x2[..., u * pool:].shape, generator=g)
        future = max(future, (model.anytime_logits(x2)[:, :u] - curve[:, :u]).abs().max().item())
    return trunc, future


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_ff_control_is_capacity_and_init_matched_to_reader(ds):
    reader, ff = _build("reader", ds), _build("ff_control", ds)
    rp = dict(reader.named_parameters())
    fp = {n.replace("second_tcn.", "backward_tcn."): p for n, p in ff.named_parameters()}
    assert set(rp) == set(fp)
    for name, p in rp.items():
        assert torch.equal(p, fp[name]), name
    assert sum(p.numel() for p in rp.values()) == sum(p.numel() for p in fp.values())


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_compact_mean_is_parameter_and_init_matched_to_compact(ds):
    compact = _build("compact", ds)
    mean = _build("compact_mean", ds)
    cp, mp = dict(compact.named_parameters()), dict(mean.named_parameters())
    assert set(cp) == set(mp)
    for name, p in cp.items():
        assert torch.equal(p, mp[name]), name


def test_ff_reads_original_order_and_reader_reads_reversed_prefix():
    ds, spec = "2b", SPECS["2b"]
    ff, reader = activate(_build("ff_control", ds)), activate(_build("reader", ds))
    reader.load_state_dict({k.replace("second_tcn.", "backward_tcn."): v for k, v in ff.state_dict().items()})
    ff.eval(); reader.eval()
    x = torch.randn(4, spec["channels"], spec["samples"])
    with torch.no_grad():
        a, b = ff(x)["logits"], reader(x)["logits"]
    assert (a - b).abs().max() > 1e-3, "identical weights must give different outputs when only direction differs"


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_compact_mean_readout_is_the_running_mean(ds):
    spec = SPECS[ds]
    model = activate(_build("compact_mean", ds)).eval()
    x = torch.randn(2, spec["channels"], spec["samples"])
    with torch.no_grad():
        got = model(x)["logits"]            # first: classify() applies the head's max-norm cap in place
        tokens = model.tokens(model.spatial_features(model.carriers(x)))
        f = model.tcn(tokens + model.pos[:, :tokens.shape[1]])
        expected = torch.nn.functional.linear(f.mean(1), model.head.weight, model.head.bias)
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()


@pytest.mark.parametrize("arm", CAUSAL_ARMS)
@pytest.mark.parametrize("ds", sorted(SPECS))
def test_activated_models_are_prefix_causal(arm, ds):
    model = activate(_build(arm, ds))
    trunc, future = causality_discrepancies(model, ds)
    assert trunc < 1e-4, f"{arm} {ds}: truncated input differs from the curve by {trunc:.2e}"
    assert future == 0.0, f"{arm} {ds}: future samples moved an earlier decision by {future:.2e}"


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_procedure_detects_a_future_reading_reverse_branch(ds):
    leaky = activate(_leaky(ds))
    trunc, future = causality_discrepancies(leaky, ds)
    assert trunc > 1e-3 and future > 1e-3, f"mutant NOT detected: trunc {trunc:.2e} future {future:.2e}"


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_identity_init_is_blind_to_the_mutant(ds):
    """Documents WHY activation is required: at the zero-initialised residual init, the mutant is
    indistinguishable from the correct model. If this ever fails the init changed; re-read the tests."""
    correct, leaky = _build("reader", ds).eval(), _leaky(ds).eval()
    leaky.load_state_dict(correct.state_dict())
    spec = SPECS[ds]
    x = torch.randn(3, spec["channels"], spec["samples"])
    with torch.no_grad():
        assert torch.allclose(correct.anytime_logits(x), leaky.anytime_logits(x), atol=1e-6)
    trunc, future = causality_discrepancies(leaky, ds)
    assert trunc < 1e-4 and future == 0.0


@pytest.mark.parametrize("arm", CAUSAL_ARMS)
def test_eval_uses_running_statistics_not_the_batch(arm):
    ds, spec = "2a", SPECS["2a"]
    model = activate(_build(arm, ds)).eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            assert m.track_running_stats and m.running_mean is not None, m
    x = torch.randn(6, spec["channels"], spec["samples"])
    with torch.no_grad():
        alone = model(x[:1])["logits"]
        batched = model(x)["logits"][:1]
    assert torch.allclose(alone, batched, atol=1e-5), (alone - batched).abs().max()
