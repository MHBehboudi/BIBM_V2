"""Is the model TRULY causal and anytime, in the deployment sense?

The existing test_anytime_causality perturbs the future and checks early deadlines do not move.
That proves no leakage inside one forward pass over a whole trial. It does NOT prove the stronger
property a deployed system needs: that the logits we report at deadline d equal what the model
produces when it is only ever GIVEN d seconds of signal. This tests that directly, by truncating
the input and comparing against the anytime curve read from the full trial.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS  # noqa: E402
from reader.anytime import deadline_index  # noqa: E402
from reader.models import reader as reader_module  # noqa: E402


def _model(ds):
    torch.manual_seed(11)
    m, _, _, _ = reader_module.build("reader", ds)
    return m.eval()


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_truncated_input_matches_anytime_curve(ds):
    """Feeding only the first t tokens must reproduce the t-th entry of the anytime curve."""
    spec, m = SPECS[ds], _model(ds)
    pool = spec["pool"]
    x = torch.randn(4, spec["channels"], spec["samples"])
    with torch.no_grad():
        curve = m.anytime_logits(x)
        for t in (2, 5, spec["samples"] // pool // 2, spec["samples"] // pool):
            if t < 1 or t > curve.shape[1]:
                continue
            truncated = m.anytime_logits(x[..., : t * pool])
            assert truncated.shape[1] == t, f"deadline {t}: got {truncated.shape[1]} tokens"
            delta = (truncated[:, t - 1] - curve[:, t - 1]).abs().max().item()
            assert delta < 1e-4, f"deadline {t} tokens: truncated differs by {delta}"


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_no_future_samples_reach_an_early_deadline(ds):
    """Replacing everything after the deadline with garbage must not move the decision at all."""
    spec, m = SPECS[ds], _model(ds)
    pool = spec["pool"]
    x = torch.randn(4, spec["channels"], spec["samples"])
    t = max(2, spec["samples"] // pool // 3)
    x2 = x.clone()
    x2[..., t * pool:] = 1e3 * torch.randn_like(x2[..., t * pool:])
    with torch.no_grad():
        a, b = m.anytime_logits(x), m.anytime_logits(x2)
    assert torch.equal(a[:, :t], b[:, :t]), (a[:, :t] - b[:, :t]).abs().max().item()


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_reported_deadline_is_not_earlier_than_claimed(ds):
    """The 128 ms token grid is inherited from BiTE's pooling; a '1 s' deadline is not on it.
    Whatever token we report for a wall-clock deadline must end at or AFTER that deadline."""
    spec = SPECS[ds]
    pool_ms = 1000.0 * spec["pool"] / spec["fs"]
    for d in (1.0, 2.0, 3.0):
        if d * 1000 > spec["samples"] / spec["fs"] * 1000:
            continue
        idx = deadline_index(d, pool_ms, 10_000)          # the REAL callee, not a copy of it
        token_end_ms = (idx + 1) * pool_ms
        assert token_end_ms >= d * 1000 - 1e-6, (
            f"{ds}: reporting {d}s at a token that ends at {token_end_ms:.1f} ms")
