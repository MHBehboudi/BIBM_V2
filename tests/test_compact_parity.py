"""Structural invariants of the decoder.

The original bit-exactness check was against the archived implementation this was re-derived from,
which is not part of this repository. What a third party can check instead are the properties the
paper actually claims: the parameter counts it reports, the graph's shape, determinism, and the
three structural choices the ablations refer to.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.data import SPECS  # noqa: E402
from reader.models import registry  # noqa: E402
from reader.models.compact import CompactDecoder  # noqa: E402

# Parameter counts reported in the paper's model table.
PARAMS = {"2a": {"reader": 20964, "compact": 17828},
          "2b": {"reader": 19010, "compact": 15874},
          "hgd": {"reader": 23076, "compact": 19940},
          "sdssvep": {"reader": 32428, "compact": 29292}}


@pytest.mark.parametrize("ds", sorted(SPECS))
@pytest.mark.parametrize("arm", ["reader", "compact"])
def test_parameter_count_matches_the_reported_table(ds, arm):
    model, _, _, _ = registry.build(arm, ds)
    assert sum(p.numel() for p in model.parameters()) == PARAMS[ds][arm]


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_shapes_and_determinism(ds):
    spec = SPECS[ds]
    torch.manual_seed(0)
    model, _, _, _ = registry.build("reader", ds)
    model.eval()
    x = torch.randn(3, spec["channels"], spec["samples"])
    with torch.no_grad():
        a, b = model(x), model(x)
    assert a["logits"].shape == (3, spec["classes"])
    assert a["sequence"].shape[0] == 3 and a["sequence"].shape[2] == spec["classes"]
    assert torch.equal(a["logits"], b["logits"]), "eval-mode forward is not deterministic"


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_structural_choices_the_ablations_depend_on(ds):
    spec = SPECS[ds]
    model = CompactDecoder(spec["channels"], spec["classes"], pool=spec["pool"])
    # one spatial filter group per carrier, so spatial mixing is per frequency carrier
    assert model.spatial.groups == model.tbn.num_features
    assert model.spatial.max_norm == 1.0 and model.head_max_norm == 0.25
    # every residual block is an identity at init: the second conv is zero-initialised
    for block in model.tcn.blocks:
        assert torch.count_nonzero(block.conv2.weight) == 0
        assert block.conv1.groups == block.conv1.in_channels    # fully depthwise
    # no bias anywhere in the front end: BiTE's convention, kept so the arms stay comparable
    for m in (model.spatial, *model.temporal):
        assert m.bias is None


@pytest.mark.parametrize("ds", sorted(SPECS))
def test_reader_shares_the_parent_front_end_bitwise(ds):
    """The two arms must be exactly paired at a given seed, or their delta is not a paired delta."""
    torch.manual_seed(5)
    reader, _, _, _ = registry.build("reader", ds)
    torch.manual_seed(5)
    parent, _, _, _ = registry.build("compact", ds)
    shared = [k for k in parent.state_dict() if not k.startswith(("backward_tcn", "backward_gate"))]
    for k in shared:
        assert torch.equal(reader.state_dict()[k], parent.state_dict()[k]), k
