"""The BiTE baseline zoo is BiTE's own code: parameter counts must equal BiTE's report.txt, every model must
train on every corpus except the one documented restriction, and import placeholders must never be used."""
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.data import SPECS  # noqa: E402
from reader.models import registry  # noqa: E402
from reader.models.baseline import REPO  # noqa: E402

pytestmark = pytest.mark.skipif(not REPO.exists(), reason="run scripts/fetch_baseline.py first")
MODELS = ["ATCNet", "DMSANet", "DeepConvNet", "EEGNeX", "EEGNet", "EEGTCNet", "EISATC", "FACTNet",
          "MBCNNEATCFNet", "ShallowConvNet"]


@pytest.mark.parametrize("model", MODELS)
def test_parameter_count_equals_bites_own_report(model):
    report = (REPO / "results/2a_within_subject" / model / "report.txt").read_text()
    expected = int(re.search(r"Number of parameters: (\d+)", report).group(1))
    net, _, _, _ = registry.build(f"zoo_{model}", "2a")
    assert sum(p.numel() for p in net.parameters()) == expected


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("ds", ["2a", "2b", "sdssvep", "hgd"])
def test_trains_on_every_supported_corpus(model, ds):
    from reader.models.zoo import UNSUPPORTED
    spec = SPECS[ds]
    torch.manual_seed(0)
    x = torch.randn(4, spec["channels"], spec["samples"])
    y = torch.arange(4) % spec["classes"]
    if (model, ds) in UNSUPPORTED:
        with pytest.raises(IndexError):
            net, _, _, _ = registry.build(f"zoo_{model}", ds)
            net(x)
        return
    net, _, _, _ = registry.build(f"zoo_{model}", ds)
    logits = net(x)["logits"]
    assert logits.shape == (4, spec["classes"])
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert all(p.grad is not None for p in net.parameters())


def test_import_placeholders_fail_loudly_if_used():
    registry.build("zoo_EEGNet", "2a")
    for module_name, attr in (("visdom", "Visdom"), ("torchstat", "stat"), ("torchsummary", "summary")):
        module = sys.modules.get(module_name)
        if module is not None and not hasattr(module, "__file__"):
            with pytest.raises(RuntimeError):
                getattr(module, attr)()
