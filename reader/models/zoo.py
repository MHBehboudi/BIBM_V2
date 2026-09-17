"""BiTE's baseline model zoo, built by BiTE's OWN `get_model` from the pinned release, code unchanged.

Every baseline in BiTE's within-subject tables (third_party/BiteEEG, fetched by scripts/fetch_baseline.py at commit 924eb322):
ATCNet, DMSANet, DeepConvNet, EEGNeX, EEGNet, EEGTCNet, EISATC, FACTNet, MBCNNEATCFNet, ShallowConvNet.
`get_model(name, model_config)` hard-codes each model's hyper-parameters; main.py's
apply_dataset_adjustments only sets chans / classes / time_points / sampling_rate for them (its only
model-specific change is for BiTE on SD-SSVEP), which is exactly what is passed here.

Two import-time workarounds, neither touching model code:
  * model/FACT_util.py imports visdom, model/EISATC.py imports torchstat and model/causalconv.py imports
    torchsummary; none of them is ever called (grep), so each missing one gets a placeholder that
    raises if it is used.
  * model/EEGNeX.py imports braindecode.models.base.EEGModuleMixin. Importing the braindecode PACKAGE
    runs its __init__, which imports torchaudio, which is broken in this environment (built for
    CUDA 13 against torch 2.5.1+cu124). base.py itself needs only numpy/torch/torchinfo/
    docstring_inheritance, so the real base.py file is loaded under empty parent packages.
Verified: parameter counts on 2a equal those printed in BiTE's own report.txt for every model.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

from reader.data import SPECS
from reader.models.baseline import REPO
MODELS = ("ATCNet", "DMSANet", "DeepConvNet", "EEGNeX", "EEGNet", "EEGTCNet", "EISATC", "FACTNet",
          "MBCNNEATCFNet", "ShallowConvNet")


def _unused(name):
    def fail(*_a, **_k):
        raise RuntimeError(f"{name} was imported as an unused dependency and must never be called")
    return fail


def _prepare_imports():
    # Import-only dependencies (grep: never called in model/): placeholders that FAIL LOUDLY if used.
    placeholders = {"visdom": {"Visdom": _unused("visdom.Visdom")},
                    "torchstat": {"stat": _unused("torchstat.stat")},              # model/EISATC.py
                    "torchsummary": {"summary": _unused("torchsummary.summary")}}  # model/causalconv.py
    for module_name, attributes in placeholders.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            sys.modules[module_name] = types.SimpleNamespace(**attributes)
    if "braindecode.models.base" not in sys.modules:
        import torch as _t  # noqa: F401  (make sure torch is initialised first)
        root = None
        for entry in sys.path:
            candidate = Path(entry) / "braindecode" / "models" / "base.py"
            if candidate.exists():
                root = candidate
                break
        if root is None:
            raise ImportError("braindecode is not installed; EEGNeX needs braindecode.models.base")
        for name, path in (("braindecode", root.parents[1]), ("braindecode.models", root.parent)):
            if name not in sys.modules:
                pkg = types.ModuleType(name)
                pkg.__path__ = [str(path)]
                sys.modules[name] = pkg
        spec = importlib.util.spec_from_file_location("braindecode.models.base", root)
        module = importlib.util.module_from_spec(spec)
        sys.modules["braindecode.models.base"] = module
        spec.loader.exec_module(module)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


class ZooAdapter(nn.Module):
    """A BiTE-convention baseline behind this repo's convention: forward(x) -> {"logits": ...}."""

    def __init__(self, inner, name):
        super().__init__()
        self.inner, self.name = inner, name

    def forward(self, x):
        out = self.inner(x, None)                  # BiTE's trainer calls model(inputs, ffqs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return {"logits": logits, "sequence": None, "route_logits": None}


def build(name: str, dataset: str, samples: int | None = None):
    if name not in MODELS:
        raise KeyError(f"unknown zoo model {name!r}; have {MODELS}")
    if not REPO.exists():
        raise RuntimeError(f"BiTE source not found at {REPO}; run python scripts/fetch_baseline.py")
    _prepare_imports()
    from get_model import get_model  # noqa: E402  (BiTE's own factory)
    spec = SPECS[dataset]
    config = {"model_name": name, "chans": int(spec["channels"]), "classes": int(spec["classes"]),
              "time_points": int(samples or spec["samples"]), "sampling_rate": int(spec["fs"])}
    inner = get_model(name, config)
    return ZooAdapter(inner, name)


# FACTNet's FA layer hard-codes seq_len = 1000 and indexes random bins below 500: it cannot run on
# SD-SSVEP's 256-sample trials as released. Not adapted here; see results/zoo_*.md.
UNSUPPORTED = {("FACTNet", "sdssvep")}
