"""BiTE baseline, imported from the authors' own released code.

The BiTE repository carries NO LICENSE file, so its source is NOT redistributed here. Run
``python scripts/fetch_baseline.py`` to clone it at the commit this work was matched against;
this module then imports the released ``BiTE`` class unchanged. Nothing about the architecture or
its hyper-parameters is re-implemented -- the values below are read from the authors' config.yaml
so the baseline is theirs, not our reading of their paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

BITE_COMMIT = "924eb32241ba1a7c80dbc4ba097f8c979da17578"
BITE_URL = "https://github.com/cindy-hong/BiteEEG"
REPO = Path(__file__).resolve().parents[2] / "third_party/BiteEEG"

# From the released config.yaml; kernel_size1/pk1 and the analysis band are per corpus.
BITE_SPECS = {
    "2a":      dict(kernel_size=64, pool=32, low_cut=4, high_cut=40, sampling_rate=250),
    "2b":      dict(kernel_size=64, pool=32, low_cut=4, high_cut=40, sampling_rate=250),
    "hgd":     dict(kernel_size=64, pool=32, low_cut=4, high_cut=40, sampling_rate=250),
    "sdssvep": dict(kernel_size=32, pool=8,  low_cut=8, high_cut=64, sampling_rate=256),
}


def _import_bite():
    if not REPO.exists():
        raise RuntimeError(
            f"BiTE source not found at {REPO}.\n"
            f"Run: python scripts/fetch_baseline.py   (clones {BITE_URL} at {BITE_COMMIT[:12]})")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from model.BiTE import BiTE  # noqa: PLC0415
    return BiTE


class BiTEAdapter(nn.Module):
    """The released BiTE behind this repository's calling convention."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, spec=None):
        logits, _ = self.inner(x, spec)
        return {"logits": logits, "sequence": None, "route_logits": None}


def build(name, dataset, samples, regime="within", **_):
    from reader.data import SPECS
    BiTE = _import_bite()
    spec, bite = SPECS[dataset], BITE_SPECS[dataset]
    inner = BiTE(
        n_channels=int(spec["channels"]), n_classes=int(spec["classes"]),
        sample_points=int(samples), F1=16,
        kernel_size1=bite["kernel_size"], D=4, pk1=bite["pool"],
        tcn_k_size=6, BiTCN=True, dropout=0.3 if regime == "within" else 0.15,
        use_time=True, use_freq=True, use_att=True,
        sampling_rate=bite["sampling_rate"], low_cut=bite["low_cut"], high_cut=bite["high_cut"],
    )
    keys = {
        "time_stream": ("inner.conv_1.spectral_spatial.6", "channels_time"),
        "freq_stream": ("inner.conv_1.stft_spatial_conv.dropout", "channels_time"),
        "reader": ("inner.classifier_1", "vector", "input"),
    }
    return BiTEAdapter(inner), keys, {}, True


def stft_exact(x, dataset, device, batch_size=64):
    """BiTE's own spectral input, reproduced from their loader.

    NOTE ``center=True`` with ``hop_length=1``: each frame is centred on its sample, so the frame at
    time t contains samples up to t + window/2. The BiTE baseline is therefore NON-CAUSAL at
    inference by construction. This is stated in the paper as the reason a deadline-respecting
    comparison needs a separately retrained BiTE per deadline.
    """
    bite = BITE_SPECS[dataset]
    window_length = int(bite["kernel_size"])
    window = torch.hann_window(window_length, device=device)
    rows = []
    for start in range(0, len(x), batch_size):
        value = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        batch, channels, samples = value.shape
        spectrum = torch.stft(
            value.reshape(batch * channels, samples), n_fft=window_length,
            hop_length=1, win_length=window_length, window=window,
            center=True, pad_mode="constant", normalized=False, onesided=True,
            return_complex=True,
        ).abs()
        spectrum = spectrum.reshape(batch, channels, window_length // 2 + 1, spectrum.shape[-1])
        begin = max(int(bite["low_cut"] * window_length / bite["sampling_rate"]), 0)
        end = min(int(bite["high_cut"] * window_length / bite["sampling_rate"]) + 1, spectrum.shape[2])
        rows.append(spectrum[:, :, begin:end].permute(0, 2, 3, 1).cpu())
    return torch.cat(rows)
