"""The re-implementation must reproduce the archived compact graph exactly."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT.parent / "_archive/DO_NOT_READ_PRE_RESET_20260904/anytime-eeg"
sys.path.insert(0, str(ARCHIVE))

from reader.models.compact import CompactDecoder  # noqa: E402
from cba_online.model_tcn_sequence import build_tcn_sequence  # noqa: E402


def check(n_ch, n_cls, samples, pool, variant):
    torch.manual_seed(0)
    old = build_tcn_sequence(variant, n_ch=n_ch, n_cls=n_cls)
    new = CompactDecoder(n_ch, n_cls, pool=pool)
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert not missing.missing_keys, missing.missing_keys
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    x = torch.randn(5, n_ch, samples)
    for mode in ("eval", "train"):
        getattr(old, mode)(); getattr(new, mode)()
        torch.manual_seed(1); a = old(x)
        torch.manual_seed(1); b = new(x)["sequence"]
        assert a.shape == b.shape, (a.shape, b.shape)
        err = (a - b).abs().max().item()
        assert err < 1e-5, (variant, mode, err)
        print(variant, mode, tuple(a.shape), "max abs diff", err, "params", sum(p.numel() for p in new.parameters()))


if __name__ == "__main__":
    check(22, 4, 1000, 32, "tcnseq_ceilmean_maxnorm_narrow")
    check(3, 2, 1000, 32, "tcnseq_ceilmean_maxnorm_narrow")
    check(44, 4, 1001, 32, "tcnseq_ceilmean_maxnorm_narrow")
    check(8, 12, 256, 4, "tcnseq_ceilmean_maxnorm_narrow_pool4")
    print("PARITY OK")
