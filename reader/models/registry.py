"""Model registry: every entry returns (module, key_points, max_norm_caps, needs_spectral)."""
from __future__ import annotations

from reader.data import SPECS


def build(name: str, dataset: str, **options):
    samples = int(options.pop("samples", SPECS[dataset]["samples"]))
    if name == "bite":
        from reader.models import baseline
        return baseline.build(name, dataset, samples, **options)
    from reader.models import reader as reader_module
    return reader_module.build(name, dataset, **options)


def available():
    from reader.models.reader import ARMS
    return sorted(ARMS) + ["bite"]
