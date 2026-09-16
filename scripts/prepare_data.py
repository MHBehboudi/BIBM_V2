#!/usr/bin/env python3
"""Build the prepared corpus tree by running BiTE's OWN preprocessing functions.

Preprocessing is not re-implemented here. This driver imports ``preprocess_2a`` / ``preprocess_2b``
/ ``preprocess_hgd`` / ``preprocess_sdssvep`` from the fetched BiTE repository and calls them with
the exact settings from their ``config.yaml``, then arranges the outputs into the layout
``reader/data.py`` expects. That keeps the baseline comparison honest: both models see the
authors' own preprocessing, not our reading of their paper.

    python scripts/fetch_baseline.py                  # required first
    pip install -r requirements-prepare.txt           # mne, scipy, braindecode (HGD only)
    python scripts/prepare_data.py --raw /path/to/downloads --out data --corpus 2a 2b sdssvep

See docs/DATA.md for where each raw download comes from and what it must contain.
Verify the result with:  python scripts/prepare_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Exactly the dataset blocks from BiTE's config.yaml (commit 924eb322).
CONFIG = {
    "2a":      dict(classes=4,  subjs=9,  chans=22, fs=250, seed=2025, start=0, stop=4,
                    time_points=1000, is_standard=True, is_ica=False, is_filter=False, is_EA=False),
    "2b":      dict(classes=2,  subjs=9,  chans=3,  fs=250, seed=2025, start=0, stop=4,
                    time_points=1000, is_standard=True, is_ica=False, is_filter=False, is_EA=False),
    "sdssvep": dict(classes=12, subjs=10, chans=8,  fs=256, seed=2025, time_points=256,
                    is_standard=True, is_filter=False, is_EA=False, delay=0.135),
    "hgd":     dict(classes=4,  subjs=14, chans=44, fs=250, seed=2025, start=0, stop=4,
                    time_points=1001, is_standard=True, is_filter=False, is_EA=False,
                    use_motor_cortex_only=True),
}

# Where this repository expects each corpus to land, relative to --out.
DEST = {"2a": "bci_iv_2a/raw/2a_pre", "2b": "bci_iv_2b/raw/2b_pre",
        "hgd": "hgd/processed_bite_repo_exact", "sdssvep": "sd_ssvep/processed_kaggle_exact"}
# What BiTE's own functions name their output directory, under data_path.
SRC_SUBDIR = {"2a": "2a_pre", "2b": "2b_pre", "hgd": "hgd_pre", "sdssvep": "sdssvep_pre"}


def _bite_module(corpus: str):
    """BiTE's get_data.py, imported whole when possible.

    Its top-level ``from braindecode.datasets import HGD`` is needed only by ``preprocess_hgd`` but
    is evaluated at import time, and braindecode pulls torchaudio, which fails on machines whose
    CUDA runtime does not match. For the other three corpora we therefore exec the module source
    with that one import removed, so 2a / 2b / SD-SSVEP can be prepared without braindecode
    installed at all. Nothing else about their code is altered, and HGD still requires the real
    import -- it is refused rather than faked.
    """
    from reader.models.baseline import REPO
    if not REPO.exists():
        sys.exit(f"BiTE source missing at {REPO}\nRun: python scripts/fetch_baseline.py")
    sys.path.insert(0, str(REPO))
    try:
        import get_data
        return get_data
    except Exception as exc:
        if corpus == "hgd":
            sys.exit(f"HGD preparation needs BiTE's braindecode import, which failed: {exc}\n"
                     f"pip install -r requirements-prepare.txt   (and a torch matching your CUDA)")
        print(f"note: importing get_data.py whole failed ({type(exc).__name__}); "
              f"loading it without the HGD-only braindecode import", flush=True)
        source = (REPO / "get_data.py").read_text()
        stripped = "\n".join("" if line.startswith("from braindecode") else line
                              for line in source.splitlines())
        namespace: dict = {"__name__": "bite_get_data"}
        exec(compile(stripped, str(REPO / "get_data.py"), "exec"), namespace)
        module = type(sys)("bite_get_data")
        module.__dict__.update(namespace)
        return module


def prepare(corpus: str, raw: Path, out: Path):
    get_data = _bite_module(corpus)
    fn = getattr(get_data, f"preprocess_{corpus}")
    cfg = dict(CONFIG[corpus], data_path=str(raw / corpus))
    if not (raw / corpus).exists():
        sys.exit(f"raw input not found: {raw / corpus}   (see docs/DATA.md)")
    for setting in ("train", "test"):
        print(f"[{corpus}] {setting} ...", flush=True)
        fn(cfg, setting=setting)
    produced = raw / corpus / SRC_SUBDIR[corpus]
    dest = out / DEST[corpus]
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(produced.glob("*.npz")):
        shutil.copy2(f, dest / f.name); n += 1
    print(f"[{corpus}] {n} files -> {dest}")


def verify(out: Path):
    """Compare a prepared tree against the anchors recorded in results/data_checksums.json."""
    import numpy as np
    anchors = json.loads((ROOT / "results/data_checksums.json").read_text())
    ok = True
    for corpus, meta in anchors.items():
        dest = out / DEST[corpus]
        files = sorted(dest.glob("*.npz"))
        status = "OK " if len(files) == meta["n_files"] else "BAD"
        if status == "BAD":
            ok = False
        print(f"{status} {corpus:9s} {len(files):3d}/{meta['n_files']} files  {dest}")
        for name, expect in meta["anchors"].items():
            p = dest / name
            if not p.exists():
                print(f"    MISSING {name}"); ok = False; continue
            with np.load(p) as z:
                shape = list(z["data"].shape)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            same_shape = shape == expect["data_shape"]
            same_bytes = digest == expect["sha256"]
            mark = "OK " if same_shape else "BAD"
            if not same_shape:
                ok = False
            print(f"    {mark} {name}  shape {shape} (expect {expect['data_shape']})"
                  f"  bytes {'match' if same_bytes else 'DIFFER'}")
    print("\nShapes and file counts must match. Byte equality is version-sensitive: a tree whose"
          "\nshapes match but whose bytes differ is usable, but attribute any accuracy difference"
          "\nto preparation before attributing it to the model.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, help="directory holding one subdirectory per corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "data", help="prepared tree to write")
    ap.add_argument("--corpus", nargs="+", default=["2a", "2b", "sdssvep", "hgd"], choices=list(CONFIG))
    ap.add_argument("--verify", action="store_true", help="check an existing tree and exit")
    args = ap.parse_args()
    if args.verify:
        sys.exit(verify(args.out))
    if not args.raw:
        ap.error("--raw is required unless --verify is given")
    for corpus in args.corpus:
        prepare(corpus, args.raw, args.out)
    print("\nNow verify:  python scripts/prepare_data.py --verify --out", args.out)


if __name__ == "__main__":
    main()
