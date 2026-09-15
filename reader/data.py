"""Dataset specifications, electrode geometry, and the official data roles.

Roles and preprocessing reproduce the released BiTE loaders: train on 100% of the official training role,
evaluate the official test role. ``input_mode='quotient'`` is the only addition:
the channel mean is removed from the RAW trial before the pointwise scaler is
fitted, so the network receives the reference-quotient signal.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Prepared corpora live here. Override with READER_DATA=/path/to/data (see docs/DATA.md).
DATA = Path(os.environ.get("READER_DATA", Path(__file__).resolve().parents[1] / "data"))

SPECS = {
    "2a": dict(subjects=9, channels=22, classes=4, samples=1000, fs=250.0, pool=32,
               bite_kernel=64, bite_pool=32, low_cut=4, high_cut=40, bipolar=False,
               root=DATA / "bci_iv_2a/raw/2a_pre",
               names=["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
                      "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]),
    "2b": dict(subjects=9, channels=3, classes=2, samples=1000, fs=250.0, pool=32,
               bite_kernel=64, bite_pool=32, low_cut=4, high_cut=40, bipolar=True,
               root=DATA / "bci_iv_2b/raw/2b_pre", names=["C3", "Cz", "C4"]),
    "hgd": dict(subjects=14, channels=44, classes=4, samples=1001, fs=250.0, pool=32,
                bite_kernel=64, bite_pool=32, low_cut=4, high_cut=40, bipolar=False,
                root=DATA / "hgd/processed_bite_repo_exact",
                names=["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "FCz", "C5", "C3", "C1", "C2",
                       "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "FFC5h",
                       "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC1h",
                       "FCC2h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h",
                       "CCP6h", "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h"]),
    # Compact family uses pool 4 on SD-SSVEP (token rate above the 9.25-14.75 Hz
    # carriers); BiTE's own config uses kernel 32 / pool 8 and an 8-64 Hz band.
    "sdssvep": dict(subjects=10, channels=8, classes=12, samples=256, fs=256.0, pool=4,
                    bite_kernel=32, bite_pool=8, low_cut=8, high_cut=64, bipolar=False,
                    root=DATA / "sd_ssvep/processed_kaggle_exact",
                    names=["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]),
}

# Hardest and easiest subject per corpus by fixed-final reference result.
SCREEN_CELLS = {"2a": (2, 9), "2b": (9, 4), "hgd": (11, 14), "sdssvep": (1, 5)}
CHANCE = {"2a": .25, "2b": .5, "hgd": .25, "sdssvep": 1 / 12}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def electrode_positions(dataset: str) -> torch.Tensor:
    """Unit-scaled standard_1005 3-D positions, centred on the montage."""
    import mne
    ch_pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    pos = np.stack([ch_pos[name] for name in SPECS[dataset]["names"]]).astype(np.float64)
    pos = pos - pos.mean(0, keepdims=True)
    pos = pos / np.linalg.norm(pos, axis=1).max()
    return torch.tensor(pos, dtype=torch.float32)


def _npz(path: Path):
    with np.load(path) as item:
        return np.asarray(item["data"], dtype=np.float64), np.asarray(item["label"], dtype=np.int64)


def raw_roles(dataset: str, subject: int):
    root = SPECS[dataset]["root"]
    if dataset == "2a":
        return (*_npz(root / f"A0{subject}T.npz"), *_npz(root / f"A0{subject}E.npz"))
    if dataset == "2b":
        train = [_npz(root / f"B0{subject}0{s}T.npz") for s in (1, 2, 3)]
        held = [_npz(root / f"B0{subject}0{s}E.npz") for s in (4, 5)]
        return (np.concatenate([t[0] for t in train]), np.concatenate([t[1] for t in train]),
                np.concatenate([h[0] for h in held]), np.concatenate([h[1] for h in held]))
    return (*_npz(root / f"S{subject:02d}_train.npz"), *_npz(root / f"S{subject:02d}_test.npz"))


def load_roles(dataset: str, subject: int, input_mode: str = "bite", test_transform=None):
    """Return float32 (train_x, train_y, test_x, test_y).

    bite     : pointwise StandardScaler fit on training (HGD: inherited artifact as is)
    quotient : channel mean removed from the raw trial, then the same scaler.
               Bipolar 2b is left native (already a difference measurement).
    """
    if input_mode not in {"bite", "quotient"}:
        raise ValueError(input_mode)
    train_x, train_y, test_x, test_y = raw_roles(dataset, subject)
    if test_transform is not None:          # stress tests: perturb the RAW test role only
        test_x = test_transform(test_x)
    quotient = input_mode == "quotient" and not SPECS[dataset]["bipolar"]
    if quotient:
        train_x = train_x - train_x.mean(1, keepdims=True)
        test_x = test_x - test_x.mean(1, keepdims=True)
    if dataset != "hgd" or quotient:
        scaler = StandardScaler().fit(train_x.reshape(len(train_x), -1))
        train_x = scaler.transform(train_x.reshape(len(train_x), -1)).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(len(test_x), -1)).reshape(test_x.shape)
    return (train_x.astype(np.float32), train_y, test_x.astype(np.float32), test_y)


# ---------------------------------------------------------------- cross-subject (LOSO)
def _euclidean_alignment_matrix(x):
    """BiTE's `fuduiban`: R^{-1/2} of the mean trial covariance, read from
    research/repos/BiTE_official/get_data.py:32-44 and train.py:424-426.

    NOTE, and this is the whole protocol: BiTE computes ONE matrix from the POOLED
    TRAINING subjects and applies it unchanged to the held-out subject, so it is a fixed
    whitening learned from the training pool, not per-subject Euclidean Alignment. The
    held-out subject's own covariance is never touched, which is why this is legal here.
    """
    r = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for trial in x:
        r += trial @ trial.T
    r /= x.shape[0]
    v, q = np.linalg.eig(r)
    ss = np.diag(v ** -0.5)
    ss[np.isnan(ss)] = 0
    ss[np.isinf(ss)] = 0
    return np.real(q @ ss @ np.linalg.inv(q))


def loso_roles(dataset: str, held: int, input_mode: str = "bite", euclidean_alignment: bool = True):
    """Leave-one-subject-out roles, matched to BiTE's `*_cross_subject` loaders.

    Train = every OTHER subject's training and evaluation sessions pooled; test = the held
    subject's sessions pooled. StandardScaler is fit on the pooled training role, then the
    Euclidean-alignment whitening is computed on the SCALED training role, exactly the order
    in BiTE's train.py (get_data standardises, then fuduiban(X_train) runs on its output).
    """
    if input_mode not in {"bite", "quotient"}:
        raise ValueError(input_mode)
    spec = SPECS[dataset]
    if not 1 <= held <= spec["subjects"]:
        raise ValueError(f"{dataset} has subjects 1..{spec['subjects']}, got {held}")
    pooled_x, pooled_y, test_x, test_y = [], [], None, None
    for subject in range(1, spec["subjects"] + 1):
        a, ay, b, by = raw_roles(dataset, subject)
        x, y = np.concatenate([a, b]), np.concatenate([ay, by])
        if subject == held:
            test_x, test_y = x, y
        else:
            pooled_x.append(x)
            pooled_y.append(y)
    train_x, train_y = np.concatenate(pooled_x), np.concatenate(pooled_y)
    if input_mode == "quotient" and not spec["bipolar"]:
        train_x = train_x - train_x.mean(1, keepdims=True)
        test_x = test_x - test_x.mean(1, keepdims=True)
    scaler = StandardScaler().fit(train_x.reshape(len(train_x), -1))
    train_x = scaler.transform(train_x.reshape(len(train_x), -1)).reshape(train_x.shape)
    test_x = scaler.transform(test_x.reshape(len(test_x), -1)).reshape(test_x.shape)
    if euclidean_alignment:
        w = _euclidean_alignment_matrix(train_x)
        train_x = np.einsum("dc,bct->bdt", w, train_x)
        test_x = np.einsum("dc,bct->bdt", w, test_x)
    return (train_x.astype(np.float32), train_y.astype(np.int64),
            test_x.astype(np.float32), test_y.astype(np.int64))
