# Data: sources, preparation, and what this repository does and does not ship

No EEG is redistributed here. All four corpora are public but carry their own terms; each must be
obtained from its own source. This document states exactly what shape and preprocessing the code
expects, so a prepared tree can be verified against `results/data_checksums.json` before any run.

Point the code at a prepared tree with `READER_DATA=/path/to/data`.

## Expected layout

```
$READER_DATA/
  bci_iv_2a/raw/2a_pre/A0{1..9}{T,E}.npz          18 files   [288, 22, 1000]  4 classes
  bci_iv_2b/raw/2b_pre/B0{1..9}0{1..5}{T,E}.npz   45 files   [120,  3, 1000]  2 classes
  hgd/processed_bite_repo_exact/S{01..14}_{train,test}.npz    [*, 44, 1001]   4 classes
  sd_ssvep/processed_kaggle_exact/S{01..10}_{train,test}.npz  [*,  8,  256]  12 classes
```

Each `.npz` holds `data` (trials x channels x samples, float) and `label` (int, 0-based).

## Sources

| corpus | source | roles |
|---|---|---|
| BCI IV-2a | [bnci-horizon-2020.eu](http://bnci-horizon-2020.eu/database/data-sets) (`BCICIV_2a_gdf` + true labels) | session `T` trains, session `E` tests |
| BCI IV-2b | same competition release | sessions 1-3 train, sessions 4-5 test |
| HGD | [gin.g-node.org/robintibor/high-gamma-dataset](https://gin.g-node.org/robintibor/high-gamma-dataset) | the BiTE-replication 80/20 split, fixed |
| SD-SSVEP | Nakanishi et al. 12-class SSVEP | fixed per-subject `trainEEG`/`testEEG` files |

## Preprocessing, and why it is what it is

Preprocessing reproduces the **released BiTE loaders**, not our reading of their paper, so that the
baseline is compared on its own terms. Per-corpus settings come from their `config.yaml`:

- **2a / 2b / HGD**: 4-40 Hz analysis band, temporal kernel 64, pooling 32, 250 Hz.
- **SD-SSVEP**: 8-64 Hz, kernel 32, pooling 8, 256 Hz, 1 s window starting after the 135 ms
  visual-latency offset.
- **HGD**: 44 motor-cortex channels of the original 128; MNE FFT resampling 500 -> 250 Hz; no
  filtering (`is_filter=false` in their config); epoch 0-4 s -> 1001 samples. Their loader applies
  an all-epoch standardisation *before* splitting, which is transductive; we retain it verbatim
  rather than silently improving it, and the retention is recorded in each `S*_metadata.json`.
- **Scaling**: a pointwise `StandardScaler` fit on the training role only and applied to the test
  role. HGD is the one exception, where the inherited artifact above already standardised it.

`reader/data.py` performs the scaling and role construction; it expects the `.npz` tree above as
its input. The `.npz` files themselves were produced by replicating the BiTE loaders against the
raw downloads; **that conversion step is not included in this repository**, so a third party
reproducing from raw sources must re-derive it from the settings above and verify against the
anchor checksums below.

## Verifying a prepared tree

`results/data_checksums.json` records, for two anchor files per corpus, the SHA-256, the array
shapes, and the label set:

| corpus | anchor | shape | sha256 (first 16) |
|---|---|---|---|
| 2a | `A01T.npz` | [288, 22, 1000] | `88a6c6f9982a48e5` |
| 2b | `B0101T.npz` | [120, 3, 1000] | `19abd7da0001a6bb` |
| HGD | `S01_train.npz` | [256, 44, 1001] | `361330fbc1ecf29d` |
| SD-SSVEP | `S01_train.npz` | [120, 8, 256] | `637af6d7f3838bff` |

A tree that matches these reproduces the numbers in `results/`. A tree that does not match is not
necessarily wrong -- file-level byte equality is sensitive to library versions -- but the shapes and
label sets must match exactly, and any accuracy difference should be attributed to preparation
before it is attributed to the model.

## Cross-subject roles

The leave-one-subject-out protocol is decoded from BiTE's `get_data.py` and `train.py`: training
pools **both** sessions of every other subject; one `StandardScaler` is fit on that pool. Their
`is_EA: true` is **not** per-subject Euclidean Alignment -- `train.py` computes `fuduiban` on the
pooled training role and applies that single matrix to the held-out subject too. `reader/data.py`
implements this as `loso_roles`, and `tests/test_loso_roles.py` checks our matrix against their own
`fuduiban`, executed from their source text, to 1e-8. That test skips until you run
`python scripts/fetch_baseline.py`.
