# REACT-EEG

**Reverse-Encoded Anytime Causal Temporal Modeling for EEG Decoding**

REACT-EEG is an EEG classifier that can be queried at different observation times using the **same trained checkpoint**. A forward temporal reader summarizes the newest observed token, while an independently parameterized reverse reader processes the observed tokens from newest to oldest and finishes at the first token. A learned, feature-wise gate combines both representations before classification.

The associated manuscript evaluates motor-imagery and steady-state visual evoked potential decoding on **BCICIV-2A, BCICIV-2B, and SD-SSVEP**, covering 28 subjects. Its main experiment trains at the complete-trial endpoint and evaluates the resulting checkpoints at earlier observation boundaries. A separate experiment retrains REACT and its forward-only control with matched random-duration exposure.

> **Scope of this source snapshot.** This repository layout preserves the exported main experiment in `main_source/` and the later mechanism-control scripts in `anchor_source/`. It includes model, training, evaluation, statistical-analysis, and plotting code, but **does not include EEG datasets, trained checkpoints, or the complete per-run result records**. Some secondary workflows in the manuscript are not included, and several scripts retain machine-specific paths. See [Reproducibility status](#reproducibility-status) before attempting a complete paper reproduction.

## Contents

- [Method](#method)
- [Datasets and input format](#datasets-and-input-format)
- [Repository layout](#repository-layout)
- [Installation and a data-free example](#installation-and-a-data-free-example)
- [Training and evaluation](#training-and-evaluation)
- [Mechanism controls](#mechanism-controls)
- [Statistics and figures](#statistics-and-figures)
- [Manuscript results](#manuscript-results)
- [Reproducibility status](#reproducibility-status)
- [Limitations and attribution](#limitations-and-attribution)

## Method

At a decision boundary containing `m` prepared EEG samples per channel, the input has shape `[batch, channels, m]`:

```text
Observed prepared EEG: [B, C, m]
    |
    | Three left-padded temporal convolutions: 16, 32, 64 taps
    | 16 maps per branch -> concatenate -> BatchNorm
    v
Temporal features: [B, 48, C, m]
    |
    | Depthwise spatial convolution: two spatial filters per map
    | BatchNorm -> LeakyReLU
    v
Spatial features: [B, 96, m]
    |
    | Non-overlapping means, including an observed-only partial window
    | Dropout -> affine projection 96 -> 64 -> positional embeddings
    v
Positioned tokens: [B, 64, N], where N = ceil(m / pool)
    |                                  |
    | Original token order             | Reversed observed-token order
    v                                  v
Forward TCN                         Independent reverse TCN
Newest-token state [B, 64]           Start-anchored state [B, 64]
    |                                  |
    +------------ feature-wise gate ---+
                      |
                      v
               Linear classifier
                      |
                      v
                 Logits [B, K]
```

Each reader contains three residual blocks with dilations **1, 2, and 4**, two bias-free depthwise convolutions per block, and kernel size **6**. Its theoretical receptive field is **71 tokens**, covering all sequences evaluated in this study. The gate has 64 learned parameters, is initialized to equal weighting, and does not depend on the trial or observation duration:

$$
\mathbf g=\sigma(\mathbf a),\qquad
\mathbf h_m=(1-\mathbf g)\odot\mathbf f_m+\mathbf g\odot\mathbf b_m.
$$

The spatial-filter and classifier-row maximum norms are **1** and **0.25**, respectively. The implementation projects these weights at initialization in the experiment runner and after optimizer updates. Positional embeddings retain their original identities when the token sequence is reversed.

**Causality is defined at the prepared model input.** Predictions at `m` use only prepared samples through `m`. Inference disables dropout and uses stored BatchNorm statistics. This is not a claim that upstream acquisition or preprocessing is causal, nor that the training-time BatchNorm graph is a strictly online computation. A direct call recomputes the readers on the supplied observation; this is not a constant-cost recurrent streaming implementation or an adaptive stopping policy.

Source: [`models.py`](main_source/fresh/models.py), [`controls.py`](main_source/fresh/controls.py), and [`run.py`](main_source/fresh/run.py).

### Names used in code

Internal names are retained to preserve imports and checkpoint compatibility. **REACT-EEG is the proposed model**; the other names identify controls or the external baseline.

| Manuscript name | Code identifier |
|---|---|
| REACT / REACT-EEG | `reader` |
| No reverse | `compact` |
| Two forward | `ff` |
| Forward mean | `compact_mean` |
| BiTE | `bite` |
| Fixed anchor | `fixed_anchor` |
| Endpoint-aligned position | `relative_position` |
| Independently trained reverse-only | `reverse_only` output directory |
| Random-duration REACT / No reverse | `react_rt` / `compact_rt` |

The earlier `random_truncation` diagnostic is **not** the final paired `react_rt` versus `compact_rt` experiment.

## Datasets and input format

| Dataset | Code key | Subjects | Channels | Classes | Sampling rate | Input | Pool | Maximum tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BCICIV-2A | `2a` | 9 | 22 | 4 | 250 Hz | 1,000 samples / 4 s | 32 | 32 |
| BCICIV-2B | `2b` | 9 | 3 | 2 | 250 Hz | 1,000 samples / 4 s | 32 | 32 |
| SD-SSVEP | `ssvep` | 10 | 8 | 12 | 256 Hz | 256 samples / 1 s | 4 | 64 |

The executable loader consumes **prepared NPZ files**, not raw GDF or MATLAB recordings. Each file must contain:

```text
 data:  numeric array [trials, channels, samples], containing finite values
 label: integer array [trials], with zero-based class labels
```

For motor imagery, the loader accepts stored trials of 1,000 or 1,001 samples and retains the first 1,000. SD-SSVEP requires exactly 256 samples. It does not infer labels or reconstruct a train/test split.

Supply one directory per dataset with the following filenames:

```text
2a_pre/
    A01T.npz                 # Subject 1: training session
    A01E.npz                 # Subject 1: evaluation session
    ...                      # Subjects 01 through 09

2b_pre/
    B0101T.npz               # Subject 1: training session 1
    B0102T.npz               # Subject 1: training session 2
    B0103T.npz               # Subject 1: training session 3
    B0104E.npz               # Subject 1: test session 4
    B0105E.npz               # Subject 1: test session 5
    ...                      # Subjects 01 through 09

sdssvep_pre/
    S01_train.npz
    S01_test.npz
    ...                      # Subjects 01 through 10
```

For each subject, `StandardScaler` is fitted on training trials after flattening channel-by-sample coordinates. Its frozen mean and scale are applied to held-out trials and their shorter observations. There is no held-out-trial normalization fit. Exact duplicate prepared trials across the training/test roles are checked; that check does not establish the provenance or independence of the original raw trials.

SD-SSVEP observation time is measured from the beginning of the retained one-second segment. The manuscript describes a nominal 135-ms visual-response offset, but this loader receives already-cropped arrays and cannot verify the upstream crop index. The raw-to-NPZ conversion and original SD-SSVEP split construction are not supplied as an executable pipeline in this export.

Source: [`fresh/data.py`](main_source/fresh/data.py). Obtain the datasets separately; do not commit EEG recordings or per-trial data to this repository.

## Repository layout

Place this README at the repository root, alongside the two exported source directories:

```text
README.md
main_source/
    fresh/
        models.py                    # Tokenizer, readers, model factory
        controls.py                  # Shared readout-control implementation
        data.py                      # Prepared-data contract and observation grid
        bite.py                      # Adapter for pinned upstream BiTE
        setup.py                     # Freeze a new study and fetch BiTE
        run.py                       # Endpoint training, inference, causality tests
        smoke.py                     # Allocated-GPU, training-role preflight
        report.py                    # Primary experiment collector
        statistics_base.py           # Subject-first statistics
    tests/                           # Synthetic software checks
    scripts/                         # Original Slurm wrappers
    docs/
    paper/                           # Earlier manuscript/report template
anchor_source/
    fresh/anchor_models.py           # Anchor/position/earlier truncation controls
    fresh/anchor_diag.py
    fresh/anchor_report.py
    train_reverse_only_final.py
    train_paired_random_trunc_final.py
    reviewer_eval_controls.py
    bite_zerofill_final.py
    summarize_reverse_only.py
    summarize_paired_random_trunc_final.py
    final_endpoint_stats.py
    fresh_primary_nauc.py
    check_fresh_causality.py
    make_fresh_anytime_figure.py
    make_final_figure2_publication.py
main_evidence/                       # Selected exported provenance and analysis
```

`anchor_source/` also contains copies of the core `fresh/` modules. Those copies match `main_source/` in this export. Avoid adding both directories to `PYTHONPATH` at once: both provide a package named `fresh`. The commands below select the intended source explicitly. Earlier README files, report templates, and diagnostic documents describe their respective historical stages; they are not the final manuscript.

## Installation and a data-free example

Use a separate environment. Do not upgrade an environment used to produce archived results. The experiment runner uses POSIX file locking; Linux is the intended training platform, while the core model and CPU checks can also be explored on macOS.

For a new environment with Python 3.11 or later:

```bash
# Run from the repository root.
python3 -m venv .venv
source .venv/bin/activate
```

Install a PyTorch build appropriate for the machine using the [official installation instructions](https://pytorch.org/get-started/locally/). The local CPU checks described below used **PyTorch 2.10.0+cpu**. Then install the remaining versions used for those checks:

```bash
python -m pip install \
    numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 \
    pandas==2.2.3 matplotlib==3.10.8 pytest==9.0.2
```

These versions document a **local software-test environment**, not a recovered lockfile for the paper's HPC runs. NumPy 2.3.5 supports both the `np.trapz` calls retained in some older scripts and the [`np.trapezoid`](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html) calls used by newer scripts. The external BiTE implementation has separate dependencies; inspect the fetched revision's `requirements.txt` and run the GPU preflight before a full experiment.

### Instantiate and query REACT

This example uses random tensors only. It verifies shapes and parameter counts; it does not load a trained model or measure classification performance.

```bash
PYTHONPATH="$PWD/main_source" python - <<'PY'
import torch
from fresh.data import SPECS, grid
from fresh.models import make_model

torch.set_num_threads(1)
expected = {"2a": 18916, "2b": 16962, "ssvep": 20140}

for dataset, cfg in SPECS.items():
    model = make_model(
        channels=cfg["channels"], classes=cfg["classes"],
        max_samples=cfg["samples"], pool=cfg["pool"],
        kind="reader", seed=2025,
    )
    model.project_constraints()
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    assert count == expected[dataset]
    x = torch.randn(2, cfg["channels"], cfg["samples"])
    with torch.inference_mode():
        for m in (grid(dataset)[0], cfg["samples"]):
            logits = model(x[:, :, :m])
            assert logits.shape == (2, cfg["classes"])
    print(dataset, "parameters:", count, "shape checks passed")
PY
```

A real inference call requires the **matching trained `state_dict` and training-fitted scaler**. Construct the same model configuration, load the checkpoint with `strict=True`, normalize with the saved scaler, call `eval()`, and supply only samples observed at the requested boundary. Exported core checkpoints store weights under `checkpoint["state_dict"]`; they are not bare state dictionaries.

### Software tests

Run tests in a working copy, not in a frozen study's `code/` directory: some tests write software-validation records.

```bash
(
    cd main_source
    PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
        python -m pytest tests -q -k 'not full_reporting_layout'
)
```

This selection passed **41 tests** in a local CPU check using Python 3.13.5 and the versions above. It covers model dimensions, parameter pairing, partial pooling, frozen normalization, checkpoint/resume behavior, nondegenerate causality tests, synthetic integration, reporting rejection checks, and mocked scheduler commands. The excluded legacy layout test writes to a fixed `/mnt/data` location. No EEG benchmark, real GPU training, or upstream BiTE execution was part of this test result.

## Training and evaluation

### Training recipe

| Setting | Core-model value |
|---|---|
| Epochs / checkpoint | 600 / final epoch |
| Seeds | 2025, 2026, 2027 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate / weight decay | 0.002 / 0.002 |
| Adam betas / epsilon | (0.9, 0.999) / 1e-8 |
| Schedule | Cosine decay to zero |
| Label smoothing | 0.1 |
| Gradient-norm clipping | 5 |
| Dropout / LeakyReLU slope | 0.3 / 0.2 |
| Main training objective | Complete-trial endpoint cross-entropy |

Shared components are initialized identically across matched core models. The second reader uses a fixed constructor seed of **918273**, shared by REACT and Two forward; it is not independently reinitialized by each outer training seed. Each direct endpoint-training forward pass constructs only the supplied full-window representation, not a loop over all shorter reverse observations.

**BiTE discrepancy:** the supplied `fresh.run.train_one()` applies the same clipping operation to every model passed to it, including BiTE. The manuscript says clipping is disabled for BiTE. This source snapshot does not implement that exception. Resolve the discrepancy against the final run provenance rather than assuming a code change reproduces existing numbers.

<details>
<summary><strong>Reproduction commands: create a study, train, and collect results</strong></summary>

### Create a new study

Do not execute the exported `launch.sh` files unchanged on another cluster: they contain original interpreter, account, storage, and scheduler settings. A new study can instead be configured through the existing Python setup function.

Set three prepared-data roots and a **new output directory outside the repository**:

```bash
export REACT_DATA_2A="/absolute/path/to/2a_pre"
export REACT_DATA_2B="/absolute/path/to/2b_pre"
export REACT_DATA_SSVEP="/absolute/path/to/sdssvep_pre"
export REACT_STUDY="/absolute/path/to/new_react_study"
```

From the repository root, review and run:

```bash
python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

repo = Path.cwd().resolve()
study = Path(os.environ["REACT_STUDY"]).expanduser().resolve()
roots = {
    "2a": str(Path(os.environ["REACT_DATA_2A"]).expanduser().resolve()),
    "2b": str(Path(os.environ["REACT_DATA_2B"]).expanduser().resolve()),
    "ssvep": str(Path(os.environ["REACT_DATA_SSVEP"]).expanduser().resolve()),
}
if not (repo / "main_source/fresh/models.py").is_file():
    raise SystemExit("Run from the repository root.")
if study.exists() or study == repo or repo in study.parents:
    raise SystemExit("Choose a new study directory outside the repository.")
if not all(Path(p).is_dir() for p in roots.values()):
    raise SystemExit("A prepared-data directory does not exist.")

study.mkdir(parents=True)
shutil.copytree(
    repo / "main_source", study / "code",
    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
)
sys.path.insert(0, str(study / "code"))
from fresh import setup
setup.DEFAULT_ROOTS.update(roots)
setup.freeze(study)
PY
```

This records the study plan and source hashes, checks the required filenames, and obtains BiTE at commit `924eb32241ba1a7c80dbc4ba097f8c979da17578`. Network access to the [upstream repository](https://github.com/cindy-hong/BiteEEG) is needed unless that exact revision is available through the original local cache. This setup step does not submit training jobs. A failed setup can leave a partial study directory; preserve its diagnostics and choose a new directory for a corrected attempt rather than overwriting an existing experiment.

### Run an allocated-GPU preflight and one task

Run computation on an allocated compute node, **not an HPC login node**. Configure scheduler resources according to the local cluster's policies.

```bash
# In an allocated GPU session; REACT_STUDY is the study created above.
(
    cd "$REACT_STUDY/code"
    python -m fresh.smoke --study "$REACT_STUDY" && \
    python -m fresh.run --study "$REACT_STUDY" --index 0 --device cuda
)
```

Task `0` is BCICIV-2A, subject 1, seed 2025. The frozen plan has **84 subject/seed tasks**: indices 0–26 for 2A, 27–53 for 2B, and 54–83 for SSVEP. Within each dataset the order is subject first, then seed. Each main task trains `compact`, `reader`, `ff`, `compact_mean`, and `bite` sequentially: **420 model fits** for the full primary cohort. Held-out data are opened after all five models in that task have been trained. Core models are evaluated on the observation grid; this primary runner evaluates BiTE at the endpoint only.

Run each index once through a reviewed scheduler array or equivalent allocation. The main trainer supports signature-checked resume from `last.pt`; do not change a frozen plan or source tree to bypass an identity mismatch.

### Generated study files

```text
study/
    study.json
    code_hashes.json
    bite_provenance.json
    third_party/BiteEEG/
    runs/2a_S01_seed2025/
        scaler.npz
        training_data.json
        held_data.json
        pairing.json
        reader/
            final.pt
            last.pt
            result.json
            predictions.npz
            causality.json
            training_history.json
        compact/ ...
```

`predictions.npz` stores logits shaped `[test trials, observation boundaries, classes]`, labels `y`, prepared-row IDs `ids`, and sample counts `lengths`. Accuracies are calculated separately for each seed, then averaged; the three checkpoints are **not a prediction ensemble**.

After the full primary cohort completes, its collector is:

```bash
(cd "$REACT_STUDY/code" && python -m fresh.report --study "$REACT_STUDY")
```

It validates result completeness and stored identities and produces primary analyses. It refuses to overwrite an existing `analysis/` directory. Its generated `paper/` uses an **earlier manuscript template**, not the final paper pasted with this release; do not treat that generated draft as a camera-ready manuscript. The supplied export also omits the original figure artwork needed for a complete PDF build.

</details>

## Mechanism controls

<details>
<summary><strong>Commands and prerequisites for the control experiments</strong></summary>

The following commands use an existing completed primary study with its `study.json`, frozen `code/`, saved scalers, checkpoints, prepared data, and result records. They are **not runnable from the source archive alone**. Run from the repository root, with `REACT_STUDY` set to that study and `REACT_CONTROLS` set to a new output directory outside the repository:

```bash
export REACT_CONTROLS="/absolute/path/to/new_control_outputs"
```

### Independently trained reverse-only control

```bash
python anchor_source/train_reverse_only_final.py \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/reverse_only" \
    --task-index 0 --device cuda
```

This copies the tokenizer, reverse-reader, and classifier **initializations** from a newly constructed REACT model, omits the forward reader and gate, and trains independently. It is not a post-hoc route ablation of a trained fused checkpoint. Repeat task indices 0–83 for the full cohort.

### Paired random-duration training

```bash
python anchor_source/train_paired_random_trunc_final.py \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/random_duration" \
    --task-index 0 --device cuda
```

Both `compact_rt` and `react_rt` receive the same minibatch order and the same duration schedule. One boundary is sampled uniformly from `fresh.data.grid(dataset)` for each minibatch using a separate random-number generator. Each update uses one label-smoothed cross-entropy loss, without an extra endpoint term. Both arms are trained before opening held-out data. This script saves completed checkpoints but does **not** provide the main trainer's mid-training `last.pt` resume mechanism; always use fresh output paths for new experiment settings.

### Fixed-anchor and endpoint-aligned-position controls

```bash
PYTHONPATH="$PWD/anchor_source" python -m fresh.anchor_diag \
    --orig-study "$REACT_STUDY" --out "$REACT_CONTROLS/anchor" \
    --index 0 --device cuda
```

This diagnostic also trains an earlier forward-only `random_truncation` control and computes additional probes. That earlier experiment must not be substituted for the final paired random-duration comparison. Its collector is:

```bash
PYTHONPATH="$PWD/anchor_source" python -m fresh.anchor_report \
    --orig-study "$REACT_STUDY" --diag-study "$REACT_CONTROLS/anchor"
```

### Mean-imputation and route interventions

```bash
python anchor_source/reviewer_eval_controls.py \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/mean_imputation" \
    --dataset all --device cuda

python anchor_source/bite_zerofill_final.py \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/bite_mean_imputation" \
    --device cuda
```

Mean-imputed controls preserve the trained input length and replace samples after the boundary with zero **in standardized space**, corresponding to the training-set mean. The BiTE adapter then performs its ordinary processing on that completed waveform. Route interventions in `reviewer_eval_controls.py` reuse a jointly trained classifier and are distinct from independently trained reverse-only results.

</details>

## Statistics and figures

BCICIV-2A/2B evaluation covers **1–4 s at 28 boundaries**; SD-SSVEP covers **0.25–1 s of the retained segment at 49 boundaries**. The grids include partial pooling windows where required.

Seeds are averaged within each subject before dataset aggregation or paired comparison. The normalized area under the accuracy–time curve is trapezoidal area divided by the evaluated interval length. It is an **accuracy–time summary**, not ROC AUC. Since accuracy is in percent, model differences in nAUC are reported in percentage points.

The manuscript uses 10,000 paired subject-bootstrap resamples and two separate nine-test Holm families: endpoint comparisons against No reverse, Two forward, and BiTE; and REACT–reverse-only comparisons of earliest accuracy, nAUC, and endpoint accuracy. The default primary collector does not by itself perform all of these later tests.

These scripts accept study paths directly:

```bash
python anchor_source/fresh_primary_nauc.py "$REACT_STUDY"
python anchor_source/check_fresh_causality.py "$REACT_STUDY"
mkdir -p "$REACT_CONTROLS/figures"
MPLBACKEND=Agg python anchor_source/make_fresh_anytime_figure.py \
    "$REACT_STUDY" "$REACT_CONTROLS/figures/fig2.pdf"
```

`make_fresh_anytime_figure.py` plots REACT, No reverse, Two forward, and Forward mean. Its shading is **mean ±1 sample SD across subjects (`ddof=1`) after averaging seeds within subject**. Dotted lines mark chance accuracy. The `make_final_figure2*.py` scripts plot different later comparisons and are not interchangeable with this four-model figure.

The following scripts require manual path configuration before use; they do not accept generic `--study` arguments:

| Script | Path constants to review |
|---|---|
| `anchor_source/final_endpoint_stats.py` | `ORIG`, `OUT` |
| `anchor_source/summarize_reverse_only.py` | `ORIG`, `REV` |
| `anchor_source/summarize_paired_random_trunc_final.py` | `ORIG`, `RT`, `REV` |
| `anchor_source/make_final_figure2_publication.py` | `REV`, `CTRL`, output paths |

Update paths only in a working release copy and preserve the archived originals. The paired-duration summary also reads the independently trained reverse-only results; those result files are required even when the main comparison of interest is `react_rt` versus `compact_rt`.

## Manuscript results

**These are results reported in the supplied manuscript, not scores recomputed from the source export.** All accuracies are percentages. “Earliest” means 1 s for 2A/2B and 0.25 s for SD-SSVEP.

### Endpoint-trained REACT

| Dataset | Earliest accuracy | nAUC | Endpoint accuracy | Trainable parameters |
|---|---:|---:|---:|---:|
| BCICIV-2A | 69.32 | 80.12 | 85.06 | 18,916 |
| BCICIV-2B | 74.31 | 83.59 | 86.20 | 16,962 |
| SD-SSVEP | 34.89 | 64.15 | 96.50 | 20,140 |

REACT adds **3,136 parameters** relative to either single-reader control: 3,072 for the second reader and 64 for the gate.

### Matched random-duration training: REACT minus No reverse

| Metric | BCICIV-2A | BCICIV-2B | SD-SSVEP |
|---|---:|---:|---:|
| Earliest accuracy | +15.34 [10.13, 21.94] | +3.47 [0.31, 6.59] | +29.83 [23.11, 36.06] |
| nAUC | +10.42 [6.44, 15.21] | +0.22 [-1.17, 1.61] | +24.96 [19.17, 30.93] |
| Endpoint accuracy | +5.95 [3.60, 8.45] | -0.86 [-2.07, 0.14] | +18.56 [11.28, 26.50] |

Values are paired differences in percentage points with 95% bootstrap intervals. These comparisons use separately retrained random-duration models, not the endpoint-trained checkpoints in the first table.

The manuscript's mechanism analyses show that preserving input geometry through training-mean imputation substantially reduces the large direct-truncation deficit. Independently trained reverse-only models match or exceed fused REACT over much of the early trajectory, particularly for SD-SSVEP, while REACT has higher mean endpoint accuracy on 2A and SD-SSVEP. Endpoint differences are descriptive: none of the declared nine endpoint comparisons remains significant after Holm correction. The reported LOSO analysis does not establish improved subject-independent generalization.

## Reproducibility status

The core model's dimensions, parameter counts, direct observed-input inference, and synthetic software checks are supported by this export. Complete scientific reproduction needs additional artifacts and reconciliation:

| Item | Status in this snapshot |
|---|---|
| Core model and four matched readout configurations | Included |
| Primary training and prepared-data loader | Included; original machine paths require configuration |
| Reverse-only, paired-duration, anchor, and mean-imputation scripts | Included; original run artifacts are external prerequisites |
| Full checkpoints, scalers, per-trial predictions, and final per-run results | Not included |
| Executable raw-data preprocessing and SD-SSVEP split construction | Not included |
| LOSO training/evaluation workflow | Not included in the exported experiment code |
| Training workflow for the additional ten endpoint baselines | Not included in the exported primary runner |
| BiTE clipping exception described in the manuscript | Not implemented by the exported main trainer |
| Final figures and final manuscript source | Not included as a complete publication package |
| Locked final HPC environment | Not supplied; `validation/environment.json` describes earlier local software tests |

Selected textual summaries are present, but they are not substitutes for complete per-subject, per-seed records. In particular, `anchor_source/fresh_causality_maxima.txt` separates each model's discrepancies from the maximum over all four core models. Those cohort-wide maxima must not be relabeled as REACT-only maxima. Keep native script outputs and result provenance when reconciling the manuscript.

The source plan also records prior inspection of the benchmark results. A new fit does not make the benchmark a previously untouched test set. This repository does not claim that full benchmark reproduction was completed merely because the CPU tests pass.

## Limitations and attribution

The evaluated cohorts contain nine or ten subjects per dataset. Causality is limited to the prepared-input boundary; acquisition-to-output timing and controlled hardware latency were not established in the manuscript. Longer token sequences require appropriate positional capacity and reader receptive fields. Random-duration training was compared for REACT and No reverse, not independently trained reverse-only. Research outputs here are not a validated clinical or assistive-device deployment.

BiTE is an external baseline from [cindy-hong/BiteEEG](https://github.com/cindy-hong/BiteEEG). The adapter records commit **`924eb32241ba1a7c80dbc4ba097f8c979da17578`** in [`fresh/bite.py`](main_source/fresh/bite.py) and retrieves that revision separately; it does not vendor BiTE source in this export. Cite the original BiTE work and the original datasets when using those components.

Associated manuscript: **REACT-EEG: Reverse-Encoded Anytime Causal Temporal Modeling for EEG Decoding.** This snapshot does not provide finalized author/venue/DOI metadata, so no publication citation is fabricated here.

No project-wide `LICENSE` file is included in this export. Preserve the existing notices and third-party attribution; select and add the appropriate project license before advertising a licensed release. Review the original source paths, scheduler accounts, logs, and metadata for identifying or sensitive information before making the repository public.
