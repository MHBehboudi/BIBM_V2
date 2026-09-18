# Main table: within-subject accuracy against baselines RE-RUN here

Every model below was trained by this repository's harness on the official within-subject roles (session 1 train, session 2 test), 600 epochs, fixed final epoch, no validation set and no model selection, **3 seeds (2025-2027)**. Cells are mean ± SD across subjects (seeds averaged per subject) / Cohen's kappa. *published* is BiTE's own table (arXiv 2510.10004, one seed, their trainer).

Baselines are BiTE's released implementations loaded through BiTE's own `get_model`, unchanged, and trained with this repository's recipe (Adam 2e-3, cosine, label smoothing .1, batch 64, float32, gradient clip 5). BiTE itself is trained with `--clip 0`, as released. FACTNet cannot run on SD-SSVEP as released (its attention hard-codes 1000 samples).

| model | params (2a) | BCI IV-2a ours | pub. | BCI IV-2b ours | pub. | HGD ours | pub. | SD-SSVEP ours | pub. | corpus-bal. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ShallowConvNet | 46084 | 73.23 ± 11.87 / 0.64 | 74.11 | 82.27 ± 9.60 / 0.65 | 83.66 | 95.42 ± 3.11 / 0.94 | 95.33 | 31.67 ± 15.47 / 0.25 | 29.17 | 70.65 |
| DMSANet | 35884 | 76.80 ± 11.22 / 0.69 | 77.59 | 81.56 ± 10.09 / 0.63 | 84.65 | 96.11 ± 3.44 / 0.95 | 95.53 | 50.94 ± 22.46 / 0.46 | 52.00 | 76.35 |
| EEGNet | 3444 | 73.21 ± 9.69 / 0.64 | 71.14 | 83.21 ± 9.31 / 0.66 | 82.70 | 93.83 ± 3.27 / 0.92 | 93.97 | 63.17 ± 23.65 / 0.60 | 64.83 | 78.35 |
| EEGNeX | 56452 | 75.01 ± 9.56 / 0.67 | 75.62 | 81.00 ± 8.17 / 0.62 | 82.87 | 93.10 ± 4.11 / 0.91 | 93.55 | 68.61 ± 24.20 / 0.66 | 67.83 | 79.43 |
| EISATC | 26383 | 76.97 ± 10.07 / 0.69 | 78.13 | 83.66 ± 8.05 / 0.67 | 82.29 | 93.26 ± 3.65 / 0.91 | 92.95 | 67.78 ± 26.57 / 0.65 | 69.83 | 80.42 |
| EEGTCNet | 4096 | 76.62 ± 10.90 / 0.69 | 78.43 | 84.57 ± 7.86 / 0.69 | 82.56 | 94.94 ± 3.15 / 0.93 | 94.67 | 74.72 ± 23.24 / 0.72 | 73.33 | 82.71 |
| FACTNet | 5406 | 75.30 ± 11.57 / 0.67 | 75.23 | 83.37 ± 9.02 / 0.67 | 80.60 | 93.21 ± 3.42 / 0.91 | 93.62 | not run | 76.50 | — |
| ATCNet | 113732 | 81.29 ± 8.46 / 0.75 | 81.02 | 84.07 ± 8.81 / 0.68 | 84.25 | 95.67 ± 2.99 / 0.94 | 95.89 | 85.33 ± 17.57 / 0.84 | 84.50 | 86.59 |
| DeepConvNet | 102329 | 71.26 ± 15.23 / 0.62 | 72.38 | 84.95 ± 9.84 / 0.70 | 85.51 | 93.68 ± 3.23 / 0.92 | 93.08 | 96.50 ± 7.77 / 0.96 | 95.50 | 86.60 |
| MBCNNEATCFNet | 29520 | 81.76 ± 7.87 / 0.76 | 82.10 | 84.40 ± 7.91 / 0.69 | 84.79 | 93.72 ± 4.00 / 0.92 | 93.70 | 94.22 ± 7.29 / 0.94 | 95.33 | 88.53 |
| BiTE | 16294 | 84.10 ± 8.04 / 0.79 | 85.34 | 87.30 ± 7.22 / 0.75 | 88.37 | 95.53 ± 3.41 / 0.94 | 95.93 | 94.22 ± 8.69 / 0.94 | 94.16 | 90.29 |
| Compact (READER without the reversed branch) | 17828 | 82.33 ± 10.06 / 0.76 | — | 85.32 ± 8.06 / 0.71 | — | 95.69 ± 2.73 / 0.94 | — | 94.11 ± 11.13 / 0.94 | — | 89.36 |
| **READER** | 20964 | 84.71 ± 8.27 / 0.80 | — | 86.41 ± 7.30 / 0.73 | — | 96.30 ± 2.79 / 0.95 | — | 96.17 ± 7.89 / 0.96 | — | 90.90 |

## READER against the strongest re-run baseline on each corpus

Strongest = highest 3-seed mean among COMPLETE re-run baselines (choosing the maximum of eleven noisy means favours the baseline). Subject-level paired difference, 95% bootstrap CI, subject W/T/L, exact Wilcoxon p; Holm over READER vs all eleven baselines on that corpus once complete. Full per-subject detail: `results/model_zoo/MODEL_ZOO.md`.

| corpus | READER | strongest re-run baseline | READER minus it | READER minus BiTE (re-run) | READER rank |
|---|---:|---|---|---|---:|
| BCI IV-2a | 84.71 | BiTE 84.10 | +0.60 [-0.80, +1.94]  6/0/3  p 0.426 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.426 | +0.60 [-0.80, +1.94]  6/0/3  p 0.426 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.426 | 1 of 12 |
| BCI IV-2b | 86.41 | BiTE 87.30 | -0.89 [-1.84, +0.10]  2/0/7  p 0.137 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.516 | -0.89 [-1.84, +0.10]  2/0/7  p 0.137 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.516 | 2 of 12 |
| HGD | 96.30 | DMSANet 96.11 | +0.19 [-1.69, +1.79]  8/0/6  p 0.626 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.626 | +0.77 [+0.03, +1.60]  9/1/4  p 0.080 (exact sign-flip (mid-ranks), zeros 1), Holm p 0.241 | 1 of 12 |
| SD-SSVEP | 96.17 | DeepConvNet 96.50 | -0.33 [-1.61, +0.67]  3/4/3  p 1.000 (exact sign-flip (mid-ranks), zeros 4), Holm p 1.000 | +1.94 [+0.72, +3.39]  7/3/0  p 0.016 (exact sign-flip (mid-ranks), zeros 3), Holm p 0.047 | 2 of 11 |

## How closely the re-runs track the published numbers

BiTE's table is one seed (2025) with BiTE's own trainer. Our seed-2025 run of the same model, subject means averaged, minus the published value. Large gaps would mean the harness disadvantages a baseline.

| corpus | models compared | mean (ours − published) | mean absolute gap | largest gaps |
|---|---:|---:|---:|---|
| BCI IV-2a | 11 | -0.51 | 1.04 | DeepConvNet -3.32, EISATC -2.05, EEGNet +1.89 |
| BCI IV-2b | 11 | -0.27 | 1.08 | DMSANet -2.60, EEGTCNet +2.34, EEGNeX -1.39 |
| HGD | 11 | +0.09 | 0.39 | DeepConvNet +1.10, FACTNet -0.88, EISATC +0.39 |
| SD-SSVEP | 10 | +0.05 | 1.08 | ShallowConvNet +2.83, EEGNet -2.33, EISATC -1.67 |

## Published bars (single seed) for reference

| corpus | best published | held by | READER (3-seed) | READER minus bar |
|---|---:|---|---:|---:|
| BCI IV-2a | 85.34 | BiTE | 84.71 | -0.63 |
| BCI IV-2b | 88.37 | BiTE | 86.41 | -1.96 |
| HGD | 95.93 | BiTE | 96.30 | +0.37 |
| SD-SSVEP | 95.50 | DeepConvNet | 96.17 | +0.67 |

A published single-seed mean and a 3-seed mean are not a paired comparison; the re-run table above is the comparison of record.
