# Sensitivity of the reverse-reader advantage to its two main hyper-parameters

Every row is a PAIRED comparison: REACT and its forward-only control (no reverse reader, everything else identical, shared initialisation at the same seed) trained at the SAME setting, 3 seeds per subject, 600 epochs, fixed final epoch, official session roles.

nAUC is Eq. (nauc) integrated over the paper's range: 2a/2b 1.0-4.0 s, SD-SSVEP 0.25-1.0 s, on the exact-duration grid in samples (the same wall-clock durations for every cell, so pooling factors are comparable). Endpoint is full-trial accuracy. W/L counts subjects by the sign of the nAUC difference.

## Q1  Prefix-supervision weight (pooling at the corpus default)

The reverse reader is not what the prefix loss pays for: it is compared with its own control at each weight.

### BCICIV-2A  (p = 32, 32 tokens)

| lambda | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.70 | 49.98 | +30.72 [+23.62, +36.95] | 9/0 | 84.71 | 82.33 | +2.38 [+0.28, +4.63] |
| 0.1 | 82.41 | 66.39 | +16.02 [+10.57, +21.70] | 9/0 | 84.39 | 82.79 | +1.59 [-0.17, +3.32] |
| 0.3 | 82.30 | 72.99 | +9.31 [+5.98, +13.23] | 9/0 | 83.50 | 82.10 | +1.40 [-0.18, +3.14] |
| 1 | 82.26 | 72.40 | +9.86 [+5.46, +16.06] | 9/0 | 83.67 | 77.39 | +6.28 [+1.57, +13.22] |

### BCICIV-2B  (p = 32, 32 tokens)

| lambda | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 82.89 | 71.83 | +11.06 [+6.41, +16.18] | 9/0 | 86.41 | 85.32 | +1.09 [+0.41, +1.81] |
| 0.1 | 83.65 | 77.29 | +6.36 [+2.51, +10.88] | 7/2 | 85.07 | 85.92 | -0.85 [-1.91, +0.17] |
| 0.3 | 83.51 | 82.38 | +1.13 [-1.62, +4.09] | 5/4 | 85.59 | 86.49 | -0.90 [-1.62, -0.25] |
| 1 | 83.13 | 84.66 | -1.53 [-3.64, +0.21] | 3/6 | 85.55 | 86.59 | -1.03 [-2.15, +0.09] |

### SD-SSVEP  (p = 4, 64 tokens)

| lambda | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64.19 | 21.76 | +42.43 [+30.99, +52.50] | 10/0 | 96.17 | 94.11 | +2.06 [+0.33, +4.33] |
| 0.1 | 83.18 | 31.63 | +51.56 [+44.64, +57.94] | 10/0 | 96.44 | 94.28 | +2.17 [-0.06, +5.22] |
| 0.3 | 84.56 | 72.27 | +12.30 [+6.97, +18.12] | 10/0 | 94.33 | 94.11 | +0.22 [-1.72, +2.50] |
| 1 | 82.90 | 77.25 | +5.65 [+1.58, +9.66] | 9/1 | 89.72 | 90.50 | -0.78 [-5.94, +3.78] |

## Q2  Token resolution (prefix-supervision weight fixed)

p sets the token duration and the token count; all three settings stay inside the 71-token receptive field of the TCN, so no condition is limited by history it cannot reach.

### BCICIV-2A, lambda = 0

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 75.40 | 47.62 | +27.78 [+22.42, +33.05] | 9/0 | 82.12 | 81.56 | +0.57 [-0.87, +1.66] |
| 32 (32 tokens, 128.0 ms) | 80.70 | 49.98 | +30.72 [+23.62, +36.95] | 9/0 | 84.71 | 82.33 | +2.38 [+0.28, +4.63] |
| 16 (63 tokens, 64.0 ms) | 81.36 | 46.09 | +35.27 [+28.95, +41.57] | 9/0 | 84.92 | 70.07 | +14.84 [+8.42, +21.46] |

### BCICIV-2A, lambda = 0.3

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 80.74 | 72.27 | +8.47 [+4.93, +12.94] | 9/0 | 83.17 | 81.42 | +1.75 [-0.90, +5.38] |
| 32 (32 tokens, 128.0 ms) | 82.30 | 72.99 | +9.31 [+5.98, +13.23] | 9/0 | 83.50 | 82.10 | +1.40 [-0.18, +3.14] |
| 16 (63 tokens, 64.0 ms) | 82.99 | 68.33 | +14.66 [+8.12, +22.10] | 9/0 | 84.94 | 73.77 | +11.18 [+5.14, +18.07] |

### BCICIV-2B, lambda = 0

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 82.46 | 70.61 | +11.85 [+7.91, +16.86] | 9/0 | 86.19 | 85.78 | +0.41 [-0.20, +1.08] |
| 32 (32 tokens, 128.0 ms) | 82.89 | 71.83 | +11.06 [+6.41, +16.18] | 9/0 | 86.41 | 85.32 | +1.09 [+0.41, +1.81] |
| 16 (63 tokens, 64.0 ms) | 82.29 | 66.44 | +15.85 [+10.01, +21.81] | 9/0 | 85.44 | 83.90 | +1.53 [+0.36, +2.61] |

### BCICIV-2B, lambda = 0.3

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 84.07 | 82.94 | +1.13 [-0.35, +2.83] | 5/4 | 86.10 | 86.64 | -0.53 [-1.53, +0.23] |
| 32 (32 tokens, 128.0 ms) | 83.51 | 82.38 | +1.13 [-1.62, +4.09] | 5/4 | 85.59 | 86.49 | -0.90 [-1.62, -0.25] |
| 16 (63 tokens, 64.0 ms) | 83.41 | 80.66 | +2.75 [-1.12, +6.98] | 6/3 | 86.08 | 85.53 | +0.55 [-0.58, +1.66] |

### SD-SSVEP, lambda = 0

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 (16 tokens, 62.5 ms) | 60.45 | 23.80 | +36.65 [+29.90, +42.70] | 10/0 | 87.28 | 84.44 | +2.83 [+0.89, +4.78] |
| 8 (32 tokens, 31.2 ms) | 66.03 | 24.59 | +41.44 [+33.05, +49.15] | 10/0 | 94.72 | 94.11 | +0.61 [-0.89, +2.06] |
| 4 (64 tokens, 15.6 ms) | 64.19 | 21.76 | +42.43 [+30.99, +52.50] | 10/0 | 96.17 | 94.11 | +2.06 [+0.33, +4.33] |

### SD-SSVEP, lambda = 0.3

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 (16 tokens, 62.5 ms) | 76.19 | 62.13 | +14.06 [+11.04, +17.47] | 10/0 | 86.67 | 84.89 | +1.78 [-0.06, +3.44] |
| 8 (32 tokens, 31.2 ms) | 82.45 | 72.42 | +10.03 [+5.93, +14.31] | 10/0 | 93.17 | 94.28 | -1.11 [-2.50, +0.06] |
| 4 (64 tokens, 15.6 ms) | 84.56 | 72.27 | +12.30 [+6.97, +18.12] | 10/0 | 94.33 | 94.11 | +0.22 [-1.72, +2.50] |

## Cells

| corpus | arm | lambda | p | runs | GPUs |
|---|---|---:|---:|---:|---|
| 2a | compact | 0 | 16 | 27 | NVIDIA H100 80GB HBM3 |
| 2a | compact | 0 | 32 | 27 | NVIDIA H100 80GB HBM3/NVIDIA H100 NVL |
| 2a | compact | 0 | 64 | 27 | NVIDIA H100 80GB HBM3 |
| 2a | compact | 0.1 | 32 | 27 | NVIDIA H200 NVL |
| 2a | compact | 0.3 | 16 | 27 | NVIDIA H100 80GB HBM3/NVIDIA H200 NVL |
| 2a | compact | 0.3 | 32 | 27 | NVIDIA H200 NVL |
| 2a | compact | 0.3 | 64 | 27 | NVIDIA H100 80GB HBM3 |
| 2a | compact | 1 | 32 | 27 | NVIDIA H100 NVL |
| 2a | reader | 0 | 16 | 27 | NVIDIA H200 NVL |
| 2a | reader | 0 | 32 | 27 | NVIDIA H100 80GB HBM3 |
| 2a | reader | 0 | 64 | 27 | NVIDIA H100 80GB HBM3 |
| 2a | reader | 0.1 | 32 | 27 | NVIDIA H200 NVL |
| 2a | reader | 0.3 | 16 | 27 | NVIDIA H200 NVL |
| 2a | reader | 0.3 | 32 | 27 | NVIDIA H200 NVL |
| 2a | reader | 0.3 | 64 | 27 | NVIDIA H100 80GB HBM3 |
| 2a | reader | 1 | 32 | 27 | NVIDIA H100 NVL |
| 2b | compact | 0 | 16 | 27 | NVIDIA H100 80GB HBM3 |
| 2b | compact | 0 | 32 | 27 | NVIDIA H100 80GB HBM3 |
| 2b | compact | 0 | 64 | 27 | NVIDIA H100 80GB HBM3 |
| 2b | compact | 0.1 | 32 | 27 | NVIDIA H200 NVL |
| 2b | compact | 0.3 | 16 | 27 | NVIDIA H100 80GB HBM3 |
| 2b | compact | 0.3 | 32 | 27 | NVIDIA H200 NVL |
| 2b | compact | 0.3 | 64 | 27 | NVIDIA H100 80GB HBM3/NVIDIA H200 NVL |
| 2b | compact | 1 | 32 | 27 | NVIDIA H200 NVL |
| 2b | reader | 0 | 16 | 27 | NVIDIA H200 NVL |
| 2b | reader | 0 | 32 | 27 | NVIDIA H100 80GB HBM3 |
| 2b | reader | 0 | 64 | 27 | NVIDIA H100 80GB HBM3 |
| 2b | reader | 0.1 | 32 | 27 | NVIDIA H200 NVL |
| 2b | reader | 0.3 | 16 | 27 | NVIDIA H200 NVL |
| 2b | reader | 0.3 | 32 | 27 | NVIDIA H100 80GB HBM3/NVIDIA H200 NVL |
| 2b | reader | 0.3 | 64 | 27 | NVIDIA H200 NVL |
| 2b | reader | 1 | 32 | 27 | NVIDIA H100 NVL/NVIDIA H200 NVL |
| sdssvep | compact | 0 | 4 | 30 | NVIDIA H100 80GB HBM3 |
| sdssvep | compact | 0 | 8 | 30 | NVIDIA H100 80GB HBM3 |
| sdssvep | compact | 0 | 16 | 30 | NVIDIA H100 80GB HBM3 |
| sdssvep | compact | 0.1 | 4 | 30 | NVIDIA H100 NVL/NVIDIA H200 NVL |
| sdssvep | compact | 0.3 | 4 | 30 | NVIDIA H200 NVL |
| sdssvep | compact | 0.3 | 8 | 30 | NVIDIA H100 80GB HBM3 |
| sdssvep | compact | 0.3 | 16 | 30 | NVIDIA H100 80GB HBM3/NVIDIA H200 NVL |
| sdssvep | compact | 1 | 4 | 30 | NVIDIA H200 NVL |
| sdssvep | reader | 0 | 4 | 30 | NVIDIA H100 80GB HBM3 |
| sdssvep | reader | 0 | 8 | 30 | NVIDIA H200 NVL |
| sdssvep | reader | 0 | 16 | 30 | NVIDIA H100 80GB HBM3 |
| sdssvep | reader | 0.1 | 4 | 30 | NVIDIA H200 NVL |
| sdssvep | reader | 0.3 | 4 | 30 | NVIDIA H100 NVL/NVIDIA H200 NVL |
| sdssvep | reader | 0.3 | 8 | 30 | NVIDIA H200 NVL |
| sdssvep | reader | 0.3 | 16 | 30 | NVIDIA H100 80GB HBM3/NVIDIA H200 NVL |
| sdssvep | reader | 1 | 4 | 30 | NVIDIA H200 NVL |

1344 runs in 48 cells.
