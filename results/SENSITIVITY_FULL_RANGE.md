# Sensitivity of the reverse-reader advantage to its two main hyper-parameters

Every row is a PAIRED comparison: REACT and its forward-only control (no reverse reader, everything else identical, shared initialisation at the same seed) trained at the SAME setting, 3 seeds per subject, 600 epochs, fixed final epoch, official session roles.

nAUC is Eq. (nauc) integrated over the whole evaluated grid: 2a/2b 0.5-4.0 s, SD-SSVEP 0.125-1.0 s, on the exact-duration grid in samples (the same wall-clock durations for every cell, so pooling factors are comparable). Endpoint is full-trial accuracy. W/L counts subjects by the sign of the nAUC difference.

## Q1  Prefix-supervision weight (pooling at the corpus default)

The reverse reader is not what the prefix loss pays for: it is compared with its own control at each weight.

### BCICIV-2A  (p = 32, 32 tokens)

| lambda | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.68 | 47.03 | +30.65 [+24.36, +36.10] | 9/0 | 84.71 | 82.33 | +2.38 [+0.28, +4.63] |
| 0.1 | 79.79 | 62.06 | +17.73 [+12.61, +23.01] | 9/0 | 84.39 | 82.79 | +1.59 [-0.17, +3.32] |
| 0.3 | 80.01 | 69.13 | +10.88 [+7.67, +14.71] | 9/0 | 83.50 | 82.10 | +1.40 [-0.18, +3.14] |
| 1 | 80.16 | 69.78 | +10.37 [+6.53, +15.82] | 9/0 | 83.67 | 77.39 | +6.28 [+1.57, +13.22] |

### BCICIV-2B  (p = 32, 32 tokens)

| lambda | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.14 | 68.99 | +11.15 [+6.68, +15.99] | 9/0 | 86.41 | 85.32 | +1.09 [+0.41, +1.81] |
| 0.1 | 81.28 | 74.29 | +6.98 [+3.12, +11.24] | 7/2 | 85.07 | 85.92 | -0.85 [-1.91, +0.17] |
| 0.3 | 81.26 | 79.49 | +1.77 [-1.12, +4.73] | 5/4 | 85.59 | 86.49 | -0.90 [-1.62, -0.25] |
| 1 | 81.07 | 82.47 | -1.40 [-3.59, +0.27] | 3/6 | 85.55 | 86.59 | -1.03 [-2.15, +0.09] |

### SD-SSVEP  (p = 4, 64 tokens)

| lambda | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 59.08 | 20.10 | +38.98 [+28.30, +48.50] | 10/0 | 96.17 | 94.11 | +2.06 [+0.33, +4.33] |
| 0.1 | 77.53 | 28.80 | +48.73 [+41.59, +55.38] | 10/0 | 96.44 | 94.28 | +2.17 [-0.06, +5.22] |
| 0.3 | 79.77 | 66.62 | +13.15 [+8.22, +18.58] | 10/0 | 94.33 | 94.11 | +0.22 [-1.72, +2.50] |
| 1 | 78.83 | 72.52 | +6.31 [+2.46, +9.96] | 9/1 | 89.72 | 90.50 | -0.78 [-5.94, +3.78] |

## Q2  Token resolution (prefix-supervision weight fixed)

p sets the token duration and the token count; all three settings stay inside the 71-token receptive field of the TCN, so no condition is limited by history it cannot reach.

### BCICIV-2A, lambda = 0

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 71.46 | 44.94 | +26.53 [+21.49, +31.50] | 9/0 | 82.12 | 81.56 | +0.57 [-0.87, +1.66] |
| 32 (32 tokens, 128.0 ms) | 77.68 | 47.03 | +30.65 [+24.36, +36.10] | 9/0 | 84.71 | 82.33 | +2.38 [+0.28, +4.63] |
| 16 (63 tokens, 64.0 ms) | 78.45 | 43.65 | +34.81 [+28.93, +40.67] | 9/0 | 84.92 | 70.07 | +14.84 [+8.42, +21.46] |

### BCICIV-2A, lambda = 0.3

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 77.75 | 68.66 | +9.09 [+5.79, +13.39] | 9/0 | 83.17 | 81.42 | +1.75 [-0.90, +5.38] |
| 32 (32 tokens, 128.0 ms) | 80.01 | 69.13 | +10.88 [+7.67, +14.71] | 9/0 | 83.50 | 82.10 | +1.40 [-0.18, +3.14] |
| 16 (63 tokens, 64.0 ms) | 80.96 | 64.48 | +16.48 [+10.30, +23.54] | 9/0 | 84.94 | 73.77 | +11.18 [+5.14, +18.07] |

### BCICIV-2B, lambda = 0

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 79.48 | 68.05 | +11.43 [+7.50, +16.17] | 9/0 | 86.19 | 85.78 | +0.41 [-0.20, +1.08] |
| 32 (32 tokens, 128.0 ms) | 80.14 | 68.99 | +11.15 [+6.68, +15.99] | 9/0 | 86.41 | 85.32 | +1.09 [+0.41, +1.81] |
| 16 (63 tokens, 64.0 ms) | 79.92 | 64.26 | +15.65 [+9.94, +21.55] | 9/0 | 85.44 | 83.90 | +1.53 [+0.36, +2.61] |

### BCICIV-2B, lambda = 0.3

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 (16 tokens, 256.0 ms) | 81.81 | 79.97 | +1.84 [+0.11, +3.59] | 7/2 | 86.10 | 86.64 | -0.53 [-1.53, +0.23] |
| 32 (32 tokens, 128.0 ms) | 81.26 | 79.49 | +1.77 [-1.12, +4.73] | 5/4 | 85.59 | 86.49 | -0.90 [-1.62, -0.25] |
| 16 (63 tokens, 64.0 ms) | 81.43 | 77.64 | +3.79 [-0.25, +7.98] | 6/3 | 86.08 | 85.53 | +0.55 [-0.58, +1.66] |

### SD-SSVEP, lambda = 0

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 (16 tokens, 62.5 ms) | 53.44 | 21.75 | +31.69 [+25.84, +36.98] | 10/0 | 87.28 | 84.44 | +2.83 [+0.89, +4.78] |
| 8 (32 tokens, 31.2 ms) | 59.27 | 22.40 | +36.87 [+29.32, +43.89] | 10/0 | 94.72 | 94.11 | +0.61 [-0.89, +2.06] |
| 4 (64 tokens, 15.6 ms) | 59.08 | 20.10 | +38.98 [+28.30, +48.50] | 10/0 | 96.17 | 94.11 | +2.06 [+0.33, +4.33] |

### SD-SSVEP, lambda = 0.3

| pooling p | REACT nAUC | no-reverse nAUC | dnAUC [95% CI] | subj W/L | REACT endpoint | no-reverse endpoint | dendpoint [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 (16 tokens, 62.5 ms) | 69.91 | 56.82 | +13.09 [+10.45, +16.11] | 10/0 | 86.67 | 84.89 | +1.78 [-0.06, +3.44] |
| 8 (32 tokens, 31.2 ms) | 76.65 | 66.47 | +10.17 [+6.33, +14.23] | 10/0 | 93.17 | 94.28 | -1.11 [-2.50, +0.06] |
| 4 (64 tokens, 15.6 ms) | 79.77 | 66.62 | +13.15 [+8.22, +18.58] | 10/0 | 94.33 | 94.11 | +0.22 [-1.72, +2.50] |

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
