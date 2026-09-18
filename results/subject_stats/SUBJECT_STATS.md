# Subject-level statistics (READER paper)

## Endpoint accuracy: mean +/- SD across subject-level seed means

| corpus | reader | compact | bite | ff_control | compact_mean |
|---|---:|---:|---:|---:|---:|
| 2a | 84.71 +/- 8.27 (n=9, runs=27) | 82.33 +/- 10.06 (n=9, runs=27) | 84.10 +/- 8.04 (n=9, runs=27) | 83.08 +/- 9.25 (n=9, runs=27) | 72.36 +/- 16.85 (n=9, runs=27) |
| 2b | 86.41 +/- 7.30 (n=9, runs=27) | 85.32 +/- 8.06 (n=9, runs=27) | 87.30 +/- 7.22 (n=9, runs=27) | 85.15 +/- 8.19 (n=9, runs=27) | 85.05 +/- 9.31 (n=9, runs=27) |
| sdssvep | 96.17 +/- 7.89 (n=10, runs=30) | 94.11 +/- 11.13 (n=10, runs=30) | 94.22 +/- 8.69 (n=10, runs=30) | 94.17 +/- 11.11 (n=10, runs=30) | 66.44 +/- 23.62 (n=10, runs=30) |

## Endpoint: READER minus comparator, subject-level (family of 12, Holm applied)

mean d [95% bootstrap CI]  subject W/T/L  Wilcoxon p

| corpus | vs compact | vs ff_control | vs compact_mean | vs bite |
|---|---|---|---|---|
| 2a | +2.38 [+0.23, +4.63]  6/0/3  p 0.098 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.488 | +1.63 [+0.27, +3.07]  6/0/3  p 0.105 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.488 | +12.35 [+6.94, +18.58]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 | +0.60 [-0.81, +1.93]  6/0/3  p 0.426 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.852 |
| 2b | +1.09 [+0.39, +1.79]  7/0/2  p 0.031 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.281 | +1.26 [+0.34, +2.19]  8/0/1  p 0.055 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.383 | +1.36 [-0.79, +3.57]  6/0/3  p 0.438 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.852 | -0.89 [-1.86, +0.09]  2/0/7  p 0.137 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.488 |
| sdssvep | +2.06 [+0.33, +4.39]  5/5/0  p 0.062 (exact sign-flip (mid-ranks), zeros 5), Holm p 0.383 | +2.00 [+0.50, +4.17]  6/4/0  p 0.031 (exact sign-flip (mid-ranks), zeros 4), Holm p 0.281 | +29.72 [+19.00, +41.06]  10/0/0  p 0.002 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.023 | +1.94 [+0.78, +3.44]  7/3/0  p 0.016 (exact sign-flip (mid-ranks), zeros 3), Holm p 0.156 |

## Exact-duration, same-checkpoint accuracy: 2a (runs: READER 27, compact 27, ff_control 27, compact_mean 27, bite 27)

| samples | seconds | READER | compact | ff_control | compact_mean | bite |
|---:|---:|---:|---:|---:|---:|---:|
| 125 | 0.500 | 49.00 | 27.70 | 27.79 | 28.25 | 40.81 |
| 250 | 1.000 | 70.10 | 30.93 | 30.25 | 40.44 | 60.26 |
| 375 | 1.500 | 78.37 | 38.54 | 39.35 | 57.47 | 74.25 |
| 500 | 2.000 | 82.05 | 45.10 | 50.82 | 65.24 | 76.79 |
| 625 | 2.500 | 82.32 | 50.33 | 58.80 | 69.14 | 76.83 |
| 750 | 3.000 | 81.98 | 50.35 | 58.40 | 71.09 | 70.32 |
| 875 | 3.500 | 82.07 | 58.91 | 64.02 | 71.89 | 64.44 |
| 1000 | 4.000 | 84.71 | 82.33 | 83.08 | 72.36 | 84.12 |

READER minus compact (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +21.30 | [+15.28, +27.15] |
| 250 | +39.17 | [+34.66, +44.47] |
| 375 | +39.83 | [+32.55, +46.76] |
| 500 | +36.95 | [+26.29, +46.49] |
| 625 | +31.98 | [+21.86, +40.61] |
| 750 | +31.64 | [+23.65, +38.89] |
| 875 | +23.16 | [+14.39, +32.38] |
| 1000 | +2.38 | [+0.24, +4.64] |

READER minus ff_control (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +21.21 | [+15.70, +26.86] |
| 250 | +39.85 | [+34.57, +45.40] |
| 375 | +39.02 | [+31.65, +46.35] |
| 500 | +31.22 | [+21.33, +39.76] |
| 625 | +23.52 | [+15.21, +30.94] |
| 750 | +23.59 | [+15.97, +30.79] |
| 875 | +18.06 | [+11.09, +24.99] |
| 1000 | +1.63 | [+0.30, +3.06] |

READER minus compact_mean (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +20.74 | [+15.97, +25.34] |
| 250 | +29.66 | [+26.49, +32.77] |
| 375 | +20.90 | [+16.60, +26.11] |
| 500 | +16.81 | [+11.99, +22.47] |
| 625 | +13.18 | [+8.78, +17.89] |
| 750 | +10.89 | [+6.91, +14.94] |
| 875 | +10.19 | [+6.03, +14.48] |
| 1000 | +12.35 | [+6.88, +18.69] |

READER minus bite (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +8.19 | [+3.05, +12.58] |
| 250 | +9.84 | [+6.46, +13.08] |
| 375 | +4.12 | [+2.02, +6.47] |
| 500 | +5.26 | [+3.43, +7.27] |
| 625 | +5.49 | [+3.24, +8.20] |
| 750 | +11.66 | [+8.65, +15.16] |
| 875 | +17.63 | [+13.26, +22.61] |
| 1000 | +0.59 | [-0.76, +1.90] |

Gate interventions on trained READER (2a): accuracy at forced g minus learned-gate accuracy, subject-level mean [95% CI]

| samples | g=0 (forward only) | g=0.5 | g=1 (reversed only) |
|---:|---|---|---|
| 125 | -21.62 [-27.25, -15.92] | +0.12 [+0.01, +0.23] | +2.04 [+0.96, +3.20] |
| 250 | -37.17 [-42.41, -32.03] | +0.19 [+0.01, +0.37] | +1.38 [+0.89, +1.84] |
| 375 | -33.92 [-39.74, -26.98] | +0.18 [+0.04, +0.32] | +0.42 [-0.04, +0.90] |
| 500 | -26.94 [-33.44, -20.10] | +0.05 [-0.03, +0.13] | +0.06 [-0.53, +0.66] |
| 625 | -20.79 [-26.71, -14.21] | +0.22 [+0.04, +0.44] | -0.04 [-0.63, +0.69] |
| 750 | -22.25 [-28.46, -15.55] | +0.08 [-0.06, +0.24] | +0.67 [-0.08, +1.56] |
| 875 | -15.72 [-22.45, -9.34] | +0.12 [-0.01, +0.27] | +0.84 [-0.30, +2.19] |
| 1000 | -0.60 [-1.43, +0.23] | -0.06 [-0.14, +0.01] | -2.12 [-3.67, -1.05] |

READER (one endpoint-trained checkpoint, exact duration) minus separately retrained specialists (2a); the full-duration specialist is the cohort run

| family | samples | specialist mean | READER - specialist |
|---|---:|---:|---|
| bite | 250 | 74.42 | -4.32 [-5.71, -2.66]  1/0/8  p 0.008 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 500 | 81.37 | +0.68 [-0.59, +1.98]  5/0/4  p 0.426 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 750 | 83.02 | -1.04 [-4.59, +1.13]  5/0/4  p 0.652 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 1000 | 84.10 | +0.60 [-0.80, +1.90]  6/0/3  p 0.426 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 250 | 74.83 | -4.73 [-7.19, -2.49]  0/0/9  p 0.004 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 500 | 81.57 | +0.48 [-0.82, +1.76]  5/0/4  p 0.477 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 750 | 83.35 | -1.36 [-5.23, +1.52]  5/0/4  p 1.000 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 1000 | 82.33 | +2.38 [+0.24, +4.62]  6/0/3  p 0.098 (exact sign-flip (mid-ranks), zeros 0) |

## Exact-duration, same-checkpoint accuracy: 2b (runs: READER 27, compact 27, ff_control 27, compact_mean 27, bite 27)

| samples | seconds | READER | compact | ff_control | compact_mean | bite |
|---:|---:|---:|---:|---:|---:|---:|
| 125 | 0.500 | 56.52 | 49.66 | 49.28 | 50.42 | 53.08 |
| 250 | 1.000 | 70.86 | 54.33 | 54.57 | 62.67 | 68.16 |
| 375 | 1.500 | 79.73 | 69.32 | 67.05 | 77.58 | 80.25 |
| 500 | 2.000 | 82.99 | 71.55 | 70.11 | 81.74 | 83.00 |
| 625 | 2.500 | 84.68 | 71.47 | 72.81 | 83.25 | 83.19 |
| 750 | 3.000 | 85.41 | 73.82 | 75.73 | 84.19 | 82.18 |
| 875 | 3.500 | 85.86 | 74.97 | 77.52 | 84.69 | 80.18 |
| 1000 | 4.000 | 86.41 | 85.32 | 85.15 | 85.05 | 87.30 |

READER minus compact (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +6.86 | [+1.16, +12.44] |
| 250 | +16.53 | [+9.51, +23.44] |
| 375 | +10.41 | [+3.56, +15.86] |
| 500 | +11.45 | [+5.20, +17.41] |
| 625 | +13.21 | [+7.45, +20.10] |
| 750 | +11.59 | [+6.23, +18.66] |
| 875 | +10.89 | [+5.08, +17.84] |
| 1000 | +1.09 | [+0.39, +1.81] |

READER minus ff_control (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +7.24 | [+2.77, +12.07] |
| 250 | +16.29 | [+10.28, +22.17] |
| 375 | +12.68 | [+5.09, +20.20] |
| 500 | +12.88 | [+6.15, +18.44] |
| 625 | +11.87 | [+7.45, +16.73] |
| 750 | +9.68 | [+6.54, +13.93] |
| 875 | +8.35 | [+2.84, +14.43] |
| 1000 | +1.26 | [+0.34, +2.16] |

READER minus compact_mean (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +6.11 | [+1.91, +10.52] |
| 250 | +8.20 | [+3.33, +12.87] |
| 375 | +2.15 | [-2.69, +5.20] |
| 500 | +1.26 | [-2.67, +4.13] |
| 625 | +1.43 | [-1.14, +3.69] |
| 750 | +1.22 | [-1.30, +3.64] |
| 875 | +1.18 | [-0.76, +3.01] |
| 1000 | +1.36 | [-0.80, +3.65] |

READER minus bite (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 125 | +3.45 | [-0.41, +7.71] |
| 250 | +2.70 | [-2.35, +7.64] |
| 375 | -0.52 | [-3.15, +1.30] |
| 500 | -0.01 | [-2.37, +2.26] |
| 625 | +1.49 | [-0.13, +3.46] |
| 750 | +3.23 | [+1.46, +5.10] |
| 875 | +5.68 | [+2.27, +9.38] |
| 1000 | -0.89 | [-1.85, +0.09] |

Gate interventions on trained READER (2b): accuracy at forced g minus learned-gate accuracy, subject-level mean [95% CI]

| samples | g=0 (forward only) | g=0.5 | g=1 (reversed only) |
|---:|---|---|---|
| 125 | -7.12 [-11.47, -3.08] | +0.15 [+0.08, +0.24] | +2.44 [+1.13, +4.05] |
| 250 | -14.99 [-19.11, -10.52] | -0.05 [-0.09, -0.00] | +1.12 [-0.44, +2.99] |
| 375 | -11.21 [-17.93, -5.23] | +0.10 [+0.02, +0.17] | +0.11 [-0.54, +0.95] |
| 500 | -11.22 [-17.14, -5.78] | +0.06 [-0.01, +0.14] | -0.42 [-0.74, -0.13] |
| 625 | -11.50 [-17.07, -6.25] | +0.03 [-0.01, +0.08] | -0.18 [-0.98, +0.41] |
| 750 | -8.52 [-13.65, -4.22] | +0.04 [-0.01, +0.08] | -0.45 [-0.97, +0.14] |
| 875 | -7.13 [-12.12, -2.98] | +0.02 [-0.08, +0.14] | -0.35 [-1.04, +0.19] |
| 1000 | -0.70 [-1.15, -0.22] | +0.02 [-0.11, +0.15] | -0.77 [-1.58, -0.07] |

READER (one endpoint-trained checkpoint, exact duration) minus separately retrained specialists (2b); the full-duration specialist is the cohort run

| family | samples | specialist mean | READER - specialist |
|---|---:|---:|---|
| bite | 250 | 77.37 | -6.51 [-11.86, -2.48]  2/0/7  p 0.020 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 500 | 85.15 | -2.16 [-4.55, -0.37]  2/0/7  p 0.059 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 750 | 86.79 | -1.38 [-2.08, -0.59]  2/0/7  p 0.020 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 1000 | 87.30 | -0.89 [-1.85, +0.11]  2/0/7  p 0.137 (exact sign-flip (mid-ranks), zeros 0) |

## Exact-duration, same-checkpoint accuracy: sdssvep (runs: READER 30, compact 30, ff_control 30, compact_mean 30, bite 30)

| samples | seconds | READER | compact | ff_control | compact_mean | bite |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.125 | 22.22 | 10.28 | 11.28 | 16.39 | 13.44 |
| 64 | 0.250 | 34.61 | 9.94 | 10.89 | 28.28 | 18.78 |
| 96 | 0.375 | 47.89 | 14.61 | 13.50 | 38.33 | 28.17 |
| 128 | 0.500 | 62.00 | 13.78 | 16.67 | 48.56 | 29.39 |
| 160 | 0.625 | 66.50 | 19.28 | 19.67 | 56.83 | 33.17 |
| 192 | 0.750 | 69.89 | 14.06 | 16.11 | 60.94 | 32.89 |
| 224 | 0.875 | 73.50 | 16.83 | 18.67 | 63.50 | 32.94 |
| 256 | 1.000 | 96.17 | 94.11 | 94.17 | 66.44 | 94.22 |

READER minus compact (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 32 | +11.94 | [+7.22, +16.44] |
| 64 | +24.67 | [+14.28, +34.56] |
| 96 | +33.28 | [+20.56, +45.22] |
| 128 | +48.22 | [+33.72, +61.22] |
| 160 | +47.22 | [+34.33, +58.94] |
| 192 | +55.83 | [+41.89, +68.28] |
| 224 | +56.67 | [+44.83, +66.83] |
| 256 | +2.06 | [+0.33, +4.39] |

READER minus ff_control (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 32 | +10.94 | [+6.00, +15.61] |
| 64 | +23.72 | [+13.11, +33.83] |
| 96 | +34.39 | [+22.11, +45.78] |
| 128 | +45.33 | [+31.83, +57.67] |
| 160 | +46.83 | [+33.67, +59.00] |
| 192 | +53.78 | [+39.94, +65.61] |
| 224 | +54.83 | [+43.06, +64.50] |
| 256 | +2.00 | [+0.50, +4.11] |

READER minus compact_mean (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 32 | +5.83 | [+1.22, +10.78] |
| 64 | +6.33 | [+1.00, +12.44] |
| 96 | +9.56 | [+1.50, +17.89] |
| 128 | +13.44 | [+4.44, +22.56] |
| 160 | +9.67 | [-0.28, +18.78] |
| 192 | +8.94 | [-0.72, +18.44] |
| 224 | +10.00 | [+1.33, +18.83] |
| 256 | +29.72 | [+19.17, +41.11] |

READER minus bite (same checkpoints, subject-level, pointwise 95% CI from whole-curve bootstrap):

| samples | mean d | 95% CI |
|---:|---:|---|
| 32 | +8.78 | [+5.05, +12.83] |
| 64 | +15.83 | [+8.05, +23.33] |
| 96 | +19.72 | [+9.94, +29.06] |
| 128 | +32.61 | [+21.28, +43.50] |
| 160 | +33.33 | [+23.22, +42.00] |
| 192 | +37.00 | [+26.11, +47.00] |
| 224 | +40.56 | [+29.61, +49.00] |
| 256 | +1.94 | [+0.78, +3.39] |

Gate interventions on trained READER (sdssvep): accuracy at forced g minus learned-gate accuracy, subject-level mean [95% CI]

| samples | g=0 (forward only) | g=0.5 | g=1 (reversed only) |
|---:|---|---|---|
| 32 | -10.83 [-14.94, -6.50] | +0.11 [-0.39, +0.72] | +2.06 [+0.39, +3.94] |
| 64 | -24.06 [-33.50, -14.61] | +0.50 [-0.33, +1.28] | +4.89 [+2.28, +7.61] |
| 96 | -33.61 [-45.61, -21.39] | +2.61 [+1.72, +3.56] | +10.94 [+8.00, +13.44] |
| 128 | -44.11 [-55.22, -31.72] | +2.89 [+1.56, +4.17] | +8.06 [+3.67, +12.44] |
| 160 | -43.78 [-54.83, -31.39] | +2.83 [+1.44, +4.39] | +9.22 [+4.39, +15.17] |
| 192 | -50.44 [-61.33, -37.22] | +3.72 [+2.44, +5.17] | +11.67 [+5.39, +19.17] |
| 224 | -50.28 [-58.61, -40.06] | +4.67 [+3.06, +6.33] | +11.33 [+5.67, +17.89] |
| 256 | -1.83 [-3.56, -0.44] | +0.11 [-0.11, +0.39] | -8.83 [-13.95, -4.28] |

READER (one endpoint-trained checkpoint, exact duration) minus separately retrained specialists (sdssvep); the full-duration specialist is the cohort run

| family | samples | specialist mean | READER - specialist |
|---|---:|---:|---|
| bite | 64 | 66.11 | -31.50 [-37.56, -24.39]  0/0/10  p 0.002 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 128 | 84.11 | -22.11 [-30.61, -14.33]  0/0/10  p 0.002 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 192 | 89.00 | -19.11 [-29.44, -10.11]  1/0/9  p 0.004 (exact sign-flip (mid-ranks), zeros 0) |
| bite | 256 | 94.22 | +1.94 [+0.72, +3.44]  7/3/0  p 0.016 (exact sign-flip (mid-ranks), zeros 3) |
| compact | 64 | 67.17 | -32.56 [-38.33, -26.61]  0/0/10  p 0.002 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 128 | 87.56 | -25.56 [-33.56, -18.17]  0/0/10  p 0.002 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 192 | 91.72 | -21.83 [-31.89, -13.00]  0/0/10  p 0.002 (exact sign-flip (mid-ranks), zeros 0) |
| compact | 256 | 94.11 | +2.06 [+0.33, +4.39]  5/5/0  p 0.062 (exact sign-flip (mid-ranks), zeros 5) |

## PREFIX family: READER vs Compact, same checkpoint, primary durations (Holm applied)

| corpus | samples | READER - Compact |
|---|---:|---|
| 2a | 250 | +39.17 [+34.61, +44.44]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| 2a | 500 | +36.95 [+26.52, +46.45]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| 2a | 750 | +31.64 [+23.47, +38.79]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| 2a | 1000 | +2.38 [+0.24, +4.58]  6/0/3  p 0.098 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.125 |
| 2b | 250 | +16.53 [+9.56, +23.50]  8/0/1  p 0.008 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| 2b | 500 | +11.45 [+5.14, +17.44]  8/0/1  p 0.012 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.047 |
| 2b | 750 | +11.59 [+6.19, +18.75]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| 2b | 1000 | +1.09 [+0.39, +1.83]  7/0/2  p 0.031 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.094 |
| sdssvep | 64 | +24.67 [+14.50, +34.72]  9/0/1  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| sdssvep | 128 | +48.22 [+34.00, +61.44]  9/0/1  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.043 |
| sdssvep | 192 | +55.83 [+41.89, +68.39]  10/0/0  p 0.002 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.023 |
| sdssvep | 256 | +2.06 [+0.33, +4.33]  5/5/0  p 0.062 (exact sign-flip (mid-ranks), zeros 5), Holm p 0.125 |
