# Ablations

Three kinds of ablation, kept apart because they answer different questions:
  A  ARCHITECTURE, retrained from scratch with the same endpoint loss:
       Compact        the prefix-reversed branch removed (READER's parent)
       FF-Control     the second branch kept, capacity- and init-matched, but reading the prefix FORWARD
       Compact-Mean   no second branch; the classifier reads the causal running mean (a readout change)
  B  INFERENCE-TIME interventions on the SAME trained READER weights (no retraining):
       g = 0 forward route only, g = 1 reversed route only, g = 0.5 the unlearned initial average
  C  TRAINING OBJECTIVE: prefix supervision (cross-entropy on every intermediate decision, weight 0.3,
     the only weight ever trained), on READER and on Compact. READER+PS minus Compact+PS isolates the
     architecture under the identical loss.

Endpoint = full trial. Early = the same checkpoint given only the first quarter of the trial (2a/2b 1 s,
SD-SSVEP 0.25 s) and half of it (2 s / 0.5 s), exact truncated input. Differences are variant minus READER,
subject-level (seeds averaged per subject), 95% paired bootstrap CI, subject W/T/L.
Inference interventions at the endpoint are reported per run (mean, se, runs made worse) as recorded in training.

## Accuracy (%, subject-level mean over 3 seeds)

| variant | 2a full | 2b full | HGD full | SD-SSVEP full | 2a 1 s | 2a 2 s | 2b 1 s | 2b 2 s | SD-SSVEP 0.25 s | SD-SSVEP 0.5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| READER (endpoint loss) | 84.71 | 86.41 | 96.30 | 96.17 | 70.10 | 82.05 | 70.86 | 82.99 | 34.61 | 62.00 |
| A1  − reversed branch (Compact) | 82.33 | 85.32 | 95.69 | 94.11 | 30.93 | 45.10 | 54.33 | 71.55 | 9.94 | 13.78 |
| A2  reversed → forward second branch (FF-Control) | 83.08 | 85.15 | — | 94.17 | 30.25 | 50.82 | 54.57 | 70.11 | 10.89 | 16.67 |
| A3  second branch → running-mean readout (Compact-Mean) | 72.36 | 85.05 | — | 66.44 | 40.44 | 65.24 | 62.67 | 81.74 | 28.28 | 48.56 |
| C1  READER + prefix supervision | 83.50 | 85.59 | — | 94.33 | 77.21 | 83.64 | 75.74 | 83.52 | 65.33 | 84.28 |
| C2  Compact + prefix supervision | 82.10 | 86.49 | — | 94.11 | 57.63 | 73.24 | 71.88 | 82.11 | 43.11 | 71.17 |
| B1  g = 0: forward route only (inference) | 84.10 | 85.71 | 95.86 | 94.33 | 32.93 | 55.11 | 55.87 | 71.78 | 10.56 | 17.89 |
| B2  g = 1: reversed route only (inference) | 82.59 | 85.64 | 96.00 | 87.33 | 71.48 | 82.11 | 71.99 | 82.57 | 39.50 | 70.06 |
| B3  g = 0.5: unlearned equal average (inference) | 84.65 | 86.43 | 96.30 | 96.28 | 70.29 | 82.10 | 70.82 | 83.05 | 35.11 | 64.89 |

## Variant minus READER (pp): mean [95% CI] subject W/T/L

| variant | 2a full | 2b full | HGD full | SD-SSVEP full | 2a 1 s | 2a 2 s | 2b 1 s | 2b 2 s | SD-SSVEP 0.25 s | SD-SSVEP 0.5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1  − reversed branch (Compact) | -2.38 [-4.63, -0.23] 3/0/6 | -1.09 [-1.81, -0.39] 2/0/7 | -0.60 [-0.91, -0.31] 1/0/13 | -2.06 [-4.33, -0.33] 0/5/5 | -39.17 [-44.38, -34.68] 0/0/9 | -36.95 [-46.51, -26.04] 0/0/9 | -16.53 [-23.51, -9.39] 1/0/8 | -11.45 [-17.60, -5.24] 1/0/8 | -24.67 [-34.50, -14.50] 1/0/9 | -48.22 [-61.28, -33.78] 1/0/9 |
| A2  reversed → forward second branch (FF-Control) | -1.63 [-3.06, -0.31] 3/0/6 | -1.26 [-2.14, -0.33] 1/0/8 | — | -2.00 [-4.11, -0.50] 0/4/6 | -39.85 [-45.31, -34.74] 0/0/9 | -31.22 [-39.75, -21.44] 0/0/9 | -16.29 [-22.18, -10.34] 0/0/9 | -12.88 [-18.46, -6.28] 1/0/8 | -23.72 [-33.83, -13.11] 1/0/9 | -45.33 [-57.50, -32.00] 0/0/10 |
| A3  second branch → running-mean readout (Compact-Mean) | -12.35 [-18.62, -6.97] 0/0/9 | -1.36 [-3.63, +0.72] 3/0/6 | — | -29.72 [-40.67, -18.78] 0/0/10 | -29.66 [-32.70, -26.47] 0/0/9 | -16.81 [-22.45, -12.09] 0/0/9 | -8.20 [-12.91, -3.52] 2/0/7 | -1.26 [-4.16, +2.58] 2/0/7 | -6.33 [-12.28, -1.00] 2/0/8 | -13.44 [-22.44, -4.67] 3/0/7 |
| C1  READER + prefix supervision | -1.21 [-2.89, -0.09] 2/1/6 | -0.82 [-1.62, -0.02] 2/1/6 | — | -1.83 [-3.17, -0.67] 0/4/6 | +7.11 [+5.65, +8.83] 9/0/0 | +1.59 [+0.86, +2.43] 9/0/0 | +4.88 [+0.62, +10.87] 5/0/4 | +0.52 [-0.72, +1.96] 5/0/4 | +30.72 [+25.11, +35.67] 10/0/0 | +22.28 [+15.61, +29.72] 10/0/0 |
| C2  Compact + prefix supervision | -2.61 [-5.43, -0.19] 3/0/6 | +0.08 [-1.10, +1.41] 4/1/4 | — | -2.06 [-4.22, -0.50] 0/4/6 | -12.47 [-17.19, -7.88] 0/0/9 | -8.81 [-13.26, -5.26] 0/0/9 | +1.02 [-5.80, +8.95] 4/0/5 | -0.88 [-4.93, +3.52] 4/0/5 | +8.50 [+0.67, +15.33] 7/0/3 | +9.17 [-1.94, +20.00] 7/0/3 |
| B1  g = 0: forward route only (inference) | -0.60 (se 0.35, 19/27 runs worse) | -0.70 (se 0.23, 16/27 runs worse) | -0.44 (se 0.14, 22/42 runs worse) | -1.83 (se 0.57, 12/30 runs worse) | -37.17 [-42.57, -32.02] 0/0/9 | -26.94 [-33.50, -19.84] 0/0/9 | -14.99 [-19.11, -10.49] 0/0/9 | -11.22 [-17.26, -5.73] 1/0/8 | -24.06 [-33.22, -14.50] 1/0/9 | -44.11 [-55.22, -31.50] 0/0/10 |
| B2  g = 1: reversed route only (inference) | -2.12 (se 0.49, 21/27 runs worse) | -0.77 (se 0.29, 15/27 runs worse) | -0.30 (se 0.14, 17/42 runs worse) | -8.83 (se 1.61, 26/30 runs worse) | +1.38 [+0.89, +1.85] 9/0/0 | +0.06 [-0.53, +0.67] 5/1/3 | +1.12 [-0.43, +3.01] 5/0/4 | -0.42 [-0.73, -0.13] 2/0/7 | +4.89 [+2.33, +7.56] 8/0/2 | +8.06 [+3.78, +12.39] 9/0/1 |
| B3  g = 0.5: unlearned equal average (inference) | -0.06 (se 0.05, 8/27 runs worse) | +0.02 (se 0.07, 4/27 runs worse) | +0.00 (se 0.03, 2/42 runs worse) | +0.11 (se 0.11, 1/30 runs worse) | +0.19 [+0.01, +0.39] 7/0/2 | +0.05 [-0.03, +0.13] 4/2/3 | -0.05 [-0.09, -0.00] 1/3/5 | +0.06 [-0.01, +0.14] 4/4/1 | +0.50 [-0.28, +1.28] 6/1/3 | +2.89 [+1.56, +4.22] 8/1/1 |

## Architecture under the identical loss: (READER + PS) minus (Compact + PS)

Both arms trained with prefix supervision at weight 0.3. A positive interval means the prefix-reversed branch helps beyond what the loss alone buys.

| variant | 2a full | 2b full | HGD full | SD-SSVEP full | 2a 1 s | 2a 2 s | 2b 1 s | 2b 2 s | SD-SSVEP 0.25 s | SD-SSVEP 0.5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| READER+PS − Compact+PS | +1.40 [-0.23, +3.19] 5/0/4 | -0.90 [-1.60, -0.25] 1/2/6 | — | +0.22 [-1.67, +2.44] 3/3/4 | +19.59 [+15.84, +24.11] 9/0/0 | +10.40 [+6.79, +14.90] 9/0/0 | +3.85 [+0.04, +7.08] 7/0/2 | +1.40 [-2.13, +5.02] 5/0/4 | +22.22 [+16.89, +28.11] 10/0/0 | +13.11 [+6.78, +19.94] 9/0/1 |

HGD has no exact-duration evaluation and no FF-Control / Compact-Mean / prefix-supervised runs (its 14 subjects make every READER run ~1 GPU-hour); its column is the endpoint only.
