# Error analysis, calibration and complementarity

Computed from the FINAL-EPOCH test logits stored with every run of record. Each cell's logits
were checked against the accuracy in its `summary.json`; a mismatch aborts the script, so these
are the same runs and the same epoch as `results/MAIN_TABLE.md`, not a second selection.
Produced by `reader/error_analysis.py`.

## Per-class recall (%), pooled over subjects and seeds

**BCI IV-2a**

| arm | left hand | right hand | feet | tongue | spread |
|---|---:|---:|---:|---:|---:|
| reader | 86.5 | 86.9 | 81.0 | 84.4 | 5.9 |
| compact | 83.7 | 84.7 | 79.6 | 81.3 | 5.1 |
| bite | 85.9 | 87.6 | 81.9 | 81.0 | 6.6 |

**BCI IV-2b**

| arm | left hand | right hand | spread |
|---|---:|---:|---:|
| reader | 87.9 | 85.3 | 2.6 |
| compact | 87.1 | 83.9 | 3.1 |
| bite | 90.4 | 84.6 | 5.8 |

**HGD**

| arm | left hand | right hand | feet | rest | spread |
|---|---:|---:|---:|---:|---:|
| reader | 96.3 | 95.1 | 96.3 | 97.2 | 2.0 |
| compact | 94.8 | 94.9 | 96.1 | 96.8 | 2.0 |
| bite | 94.7 | 94.2 | 96.5 | 96.0 | 2.4 |

**SD-SSVEP**

| arm | class 1 | class 2 | class 3 | class 4 | class 5 | class 6 | class 7 | class 8 | class 9 | class 10 | class 11 | class 12 | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reader | 96.0 | 98.7 | 94.7 | 96.7 | 96.7 | 98.0 | 98.0 | 93.3 | 93.3 | 98.7 | 99.3 | 90.7 | 8.7 |
| compact | 94.7 | 98.0 | 91.3 | 93.3 | 94.0 | 94.0 | 98.7 | 91.3 | 90.7 | 96.7 | 96.7 | 90.0 | 8.7 |
| bite | 96.0 | 98.7 | 94.0 | 96.7 | 92.0 | 96.0 | 98.7 | 93.3 | 90.7 | 98.7 | 94.7 | 81.3 | 17.3 |

## Calibration

ECE over 15 equal-width confidence bins. Confidence is a property of trials, so this table is
trial-pooled and its accuracy column is therefore the trial-pooled mean, which differs slightly
from the subject-mean in `results/MAIN_TABLE.md` where test sessions differ in size (2b, HGD).
`conf − acc` is positive when a model is overconfident. Bin tables: `results/error_analysis.json`.

**Every model here is UNDER-confident, on every corpus.** That is the expected direction for this
recipe rather than a finding about the architecture: all arms train with label smoothing 0.1, which
caps the target probability, and the effect grows with the number of classes (largest on SD-SSVEP's
twelve). It does mean a confidence threshold tuned on one corpus will not transfer to another.

| corpus | arm | accuracy (trial-pooled) | mean confidence | conf − acc | ECE |
|---|---|---:|---:|---:|---:|
| BCI IV-2a | reader | 84.71 | 69.30 | -15.41 | 0.1541 |
| BCI IV-2a | compact | 82.33 | 69.22 | -13.11 | 0.1311 |
| BCI IV-2a | bite | 84.10 | 71.29 | -12.82 | 0.1282 |
| BCI IV-2b | reader | 86.60 | 81.91 | -4.69 | 0.0470 |
| BCI IV-2b | compact | 85.52 | 83.39 | -2.13 | 0.0213 |
| BCI IV-2b | bite | 87.48 | 83.54 | -3.94 | 0.0394 |
| HGD | reader | 96.19 | 82.90 | -13.29 | 0.1329 |
| HGD | compact | 95.61 | 84.44 | -11.18 | 0.1123 |
| HGD | bite | 95.30 | 82.25 | -13.06 | 0.1306 |
| SD-SSVEP | reader | 96.17 | 63.01 | -33.16 | 0.3317 |
| SD-SSVEP | compact | 94.11 | 63.15 | -30.97 | 0.3098 |
| SD-SSVEP | bite | 94.22 | 60.80 | -33.43 | 0.3345 |

## READER and BiTE make different mistakes

Same trials, same order, trial-by-trial. `oracle` = either model is right (an upper bound no
selector can exceed); `ensemble` = the two softmax outputs averaged, which needs both models at
inference and is reported as a diagnostic, not as a proposed system.

| corpus | both right | only READER | only BiTE | both wrong | oracle | softmax-average |
|---|---:|---:|---:|---:|---:|---:|
| BCI IV-2a | 77.80 | 6.91 | 6.30 | 8.99 | 91.01 | 86.50 |
| BCI IV-2b | 82.13 | 4.28 | 5.16 | 8.43 | 91.57 | 87.96 |
| HGD | 93.90 | 2.40 | 1.63 | 2.07 | 97.93 | 96.66 |
| SD-SSVEP | 92.39 | 3.78 | 1.83 | 2.00 | 98.00 | 96.61 |

Subject-level, with the paper's bootstrap CI and exact Wilcoxon test:

| corpus | oracle − READER | softmax-average − READER |
|---|---|---|
| BCI IV-2a | +6.30 [+4.19, +8.55]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0) | +1.79 [+0.95, +2.80]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0) |
| BCI IV-2b | +5.16 [+4.18, +6.19]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0) | +1.55 [+1.06, +2.04]  9/0/0  p 0.004 (exact sign-flip (mid-ranks), zeros 0) |
| HGD | +1.63 [+1.24, +2.04]  14/0/0  p 0.000 (exact sign-flip (mid-ranks), zeros 0) | +0.36 [+0.11, +0.60]  10/3/1  p 0.032 (exact sign-flip (mid-ranks), zeros 3) |
| SD-SSVEP | +1.83 [+0.00, +4.33]  3/7/0  p 0.250 (exact sign-flip (mid-ranks), zeros 7) | +0.44 [-0.72, +1.78]  3/5/2  p 0.812 (exact sign-flip (mid-ranks), zeros 5) |

## Re-run noise: SD across the three seeds, within subject

Every delta in the paper is a difference of numbers with this much spread underneath it.

| corpus | arm | mean SD (pp) | max SD (pp) | mean seed range (pp) | max range (pp) |
|---|---|---:|---:|---:|---:|
| BCI IV-2a | reader | 1.91 | 5.54 | 3.51 | 10.07 |
| BCI IV-2a | compact | 1.27 | 3.34 | 2.47 | 6.60 |
| BCI IV-2a | bite | 1.34 | 3.18 | 2.62 | 6.25 |
| BCI IV-2b | reader | 1.24 | 4.03 | 2.35 | 7.81 |
| BCI IV-2b | compact | 1.06 | 2.60 | 2.06 | 5.00 |
| BCI IV-2b | bite | 0.77 | 1.43 | 1.49 | 2.81 |
| HGD | reader | 0.72 | 1.25 | 1.35 | 2.50 |
| HGD | compact | 1.03 | 2.05 | 1.91 | 3.98 |
| HGD | bite | 0.67 | 1.50 | 1.24 | 2.84 |
| SD-SSVEP | reader | 1.12 | 4.81 | 2.00 | 8.33 |
| SD-SSVEP | compact | 0.67 | 1.92 | 1.17 | 3.33 |
| SD-SSVEP | bite | 1.33 | 3.47 | 2.50 | 6.67 |
