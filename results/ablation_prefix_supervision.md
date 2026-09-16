# Prefix supervision: deep supervision of the intermediate decisions

`--prefix-weight 0.0` (endpoint-only) vs `--prefix-weight 0.3`. One predeclared weight, no sweep. Same model, seeds, subjects, epochs and protocol; the loss is the only difference, checked by `_matched()`. Paired per (subject, seed); se is of the paired delta.

Accuracy at each deadline is the SAME trained model read at that deadline -- no retraining, no separate models. The endpoint row is the full-trial accuracy reported in the within-subject table.

## 2a  (n=27 cells, 128.000 ms grid, 32 tokens)

| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |
|---|---:|---:|---:|---:|---:|
| 1 s | 70.28 | 76.84 | +6.56 | 0.95 | 24/3 |
| 2 s | 82.16 | 83.37 | +1.21 | 0.50 | 15/9 |
| 3 s | 82.12 | 84.10 | +1.98 | 1.07 | 16/9 |
| 4 s (endpoint) | 84.71 | 84.84 | +0.13 | 0.54 | 9/15 |

## 2b  (n=27 cells, 128.000 ms grid, 32 tokens)

| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |
|---|---:|---:|---:|---:|---:|
| 1 s | 70.79 | 75.92 | +5.14 | 1.81 | 18/8 |
| 2 s | 82.67 | 83.62 | +0.95 | 0.59 | 13/11 |
| 3 s | 85.69 | 85.42 | -0.27 | 0.35 | 8/15 |
| 4 s (endpoint) | 86.41 | 85.74 | -0.67 | 0.42 | 7/18 |

## sdssvep  (n=30 cells, 15.625 ms grid, 64 tokens)

| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |
|---|---:|---:|---:|---:|---:|
| 0.25 s | 34.61 | 65.33 | +30.72 | 2.00 | 29/1 |
| 0.5 s | 62.00 | 84.28 | +22.28 | 2.43 | 30/0 |
| 0.75 s | 69.89 | 88.00 | +18.11 | 2.50 | 27/0 |
| 1 s (endpoint) | 96.17 | 94.33 | -1.83 | 0.48 | 1/14 |

