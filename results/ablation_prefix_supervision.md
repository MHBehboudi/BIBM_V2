# Prefix supervision: deep supervision of the intermediate decisions

`--prefix-weight 0.0` (endpoint-only) vs `--prefix-weight 0.3`. One predeclared weight, no sweep. Same model, seeds, subjects, epochs and protocol; the loss is the only difference, checked by `_matched()`. Paired per (subject, seed); se is of the paired delta.

Accuracy at each deadline is the SAME trained model read at that deadline -- no retraining, no separate models. The endpoint row is the full-trial accuracy reported in the within-subject table.

## 2a  (n=27 cells, 128.000 ms grid, 32 tokens)

| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |
|---|---:|---:|---:|---:|---:|
| 1 s | 70.28 | 77.58 | +7.30 | 0.91 | 26/1 |
| 2 s | 82.16 | 83.68 | +1.52 | 0.42 | 18/6 |
| 3 s | 82.12 | 82.82 | +0.69 | 0.91 | 16/10 |
| 4 s (endpoint) | 84.71 | 83.50 | -1.21 | 0.78 | 8/15 |

## 2b  (n=27 cells, 128.000 ms grid, 32 tokens)

| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |
|---|---:|---:|---:|---:|---:|
| 1 s | 70.79 | 75.90 | +5.11 | 1.82 | 18/9 |
| 2 s | 82.67 | 83.56 | +0.89 | 0.61 | 14/12 |
| 3 s | 85.69 | 85.33 | -0.36 | 0.36 | 8/15 |
| 4 s (endpoint) | 86.41 | 85.59 | -0.82 | 0.43 | 6/19 |

## sdssvep  (n=30 cells, 15.625 ms grid, 64 tokens)

| deadline | endpoint-only | prefix-supervised | paired delta | se | W/L |
|---|---:|---:|---:|---:|---:|
| 0.25 s | 34.61 | 65.33 | +30.72 | 2.00 | 29/1 |
| 0.5 s | 62.00 | 84.28 | +22.28 | 2.43 | 30/0 |
| 0.75 s | 69.89 | 88.00 | +18.11 | 2.50 | 27/0 |
| 1 s (endpoint) | 96.17 | 94.33 | -1.83 | 0.48 | 1/14 |

