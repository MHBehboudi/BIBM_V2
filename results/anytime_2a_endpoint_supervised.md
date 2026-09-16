# Anytime: one reader vs per-deadline specialist banks (2a)

Each baseline column at deadline d is a **separately trained model**; the reader column is
**one model** read at d. Paired by subject and seed.

Reader arm: `reader`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate
retrained specialist), not the weakest; every family's paired delta is also listed.

| deadline | reader (1 model) | bite bank | compact bank | paired delta vs STRONGEST bank | vs bite | vs compact |
|---|---|---|---|---|---|---|
| 1.0 s | 70.28 (27) | 74.27 (27) | 74.83 (27) | -4.55 vs compact | -3.99 | -4.55 |
| 2.0 s | 82.16 (27) | 81.49 (27) | 81.57 (27) | +0.59 vs compact | +0.67 | +0.59 |
| 3.0 s | 82.12 (27) | 82.93 (27) | 83.35 (27) | -1.22 vs compact | -0.81 | -1.22 |
| 4.0 s | 84.71 (27) | 84.08 (27) | 82.33 (27) | +0.63 vs bite | +0.63 | +2.38 |
