# Anytime: one reader vs per-deadline specialist banks (2a)

Each baseline column at deadline d is a **separately trained model**; the reader column is
**one model** read at d. Paired by subject and seed.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate
retrained specialist), not the weakest; every family's paired delta is also listed.

| deadline | reader (1 model) | bite bank | compact bank | paired delta vs STRONGEST bank | vs bite | vs compact |
|---|---|---|---|---|---|---|
| 1.0 s | 76.84 (27) | 74.27 (27) | 74.83 (27) | +2.01 vs compact | +2.57 | +2.01 |
| 2.0 s | 83.37 (27) | 81.49 (27) | 81.57 (27) | +1.80 vs compact | +1.88 | +1.80 |
| 3.0 s | 84.10 (27) | 82.93 (27) | 83.35 (27) | +0.76 vs compact | +1.17 | +0.76 |
| 4.0 s | 84.84 (27) | 84.08 (27) | 82.33 (27) | +0.76 vs bite | +0.76 | +2.51 |
