# Anytime: one reader vs per-deadline specialist banks (2b)

Each baseline column at deadline d is a **separately trained model**; the reader column is
**one model** read at d. Paired by subject and seed.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate
retrained specialist), not the weakest; every family's paired delta is also listed.

| deadline | reader (1 model) | bite bank | paired delta vs STRONGEST bank | vs bite |
|---|---|---|---|---|
| 1.0 s | 75.92 (27) | 77.42 (27) | -1.50 vs bite | -1.50 |
| 2.0 s | 83.62 (27) | 85.24 (27) | -1.62 vs bite | -1.62 |
| 3.0 s | 85.42 (27) | 86.86 (27) | -1.44 vs bite | -1.44 |
| 4.0 s | 85.74 (27) | 87.33 (27) | -1.59 vs bite | -1.59 |
