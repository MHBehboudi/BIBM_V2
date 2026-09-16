# Anytime: one reader vs per-deadline specialist banks (2a)

Each baseline column at deadline d is a **separately trained model**; the reader column is
**one model** read at d. Paired by subject and seed.

Reader arm: `reader`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

| deadline | reader (1 model) | bite bank | compact bank | best paired delta |
|---|---|---|---|---|
| 1.0 s | 70.28 (27) | 74.27 (27) | 74.83 (27) | -3.99 vs bite |
| 2.0 s | 82.16 (27) | 81.49 (27) | 81.57 (27) | +0.67 vs bite |
| 3.0 s | 82.12 (27) | 82.93 (27) | 83.35 (27) | -0.81 vs bite |
| 4.0 s | 84.71 (27) | 84.08 (27) | 82.33 (27) | +2.38 vs compact |
