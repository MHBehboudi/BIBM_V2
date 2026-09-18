# Anytime: one reader vs per-deadline specialist banks (2b)

Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its within-subject cohort run.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.

| deadline | reader (1 model) | bite bank | reader minus STRONGEST bank, subject-level | vs bite (runs) |
|---|---|---|---|---|
| 1 s | 75.90 (27) | 77.37 (27) | bite: -1.47 [-3.54, +0.37]  3/0/6  p 0.301 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.301 | -1.47 |
| 2 s | 83.56 (27) | 85.15 (27) | bite: -1.59 [-2.84, -0.52]  1/0/8  p 0.020 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.059 | -1.59 |
| 3 s | 85.33 (27) | 86.79 (27) | bite: -1.46 [-2.46, -0.50]  1/0/8  p 0.039 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.078 | -1.46 |
| 4 s (full trial) | 85.59 (27) | 87.30 (27) | bite: -1.71 [-2.79, -0.33]  1/0/8  p 0.059 (exact sign-flip (mid-ranks), zeros 0) | -1.71 |
