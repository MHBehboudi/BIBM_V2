# Anytime: one reader vs per-deadline specialist banks (sdssvep)

Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its within-subject cohort run.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.

| deadline | reader (1 model) | bite bank | compact bank | reader minus STRONGEST bank, subject-level | vs bite (runs) | vs compact (runs) |
|---|---|---|---|---|---|---|
| 0.25 s | 65.33 (30) | 66.11 (30) | 67.17 (30) | compact: -1.83 [-5.17, +2.06]  2/0/8  p 0.275 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.275 | -0.78 | -1.83 |
| 0.5 s | 84.28 (30) | 84.11 (30) | 87.56 (30) | compact: -3.28 [-4.89, -1.83]  0/0/10  p 0.002 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.006 | +0.17 | -3.28 |
| 0.75 s | 88.00 (30) | 89.00 (30) | 91.72 (30) | compact: -3.72 [-6.44, -1.06]  2/2/6  p 0.047 (exact sign-flip (mid-ranks), zeros 2), Holm p 0.094 | -1.00 | -3.72 |
| 1 s (full trial) | 94.33 (30) | 94.22 (30) | 94.11 (30) | bite: +0.11 [-1.33, +1.56]  5/3/2  p 0.672 (exact sign-flip (mid-ranks), zeros 3) | +0.11 | +0.22 |
