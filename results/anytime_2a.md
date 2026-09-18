# Anytime: one reader vs per-deadline specialist banks (2a)

Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its within-subject cohort run.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.

| deadline | reader (1 model) | bite bank | compact bank | reader minus STRONGEST bank, subject-level | vs bite (runs) | vs compact (runs) |
|---|---|---|---|---|---|---|
| 1 s | 77.58 (27) | 74.42 (27) | 74.83 (27) | compact: +2.75 [+1.35, +3.95]  8/0/1  p 0.020 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.039 | +3.16 | +2.75 |
| 2 s | 83.68 (27) | 81.37 (27) | 81.57 (27) | compact: +2.11 [+1.02, +3.00]  8/0/1  p 0.012 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.035 | +2.31 | +2.11 |
| 3 s | 82.82 (27) | 83.02 (27) | 83.35 (27) | compact: -0.53 [-4.53, +2.35]  5/0/4  p 0.652 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.652 | -0.21 | -0.53 |
| 4 s (full trial) | 83.50 (27) | 84.10 (27) | 82.33 (27) | bite: -0.60 [-3.45, +1.34]  6/0/3  p 0.652 (exact sign-flip (mid-ranks), zeros 0) | -0.60 | +1.17 |
