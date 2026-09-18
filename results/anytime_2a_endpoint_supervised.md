# Anytime: one reader vs per-deadline specialist banks (2a)

Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its within-subject cohort run.

Reader arm: `reader`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.

| deadline | reader (1 model) | bite bank | compact bank | reader minus STRONGEST bank, subject-level | vs bite (runs) | vs compact (runs) |
|---|---|---|---|---|---|---|
| 1 s | 70.28 (27) | 74.42 (27) | 74.83 (27) | compact: -4.55 [-6.96, -2.47]  0/0/9  p 0.004 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.012 | -4.14 | -4.55 |
| 2 s | 82.16 (27) | 81.37 (27) | 81.57 (27) | compact: +0.59 [-0.68, +1.84]  5/0/4  p 0.410 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.820 | +0.80 | +0.59 |
| 3 s | 82.12 (27) | 83.02 (27) | 83.35 (27) | compact: -1.22 [-5.12, +1.67]  5/0/4  p 1.000 (exact sign-flip (mid-ranks), zeros 0), Holm p 1.000 | -0.90 | -1.22 |
| 4 s (full trial) | 84.71 (27) | 84.10 (27) | 82.33 (27) | bite: +0.60 [-0.80, +1.94]  6/0/3  p 0.426 (exact sign-flip (mid-ranks), zeros 0) | +0.60 | +2.38 |
