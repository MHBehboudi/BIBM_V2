# Anytime: one reader vs per-deadline specialist banks (2a)

Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its within-subject cohort run.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.

| deadline | reader (1 model) | bite bank | compact bank | reader minus STRONGEST bank, subject-level | vs bite (runs) | vs compact (runs) |
|---|---|---|---|---|---|---|
| 1 s | 76.84 (27) | 74.27 (27) | 74.83 (27) | compact: +2.01 [+0.76, +3.09]  7/0/2  p 0.027 (exact sign-flip (mid-ranks), zeros 0), Holm p 0.082 | +2.57 | +2.01 |
| 2 s | 83.37 (27) | 81.49 (27) | 81.57 (27) | compact: +1.80 [+0.48, +2.89]  7/1/1  p 0.039 (exact sign-flip (mid-ranks), zeros 1), Holm p 0.082 | +1.88 | +1.80 |
| 3 s | 84.10 (27) | 82.93 (27) | 83.35 (27) | compact: +0.76 [-0.72, +2.42]  5/1/3  p 0.531 (exact sign-flip (mid-ranks), zeros 1), Holm p 0.531 | +1.17 | +0.76 |
| 4 s (full trial) | 84.84 (27) | 84.10 (27) | 82.33 (27) | bite: +0.73 [+0.03, +1.59]  6/1/2  p 0.109 (exact sign-flip (mid-ranks), zeros 1) | +0.73 | +2.51 |
