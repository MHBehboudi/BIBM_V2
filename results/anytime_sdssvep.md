# Anytime: one reader vs per-deadline specialist banks (sdssvep)

Each baseline column at deadline d is a **separately trained model**, trained from scratch on trials truncated to d; the reader column is **one model** read at d. The full-trial row of every bank is its within-subject cohort run.

Reader arm: `reader_anytime`. Prefix supervision moves the early deadlines by several points, so which arm this is forms part of the result, not metadata.

The comparison of record is against the STRONGEST bank at each deadline (the most accurate retrained specialist family), not the weakest. Mean columns are over (subject, seed) runs; the difference against the strongest bank is subject-level (seeds averaged per subject) with a 95% paired bootstrap CI, subject W/T/L and an exact sign-flip Wilcoxon p, Holm over the early deadlines.

| deadline | reader (1 model) |  | reader minus STRONGEST bank, subject-level |  |
|---|---|---|
| 0.25 s | 65.33 (30) |  | no bank |  |
| 0.5 s | 84.28 (30) |  | no bank |  |
| 0.75 s | 88.00 (30) |  | no bank |  |
| 1 s (full trial) | 94.33 (30) |  | no bank |  |

No specialist bank exists for this corpus yet.
