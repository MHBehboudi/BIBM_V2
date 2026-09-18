# Cross-subject (leave-one-subject-out)

BiTE's cross-subject protocol: every other subject's sessions pooled as the training role, the held subject never seen; Euclidean alignment as BiTE enables it; 600 epochs, fixed final epoch. BiTE publishes no HGD row, so none is run. Mean ± SD across held subjects (seeds averaged per subject).

| model | BCI IV-2a | seeds (runs) | BCI IV-2b | seeds (runs) | SD-SSVEP | seeds (runs) |
|---|---:|---:|---:|---:|---:|---:|
| BiTE (--clip 0, as released) | 61.32 ± 12.83 | 3 (27) | 76.17 ± 6.29 | 3 (27) | 78.94 ± 20.79 | 3 (30) |
| Compact | 60.11 ± 14.44 | 3 (27) | 76.48 ± 6.48 | 3 (27) | 79.65 ± 21.69 | 3 (30) |
| **READER** | 61.10 ± 14.36 | 3 (27) | 75.86 ± 6.86 | 3 (27) | 80.57 ± 21.41 | 3 (30) |
| BiTE published (one seed) | 64.56 | 1 | 76.44 | 1 | 79.72 | 1 |
| *superseded: BiTE seed 2025 at clip 5* | *61.11* | 1 (9) | *76.08* | 1 (9) | *79.44* | 1 (10) |

READER minus comparator, subject-level (seeds present for BOTH arms averaged per subject):

| corpus | vs BiTE | vs Compact |
|---|---|---|
| BCI IV-2a | -0.23 [-2.58, +2.01]  5/0/4  p 0.910 (exact sign-flip (mid-ranks), zeros 0) | +0.98 [+0.20, +1.70]  7/0/2  p 0.074 (exact sign-flip (mid-ranks), zeros 0) |
| BCI IV-2b | -0.32 [-1.53, +1.02]  4/0/5  p 0.570 (exact sign-flip (mid-ranks), zeros 0) | -0.62 [-1.51, +0.16]  4/0/5  p 0.250 (exact sign-flip (mid-ranks), zeros 0) |
| SD-SSVEP | +1.63 [-0.15, +3.50]  6/0/4  p 0.232 (exact sign-flip (mid-ranks), zeros 0) | +0.93 [-0.46, +2.46]  8/0/2  p 0.152 (exact sign-flip (mid-ranks), zeros 0) |

The seed-2025 BiTE runs first reported here were trained at the trainer's default gradient clip of 5 by mistake; BiTE's release does not clip. They are shown in italics and are not the record.
