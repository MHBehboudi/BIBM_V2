# Cross-subject (leave-one-subject-out)

BiTE's cross-subject protocol: every other subject's sessions pooled as the training role, the held subject never seen; Euclidean alignment as BiTE enables it; 600 epochs, fixed final epoch. BiTE publishes no HGD row, so none is run. Mean ± SD across held subjects (seeds averaged per subject).

| model | BCI IV-2a | seeds (runs) | BCI IV-2b | seeds (runs) | SD-SSVEP | seeds (runs) |
|---|---:|---:|---:|---:|---:|---:|
| BiTE (--clip 0, as released) | pending | 0 | pending | 0 | pending | 0 |
| Compact | 59.66 ± 15.10 | 1 (9) | 76.65 ± 6.73 | 1 (9) | 80.11 ± 21.69 | 1 (10) |
| **READER** | 60.36 ± 13.11 | 1 (9) | 75.83 ± 6.88 | 1 (9) | 81.22 ± 21.78 | 1 (10) |
| BiTE published (one seed) | 64.56 | 1 | 76.44 | 1 | 79.72 | 1 |
| *superseded: BiTE seed 2025 at clip 5* | *61.11* | 1 (9) | *76.08* | 1 (9) | *79.44* | 1 (10) |

READER minus comparator, subject-level (seeds present for BOTH arms averaged per subject):

| corpus | vs BiTE | vs Compact |
|---|---|---|
| BCI IV-2a | pending | +0.69 [-0.91, +2.91]  5/1/3  p 0.844 (exact sign-flip (mid-ranks), zeros 1) |
| BCI IV-2b | pending | -0.82 [-2.29, +0.42]  4/0/5  p 0.426 (exact sign-flip (mid-ranks), zeros 0) |
| SD-SSVEP | pending | +1.11 [+0.06, +2.11]  8/0/2  p 0.100 (exact sign-flip (mid-ranks), zeros 0) |

The seed-2025 BiTE runs first reported here were trained at the trainer's default gradient clip of 5 by mistake; BiTE's release does not clip. They are shown in italics and are not the record.
