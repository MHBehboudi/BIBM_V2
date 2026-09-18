# Reproducibility

## GPU of every run of record

H100 80GB and H200 NVL reproduce each other bit-for-bit (45/45 identical READER re-runs). MIG-partitioned H100 NVL slices do not, so every MIG run of a comparison of record is re-run on a full GPU and the re-run is the record.

| runs | arm | GPU | runs | record status |
|---|---|---|---:|---|
| anytime | compact_anytime | H200 NVL | 84 | original |
| anytime | reader_anytime | H100 80GB HBM3 | 16 | original |
| anytime | reader_anytime | H100 NVL | 19 | original |
| anytime | reader_anytime | H200 NVL | 22 | full-GPU re-run replaces MIG original |
| anytime | reader_anytime | H200 NVL | 27 | original |
| bank | bite_0.25s | H200 NVL | 30 | original |
| bank | bite_0.5s | H200 NVL | 30 | original |
| bank | bite_0.75s | H200 NVL | 30 | original |
| bank | bite_1s | H100 80GB HBM3 | 9 | original |
| bank | bite_1s | H100 NVL | 4 | original |
| bank | bite_1s | H200 NVL | 23 | full-GPU re-run replaces MIG original |
| bank | bite_1s | H200 NVL | 18 | original |
| bank | bite_2s | H100 80GB HBM3 | 7 | original |
| bank | bite_2s | H100 NVL | 5 | original |
| bank | bite_2s | H200 NVL | 23 | full-GPU re-run replaces MIG original |
| bank | bite_2s | H200 NVL | 19 | original |
| bank | bite_3s | H100 80GB HBM3 | 9 | original |
| bank | bite_3s | H100 NVL | 3 | original |
| bank | bite_3s | H200 NVL | 23 | full-GPU re-run replaces MIG original |
| bank | bite_3s | H200 NVL | 19 | original |
| bank | compact_0.25s | H200 NVL | 30 | original |
| bank | compact_0.5s | H200 NVL | 30 | original |
| bank | compact_0.75s | H200 NVL | 30 | original |
| bank | compact_1s | H100 80GB HBM3 | 13 | original |
| bank | compact_1s | H200 NVL | 14 | original |
| bank | compact_2s | H100 80GB HBM3 | 14 | original |
| bank | compact_2s | H200 NVL | 13 | original |
| bank | compact_3s | H100 80GB HBM3 | 14 | original |
| bank | compact_3s | H200 NVL | 13 | original |
| cohort | bite | H100 80GB HBM3 | 6 | original |
| cohort | bite | H100 NVL | 24 | original |
| cohort | bite | H200 NVL | 48 | full-GPU re-run replaces MIG original |
| cohort | bite | H200 NVL | 48 | original |
| cohort | compact | H100 80GB HBM3 | 102 | original |
| cohort | compact | H100 NVL | 24 | original |
| cohort | reader | H100 80GB HBM3 | 84 | original |
| cohort | reader | H200 NVL | 42 | original |
| controls | compact_mean | H200 NVL | 84 | original |
| controls | ff_control | H200 NVL | 84 | original |
| loso | bite | H200 NVL | 84 | original |
| loso | compact | H100 80GB HBM3 | 11 | original |
| loso | compact | H100 NVL | 6 | original |
| loso | compact | H200 NVL | 8 | full-GPU re-run replaces MIG original |
| loso | compact | H200 NVL | 59 | original |
| loso | reader | H100 80GB HBM3 | 7 | original |
| loso | reader | H100 NVL | 4 | original |
| loso | reader | H200 NVL | 73 | original |

## MIG original minus full-GPU re-run (identical arguments)

| runs | arm | pairs | mean (MIG − full) | se | pairs that differ | mean abs difference |
|---|---|---:|---:|---:|---:|---:|
| anytime | reader_anytime | 22 | +1.83 | 1.35 | 21 | 3.08 |
| bank | bite_1s | 23 | -0.13 | 0.22 | 17 | 0.70 |
| bank | bite_2s | 23 | +0.26 | 0.16 | 17 | 0.52 |
| bank | bite_3s | 23 | -0.02 | 0.12 | 16 | 0.42 |
| cohort | bite | 48 | +0.04 | 0.06 | 19 | 0.21 |
| loso | compact | 8 | -0.00 | 0.41 | 8 | 0.74 |

## Test-curve peak minus reported final epoch (diagnostic; never used for selection)

BiTE's protocol trains 600 epochs and reports the final epoch with no validation set. Every arm loses accuracy between its own test-curve peak and that final epoch. The loss is a property of the recipe, not of READER, if it is similar across arms.

| runs | arm | runs | mean peak − final (pp) | se |
|---|---|---:|---:|---:|
| cohort | reader | 126 | 1.89 | 0.15 |
| zoo | EEGTCNet | 126 | 2.03 | 0.12 |
| cohort | compact | 126 | 2.04 | 0.13 |
| cohort | bite | 126 | 2.10 | 0.14 |
| controls | ff_control | 84 | 2.32 | 0.20 |
| zoo | ATCNet | 126 | 2.73 | 0.18 |
| zoo | DeepConvNet | 126 | 2.77 | 0.34 |
| zoo | EEGNet | 126 | 2.82 | 0.20 |
| zoo | MBCNNEATCFNet | 126 | 3.02 | 0.24 |
| zoo | FACTNet | 96 | 3.08 | 0.17 |
| zoo | EISATC | 126 | 3.16 | 0.14 |
| zoo | DMSANet | 126 | 3.73 | 0.34 |
| zoo | ShallowConvNet | 126 | 3.75 | 0.30 |
| zoo | EEGNeX | 126 | 4.19 | 0.21 |
| controls | compact_mean | 84 | 6.10 | 0.63 |
