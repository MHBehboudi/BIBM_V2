# Reproducibility

## GPU of every run of record

H100 80GB and H200 NVL reproduce each other bit-for-bit (45/45 identical READER re-runs). MIG-partitioned H100 NVL slices do not, so every MIG run of a comparison of record is re-run on a full GPU and the re-run is the record.

| runs | arm | GPU | runs | record status |
|---|---|---|---:|---|
| anytime | compact_anytime | H200 NVL | 84 | original |
| anytime | reader_anytime | H100 80GB HBM3 | 16 | original |
| anytime | reader_anytime | H100 NVL | 19 | original |
| anytime | reader_anytime | H100 NVL MIG 3g.47gb | 22 | mig_pending |
| anytime | reader_anytime | H200 NVL | 27 | original |
| bank | bite_0.25s | H200 NVL | 30 | original |
| bank | bite_0.5s | H200 NVL | 30 | original |
| bank | bite_0.75s | H200 NVL | 30 | original |
| bank | bite_1s | H100 80GB HBM3 | 9 | original |
| bank | bite_1s | H100 NVL | 4 | original |
| bank | bite_1s | H100 NVL MIG 3g.47gb | 23 | mig_pending |
| bank | bite_1s | H200 NVL | 18 | original |
| bank | bite_2s | H100 80GB HBM3 | 7 | original |
| bank | bite_2s | H100 NVL | 5 | original |
| bank | bite_2s | H100 NVL MIG 3g.47gb | 23 | mig_pending |
| bank | bite_2s | H200 NVL | 19 | original |
| bank | bite_3s | H100 80GB HBM3 | 9 | original |
| bank | bite_3s | H100 NVL | 3 | original |
| bank | bite_3s | H100 NVL MIG 3g.47gb | 23 | mig_pending |
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
| cohort | bite | H100 NVL MIG 3g.47gb | 20 | mig_pending |
| cohort | bite | H200 NVL | 28 | full-GPU re-run replaces MIG original |
| cohort | bite | H200 NVL | 48 | original |
| cohort | compact | H100 80GB HBM3 | 102 | original |
| cohort | compact | H100 NVL | 24 | original |
| cohort | reader | H100 80GB HBM3 | 84 | original |
| cohort | reader | H200 NVL | 42 | original |
| controls | compact_mean | H200 NVL | 84 | original |
| controls | ff_control | H200 NVL | 84 | original |
| loso | compact | H100 80GB HBM3 | 11 | original |
| loso | compact | H100 NVL | 6 | original |
| loso | compact | H100 NVL MIG 3g.47gb | 8 | mig_pending |
| loso | compact | H200 NVL | 3 | original |
| loso | reader | H100 80GB HBM3 | 7 | original |
| loso | reader | H100 NVL | 4 | original |
| loso | reader | H200 NVL | 17 | original |

## MIG original minus full-GPU re-run (identical arguments)

| runs | arm | pairs | mean (MIG − full) | se | pairs that differ | mean abs difference |
|---|---|---:|---:|---:|---:|---:|
| cohort | bite | 28 | -0.02 | 0.07 | 13 | 0.22 |

119 MIG runs are still waiting for their re-runs.

## Test-curve peak minus reported final epoch (diagnostic; never used for selection)

BiTE's protocol trains 600 epochs and reports the final epoch with no validation set. Every arm loses accuracy between its own test-curve peak and that final epoch. The loss is a property of the recipe, not of READER, if it is similar across arms.

| runs | arm | runs | mean peak − final (pp) | se |
|---|---|---:|---:|---:|
| cohort | reader | 126 | 1.89 | 0.15 |
| cohort | compact | 126 | 2.04 | 0.13 |
| cohort | bite | 126 | 2.08 | 0.14 |
| zoo | EEGTCNet | 94 | 2.19 | 0.15 |
| controls | ff_control | 84 | 2.32 | 0.20 |
| zoo | DeepConvNet | 95 | 2.69 | 0.42 |
| zoo | ATCNet | 94 | 3.03 | 0.22 |
| zoo | EEGNet | 94 | 3.18 | 0.26 |
| zoo | EISATC | 94 | 3.29 | 0.17 |
| zoo | MBCNNEATCFNet | 94 | 3.40 | 0.31 |
| zoo | FACTNet | 64 | 3.47 | 0.21 |
| zoo | DMSANet | 94 | 4.46 | 0.43 |
| zoo | ShallowConvNet | 95 | 4.49 | 0.37 |
| zoo | EEGNeX | 94 | 4.55 | 0.25 |
| controls | compact_mean | 84 | 6.10 | 0.63 |
