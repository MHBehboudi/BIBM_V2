# Reproducibility

## GPU of every run of record

H100 80GB and H200 NVL reproduce each other bit-for-bit (45/45 identical READER re-runs). MIG-partitioned H100 NVL slices do not, so every MIG run of a comparison of record is re-run on a full GPU and the re-run is the record.

| runs | arm | GPU | runs | record status |
|---|---|---|---:|---|
| anytime | reader_anytime | H100 80GB HBM3 | 16 | original |
| anytime | reader_anytime | H100 NVL | 19 | original |
| anytime | reader_anytime | H100 NVL MIG 3g.47gb | 22 | mig_pending |
| anytime | reader_anytime | H200 NVL | 27 | original |
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
| bank | compact_1s | H100 80GB HBM3 | 13 | original |
| bank | compact_1s | H200 NVL | 14 | original |
| bank | compact_2s | H100 80GB HBM3 | 14 | original |
| bank | compact_2s | H200 NVL | 13 | original |
| bank | compact_3s | H100 80GB HBM3 | 14 | original |
| bank | compact_3s | H200 NVL | 13 | original |
| cohort | bite | H100 80GB HBM3 | 6 | original |
| cohort | bite | H100 NVL | 24 | original |
| cohort | bite | H100 NVL MIG 3g.47gb | 48 | mig_pending |
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

Pending: 147 MIG runs are queued for full-GPU re-runs (jobs 409509/409510).

## Test-curve peak minus reported final epoch (diagnostic; never used for selection)

BiTE's protocol trains 600 epochs and reports the final epoch with no validation set. Every arm loses accuracy between its own test-curve peak and that final epoch. The loss is a property of the recipe, not of READER, if it is similar across arms.

| runs | arm | runs | mean peak − final (pp) | se |
|---|---|---:|---:|---:|
| cohort | reader | 126 | 1.89 | 0.15 |
| cohort | compact | 126 | 2.04 | 0.13 |
| cohort | bite | 126 | 2.06 | 0.14 |
| zoo | EEGTCNet | 91 | 2.20 | 0.15 |
| controls | ff_control | 84 | 2.32 | 0.20 |
| zoo | DeepConvNet | 93 | 2.71 | 0.43 |
| zoo | ATCNet | 92 | 3.05 | 0.23 |
| zoo | EEGNet | 93 | 3.19 | 0.26 |
| zoo | EISATC | 92 | 3.31 | 0.17 |
| zoo | MBCNNEATCFNet | 92 | 3.43 | 0.32 |
| zoo | FACTNet | 62 | 3.53 | 0.21 |
| zoo | DMSANet | 92 | 4.53 | 0.43 |
| zoo | ShallowConvNet | 94 | 4.53 | 0.37 |
| zoo | EEGNeX | 91 | 4.54 | 0.26 |
| controls | compact_mean | 84 | 6.10 | 0.63 |
