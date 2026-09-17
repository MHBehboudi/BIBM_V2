# Efficiency: parameters and CPU latency per decision

CPU: AMD EPYC 9334 32-Core Processor, 1 thread, float32, batch 1, median of 50 calls. Latency is milliseconds of compute per decision, not including acquisition. BiTE includes its STFT.

**Latency is provisional.** These timings were taken on a shared cluster node; re-timing the same configuration later differed by up to 3x (READER on 2a at 1,000 samples: 10.6 ms here, 3.6 ms on re-measurement). Parameter counts are exact; do not quote the millisecond values, or any READER-vs-BiTE latency ordering, until they are re-measured on an idle node with interleaved repeats.

## 2a (22 channels, 1000 samples, 32 tokens of 128 ms)

| model | parameters | full-trial decision (ms) |
|---|---:|---:|
| **READER** | 20,964 | 10.30 |
| Compact | 17,828 | 10.49 |
| BiTE | 16,294 | 13.71 |
| EEGNet | 3,444 | 1.30 |
| EEGTCNet | 4,096 | 1.03 |
| FACTNet | 5,406 | 7.46 |
| EISATC | 26,383 | 2.29 |
| MBCNNEATCFNet | 29,520 | 15.72 |
| DMSANet | 35,884 | 21.49 |
| ShallowConvNet | 46,084 | 7.14 |
| EEGNeX | 56,452 | 13.43 |
| DeepConvNet | 102,329 | 1.69 |
| ATCNet | 113,732 | 4.80 |

Anytime deployment with decisions at 1 s, 2 s, 3 s, 4 s:

| deadline | READER single decision (ms) | BiTE specialist at that deadline (ms) |
|---|---:|---:|
| 1 s | 2.29 | 4.01 |
| 2 s | 2.94 | 5.35 |
| 3 s | 8.88 | 13.08 |
| 4 s | 10.59 | 14.57 |

- Models to store: READER **1** (20,964 parameters) vs a BiTE bank of **4** specialists (65,176 parameters in total).
- Timings are single-thread medians on a shared compute node; small steps between deadlines (e.g. a jump at the same length for both models) come from the convolution backend, not the models.
- READER's full anytime curve (all 32 decisions recomputed from scratch): 32.1 ms; a streaming system computes only the newest decision.

## 2b (3 channels, 1000 samples, 32 tokens of 128 ms)

| model | parameters | full-trial decision (ms) |
|---|---:|---:|
| **READER** | 19,010 | 2.08 |
| Compact | 15,874 | 1.28 |
| BiTE | 14,738 | 3.36 |
| EEGNet | 2,146 | 0.43 |
| EEGTCNet | 3,766 | 0.74 |
| FACTNet | 4,620 | 6.13 |
| ShallowConvNet | 10,802 | 0.50 |
| DMSANet | 18,722 | 2.71 |
| EISATC | 24,683 | 1.89 |
| MBCNNEATCFNet | 25,098 | 5.46 |
| EEGNeX | 54,738 | 2.63 |
| DeepConvNet | 86,052 | 0.98 |
| ATCNet | 112,794 | 3.50 |

Anytime deployment with decisions at 1 s, 2 s, 3 s, 4 s:

| deadline | READER single decision (ms) | BiTE specialist at that deadline (ms) |
|---|---:|---:|
| 1 s | 1.70 | 2.55 |
| 2 s | 1.75 | 2.90 |
| 3 s | 1.97 | 2.99 |
| 4 s | 1.84 | 3.44 |

- Models to store: READER **1** (19,010 parameters) vs a BiTE bank of **4** specialists (58,952 parameters in total).
- Timings are single-thread medians on a shared compute node; small steps between deadlines (e.g. a jump at the same length for both models) come from the convolution backend, not the models.
- READER's full anytime curve (all 32 decisions recomputed from scratch): 23.3 ms; a streaming system computes only the newest decision.

## hgd (44 channels, 1001 samples, 32 tokens of 128 ms)

| model | parameters | full-trial decision (ms) |
|---|---:|---:|
| **READER** | 23,076 | 21.14 |
| Compact | 19,940 | 20.60 |
| BiTE | 17,922 | 27.14 |
| EEGNet | 3,796 | 2.00 |
| EEGTCNet | 4,448 | 1.53 |
| FACTNet | 5,758 | 8.81 |
| EISATC | 27,087 | 2.65 |
| MBCNNEATCFNet | 33,744 | 16.72 |
| DMSANet | 38,524 | 32.54 |
| EEGNeX | 57,860 | 25.03 |
| ShallowConvNet | 81,284 | 5.17 |
| ATCNet | 114,436 | 6.29 |
| DeepConvNet | 116,079 | 2.80 |

Anytime deployment with decisions at 1 s, 2 s, 3 s, 4 s:

| deadline | READER single decision (ms) | BiTE specialist at that deadline (ms) |
|---|---:|---:|
| 1 s | 2.47 | 4.74 |
| 2 s | 3.39 | 8.32 |
| 3 s | 4.63 | 21.68 |
| 4 s | 5.71 | 29.95 |

- Models to store: READER **1** (23,076 parameters) vs a BiTE bank of **4** specialists (71,688 parameters in total).
- Timings are single-thread medians on a shared compute node; small steps between deadlines (e.g. a jump at the same length for both models) come from the convolution backend, not the models.
- READER's full anytime curve (all 32 decisions recomputed from scratch): 40.8 ms; a streaming system computes only the newest decision.

## sdssvep (8 channels, 256 samples, 64 tokens of 15.625 ms)

| model | parameters | full-trial decision (ms) |
|---|---:|---:|
| **READER** | 32,428 | 1.88 |
| Compact | 29,292 | 1.22 |
| BiTE | 14,548 | 2.79 |
| FACTNet | cannot run as released | — |
| EEGNet | 2,780 | 0.36 |
| EEGTCNet | 3,976 | 0.66 |
| ShallowConvNet | 19,212 | 0.37 |
| DMSANet | 21,732 | 2.00 |
| EISATC | 26,057 | 1.74 |
| MBCNNEATCFNet | 29,928 | 4.49 |
| DeepConvNet | 46,262 | 0.35 |
| EEGNeX | 55,340 | 1.48 |
| ATCNet | 114,604 | 2.97 |

Anytime deployment with decisions at 0.25 s, 0.5 s, 0.75 s, 1 s:

| deadline | READER single decision (ms) | BiTE specialist at that deadline (ms) |
|---|---:|---:|
| 0.25 s | 1.55 | 2.42 |
| 0.5 s | 1.72 | 2.47 |
| 0.75 s | 1.63 | 2.56 |
| 1 s | 1.92 | 2.70 |

- Models to store: READER **1** (32,428 parameters) vs a BiTE bank of **4** specialists (58,192 parameters in total).
- Timings are single-thread medians on a shared compute node; small steps between deadlines (e.g. a jump at the same length for both models) come from the convolution backend, not the models.
- READER's full anytime curve (all 64 decisions recomputed from scratch): 43.6 ms; a streaming system computes only the newest decision.

