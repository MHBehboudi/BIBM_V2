# Run card: paper-completion runs (declared 2026-09-17, before any of them finished)

No new architecture. These runs fill evidence gaps a reviewer will find in the READER paper, and fix two
fairness defects found while packaging. They were launched in the development harness with arguments copied
from the manifests that produced the originals (only `--out` and `--seed` change); in this repository the same
runs are produced by `scripts/reproduce_anytime.sh` (bank, compact_anytime), `scripts/reproduce_loso.sh` and
`scripts/reproduce_within.sh`. All jobs pinned to H200 NVL (bit-identical to H100 80GB; no MIG slices).
`scripts/link_record_runs.py` applies the hardware and clipping rules below when assembling the runs of record.

## A — two-paradigm anytime, matched-loss ablation, clean BiTE 2a pairing (294 runs)
| arm | cells | why |
|---|---|---|
| `bank_20260917/{bite,compact}_{0.25,0.5,0.75}s` | SD-SSVEP 10 x 3 seeds | the anytime-vs-bank claim exists only on motor imagery (2a/2b); SSVEP is the second paradigm |
| `anytime_20260917/compact_anytime` (compact, `--prefix-weight 0.3`) | 2a, 2b, SD x 3 seeds | the anytime win uses a loss change; this isolates the ARCHITECTURE under the identical loss |
| `fullgpu_20260917/cohort_20260915/bite/2a_*` | 2a 9 x 3 seeds | all 27 cohort BiTE 2a runs ran on a MIG slice, READER on full GPUs |
| zoo retries | SD S1 seed 2025: DeepConvNet, EEGNeX, EEGTCNet | startup bus-error crashes |

## B — remaining MIG runs of comparisons of record, re-run on H200 (120 runs)
reader_anytime 2a 19 / 2b 3; bank bite 2a 41 / 2b 28; cohort bite 2b 9 / HGD 12; LOSO compact 2a 4 / 2b 4.

## C — cross-subject uncertainty (196 runs)
compact and reader: seeds 2026, 2027 (28 subjects each). BiTE: seeds 2025, 2026, 2027 at `--clip 0`.
DEFECT FIXED: the seed-2025 BiTE LOSO runs used the trainer default clip 5, while BiTE's release does not clip
and every within-subject BiTE run here uses `--clip 0`. The clip-5 runs are kept only as a reported note.

## Declared analyses (fixed before results)
1. **Hardware rule.** For any cell whose original ran on MIG, the full-GPU re-run is the comparison of record.
   Report MIG minus full-GPU per arm (paired, n, mean, se, cells that differ) as a reproducibility result.
2. **SD-SSVEP anytime.** One `reader_anytime` read at 0.25 / 0.5 / 0.75 s vs the STRONGEST retrained bank
   family at each deadline; the endpoint (1 s) row is the cohort. Endpoint-supervised `reader` reported too.
   Subject-level mean d, 95% bootstrap CI, exact sign-flip Wilcoxon. SD-SSVEP prefix supervision costs -1.83
   at the endpoint; that is reported beside it, not hidden.
3. **Matched-loss ablation.** `reader_anytime` minus `compact_anytime` at each deadline (2a/2b: 1,2,3,4 s;
   SD: .25,.5,.75,1 s), same checkpoints on exact truncated input, subject-level, Holm over the 12 tests.
   Reading: d > 0 with CI above 0 = the architecture helps beyond the loss; d ~ 0 = the anytime gain is the loss.
4. **LOSO.** 3 seeds per arm, subject-level READER minus BiTE and minus compact, CI + exact test; published
   BiTE bars beside. No HGD row (BiTE publishes none).
Estimated 157 GPU-hours.
