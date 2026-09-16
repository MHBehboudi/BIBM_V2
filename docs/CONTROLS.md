# Controls for the reverse-reading claim (declared before they were trained)

## Questions
1. Does READER's reversed, start-anchored second branch help beyond adding **another temporal branch**?
2. Does it help beyond a **better summary** of the observed sequence (a readout change)?
3. Does it improve the **same endpoint-trained checkpoint's** intermediate predictions over its forward-only parent?

## Arms
- `ff_control` — F1 + F2. F2 is built by the same `CausalTCN` call, forked RNG seed 918273, as READER's
  reverse branch, so its initial weights are identical; same convex per-feature gate (gamma = 0), same endpoint
  objective and recipe. It reads the prefix in ORIGINAL order; its output at deadline t is its last output in
  processing order. Parameter count equals READER's exactly. It does not separate *direction* from
  *start-anchoring*: reversal changes both.
- `compact_mean` — the forward-only parent with its classifier on the causal running mean of the forward TCN
  outputs, (1/t) sum_{i<=t} f_i, TRAINED with that readout (not swapped in after training). Parameters and
  initial weights identical to `compact`.
- References: the within-subject cohort checkpoints of `reader` and `compact` (not re-trained).

## Cells, seeds, hardware
2a S1-9, 2b S1-9, SD-SSVEP S1-10, seeds 2025 / 2026 / 2027: 84 runs per arm. Trained on H200 NVL, which
reproduces the cohort's H100 80GB runs bit-for-bit.

## Analysis (reader/subject_stats.py)
The unit of inference is the SUBJECT: d_i = (1/3) sum_s (A_{i,s}^READER - A_{i,s}^M). Per corpus and comparator:
mean d, 95% percentile bootstrap CI over subjects (10,000 resamples of d_i, i.e. paired), two-sided Wilcoxon
signed-rank on d_i with zero differences discarded and counted and the EXACT sign-flip null (mid-ranks, valid
under ties), subject W/T/L. Holm families:
- ENDPOINT (12): READER vs {Compact, FF-Control, Compact-Mean, BiTE} x {2a, 2b, SD-SSVEP}
- PREFIX (12): READER vs Compact, same checkpoint, at 1/4, 1/2, 3/4 and the full trial x 3 corpora
Exact-duration curves bootstrap whole subject curves together; no duration is singled out after the fact.

## Interpretation fixed in advance
- READER > FF-Control with a CI excluding 0: reversal/anchoring helps beyond a second forward branch.
- FF-Control ~ READER: the gain is a second branch's capacity or ensembling, not direction.
- Compact-Mean ~ READER: a running-mean summary explains the gain.
A null is reported as a null, with its interval.
