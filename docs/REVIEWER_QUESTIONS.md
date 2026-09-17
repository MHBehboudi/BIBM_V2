# Questions a reviewer will ask, and where the answer is

Each entry states the concern, what was done about it, the evidence file, and what remains open. Numbers are
subject-level (seeds averaged per subject) with 95% paired bootstrap CIs unless marked otherwise. Entries marked
**PENDING** depend on runs launched 2026-09-17 (declared analyses fixed in advance:
`docs/RUN_CARD_PAPER_COMPLETION.md`); `scripts/rebuild_results.sh` regenerates every file once they land.

## 1. Does READER beat the state of the art?

**Done.** Every baseline BiTE ships (ATCNet, DMSANet, DeepConvNet, EEGNeX, EEGNet, EEGTCNet, EISATC, FACTNet,
MBCNNEATCFNet, ShallowConvNet) and BiTE itself were re-run with 3 seeds under the identical protocol, so the
comparison is paired rather than against single-seed published numbers. Our seed-2025 re-runs track BiTE's
published seed-2025 table closely (mean absolute gap about 1 pp), so the harness does not handicap the baselines.
Evidence: `results/MAIN_TABLE.md`, `results/model_zoo/MODEL_ZOO.md`, `results/figures/fig_reader_vs_zoo.pdf`.

**Answer, stated as it is.** READER minus BiTE (re-run): 2a +0.60 [−0.80, +1.94], 2b −0.92 [−1.88, +0.08],
HGD +0.73 [−0.02, +1.59], SD-SSVEP +1.94 [+0.72, +3.39]. At the endpoint READER is at **parity with BiTE on motor
imagery** (both intervals span zero, 2b leans to BiTE) and ahead of BiTE on SD-SSVEP (Holm p .047 over the eleven
SD-SSVEP comparisons). It beats every other re-run baseline on 2a and 2b, and is level with DeepConvNet on SD-SSVEP
(−0.33 [−1.56, +0.67]). Against the published single-seed bars it is −0.63 (2a), −1.96 (2b), +0.37 (HGD), +0.67
(SD-SSVEP). The paper's claim is therefore not "higher endpoint accuracy everywhere" but endpoint parity or
better **plus** a legal decision at every deadline from one model (sections 5–6).

**Open.** The HGD zoo wave is PENDING.

## 2. Is the gain just extra parameters or a second branch?

**Done.** FF-Control keeps READER's second branch with identical constructor, initial weights, gate and
objective, but reads the prefix forward. READER minus FF-Control: 2a +1.63 [+0.27, +3.07], 2b +1.26 [+0.34,
+2.19], SD +2.00 [+0.50, +4.17]; FF-Control minus Compact is inside noise on all three. A second branch of the
same size does not reproduce the gain. Evidence: `results/ABLATION.md` (A2), `results/subject_stats/SUBJECT_STATS.md`.

**Open.** None of these endpoint intervals survives Holm correction over the 12 declared endpoint tests (9–10
subjects per corpus bounds the smallest attainable p). State the CIs, not significance.

## 3. Is it reversal, or just a readout anchored at the trial start?

**Partly separated.** Compact-Mean (position-free running-mean readout) recovers much of Compact's early-deadline
collapse, so anchoring matters; READER still leads it at every duration on 2a and SD-SSVEP, and at the
endpoint Compact-Mean is far worse (2a −12.35, SD −29.72). Inference-time interventions agree: the reversed route
alone (g = 1) is *better* than the learned mixture early and worse at the endpoint.
Evidence: `results/ABLATION.md` (A3, B1–B3), `docs/CONTROLS.md`.

**Open, and stated as a limitation.** FF-Control is end-anchored like Compact, so no trained control separates
*direction* from *start-anchoring*.

## 4. The fusion gate does not learn anything

**Conceded and reported.** The gate stays within 0.01 of its 0.5 initialisation and pinning it there is free on
every corpus (−0.06 to +0.11). The fusion is load-bearing (removing either route costs accuracy on all four
corpora) but it is described as a **fixed equal-weight average**, never as a learned adaptive gate.
Evidence: `results/gate_intervention.md`, `results/ABLATION.md` (B).

## 5. The anytime win needs a loss change

**Done: loss ablation and matched-loss architecture ablation.** Prefix supervision (weight 0.3, the only
weight ever trained) buys early accuracy on every corpus (2a +6.56 at 1 s, SD +30.72 at 0.25 s) and is free at the
2a endpoint (+0.13) but costs −0.67 on 2b and −1.83 on SD-SSVEP, so the within-subject table is not built on it.
To separate the architecture from the loss, Compact was re-trained with the identical prefix loss. READER+PS minus
Compact+PS on the same truncated inputs: 2a +19.20 [+15.74, +23.30] at 1 s and +10.12 [+6.31, +14.99] at 2 s (9/0/0
subjects both), 2b +3.89 [+0.08, +7.10] at 1 s, SD-SSVEP +22.22 [+16.89, +28.11] at 0.25 s and +13.11 [+6.78, +19.94]
at 0.5 s; at the full trial +2.74 / −0.75 / +0.22. The early-decision advantage of a single model is the
architecture's, not the loss's.
Evidence: `results/ablation_prefix_supervision.md`, `results/ABLATION.md`.

## 6. The anytime claim covers one corpus and one paradigm

**Done, and conceded: the claim holds on 2a only.** 2a, one READER vs the strongest bank of per-deadline
specialists: +2.01 [+0.76, +3.09] at 1 s, +1.80 [+0.48, +2.89] at 2 s, +0.76 [−0.72, +2.42] at 3 s (Holm over the
three early deadlines: p = .082, .082, .53). 2b is a flat −1.5 at every deadline, the size of BiTE's standing 2b
endpoint lead. SD-SSVEP (second paradigm): READER is level with BiTE's bank but Compact specialists retrained at each
deadline beat it: −1.83 [−5.17, +2.06] at 0.25 s, −3.28 [−4.89, −1.83] at 0.5 s (0/0/10 subjects, Holm p .006),
−3.72 [−6.44, −1.06] at 0.75 s. The paper can claim that one READER matches or beats a bank of retrained
specialists on 2a, and that as a single model it decides early far better than the same backbone without the
reversed branch trained with the same loss (section 5); not that it replaces the bank at no cost in general. Evidence: `results/anytime_2a.md`, `anytime_2b.md`, `anytime_sdssvep.md`,
`results/figures/fig_anytime.pdf`.

**Open.** No bank on HGD (same paradigm as 2a/2b; cost).

## 7. Is it really causal?

**Done.** On 252 trained checkpoints the truncated-input logits equal the anytime curve exactly (0.0 over 10,782
pooling boundaries), and replacing all future samples by 1e3-scale noise changes earlier decisions by 0.0. The
unit tests run on activated weights, where a full-trial-reversal mutant is detected (it is invisible at the
zero-initialised identity). Caveat stated: BatchNorm uses batch statistics during training.
Evidence: `results/exact_duration.json` (`causality_summary`), `tests/test_controls_and_causality.py`.

## 8. The architecture was chosen on the test set

**Disclosed and bounded.** READER was selected among 8 arms on an 8-subject screen scored on the official test
session. On the 34 subjects the screen never used: READER − Compact +1.43 (screen subjects +1.92), READER − BiTE
+0.57 (screen +0.70). The gain shrinks by about a quarter but does not vanish.
Evidence: `results/DEVELOPMENT_DISCLOSURE.md`.

## 9. Cross-subject evaluation is a single seed

**PENDING (3 seeds).** Seeds 2026 and 2027 for READER and Compact, and all three seeds for BiTE. A defect was
found and fixed while doing this: the first BiTE cross-subject seed was trained with gradient clip 5 (our trainer's
default), while BiTE's release does not clip and every within-subject BiTE run uses `--clip 0`. The clip-5 runs
are kept in the table in italics as superseded. Seed-2025 state: parity, SD-SSVEP READER − Compact +1.11
[+0.06, +2.11]. Evidence: `results/CROSS_SUBJECT.md`.

## 10. GPU nondeterminism

**Measured; re-runs in progress.** H100 80GB and H200 NVL reproduce each other bit-for-bit (45/45). MIG-partitioned
slices do not (one READER 2a run moved 5.56 pp). 147 runs of comparisons of record ran on MIG slices; each is re-run
on H200 and the re-run becomes the record. So far 28 BiTE pairs (all 27 within-subject 2a runs and one more): MIG
minus full GPU −0.02 pp (se 0.07), 13 pairs identical, i.e. noise without bias. 119 re-runs are PENDING. Evidence: `results/REPRODUCIBILITY.md`,
`results/raw/record_provenance.csv` (which GPU and which rule selected every run).

## 11. Final-epoch reporting without a validation set

**Matched deliberately, measured.** BiTE's protocol (600 epochs, no validation set, final epoch) is used for
every model. Every model loses accuracy from its own test-curve peak to the final epoch: READER 1.89, Compact
2.04, BiTE 2.06, zoo baselines 2.20–4.54 pp. The protocol does not favour READER over BiTE; it costs the zoo
baselines somewhat more, which is stated. The peak is a diagnostic only and is never used for selection.
Evidence: `results/REPRODUCIBILITY.md`, `results/raw/runs.csv` (`test_peak_*_diagnostic`).

## 12. Parameters, latency, and the O(T²) anytime curve

**Parameters: measured. Latency: PENDING a clean measurement.** READER has 19.0K–32.4K parameters against BiTE's
14.5K–17.9K, so no efficiency claim is made for a single model; one READER does replace a bank of four BiTE
specialists (58K–72K parameters in total). Recomputing the whole anytime curve from scratch is O(T²); a single
decision at deadline t reads t tokens (O(t)), and the timed code path is asserted equal to the model's output.
The CPU timings in `results/EFFICIENCY.md` were taken on a shared cluster node and repeated measurements of the
same configuration differed by up to 3×, so no latency ordering between READER and BiTE is claimed yet.
Evidence: `results/EFFICIENCY.md`, `tests/test_results_pipeline.py::test_single_decision_path_equals_the_model`.

## 13. Are the baselines trained fairly?

**Stated.** Zoo baselines are BiTE's released code built by BiTE's own `get_model`, unchanged, trained with this
repository's recipe (float32, gradient clip 5); BiTE's own trainer uses AMP and no clipping. BiTE itself is
trained with `--clip 0`. The seed-2025 re-runs track BiTE's published table within about 1 pp on average
(`results/MAIN_TABLE.md`, "How closely the re-runs track"). Preprocessing is BiTE's own `preprocess_*` code
(`scripts/prepare_data.py`, checksums in `results/data_checksums.json`).

## 14. Kappa

**Reported.** Cohen's kappa from the saved final-epoch predictions for every run (`results/raw/runs.csv`) and in
`results/MAIN_TABLE.md`.

## 15. HGD coverage

HGD has the endpoint comparison (READER, Compact, BiTE, zoo PENDING) and the gate interventions, but no
FF-Control, Compact-Mean, exact-duration, prefix-supervision or bank runs: its 14 subjects make each READER run
about one GPU-hour. Stated as a limitation.
