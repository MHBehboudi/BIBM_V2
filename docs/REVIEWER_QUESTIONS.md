# Questions a reviewer will ask, and where the answer is

Each entry states the concern, what was done about it, the evidence file, and what remains open. Numbers are
subject-level (seeds averaged per subject) with 95% paired bootstrap CIs unless marked otherwise. The runs launched 2026-09-17 under
`docs/RUN_CARD_PAPER_COMPLETION.md` (HGD baseline zoo, cross-subject seeds 2026-27, BiTE at `--clip 0`, and the
full-GPU re-run of every cell that had run on a MIG slice) have all landed and every file below is rebuilt from
its complete record. What remains open is marked **Open** and is open for a stated reason, not for a missing run.

## 1. Does READER beat the state of the art?

**Done.** Every baseline BiTE ships (ATCNet, DMSANet, DeepConvNet, EEGNeX, EEGNet, EEGTCNet, EISATC, FACTNet,
MBCNNEATCFNet, ShallowConvNet) and BiTE itself were re-run with 3 seeds under the identical protocol, so the
comparison is paired rather than against single-seed published numbers. Our seed-2025 re-runs track BiTE's
published seed-2025 table closely (mean absolute gap about 1 pp), so the harness does not handicap the baselines.
Evidence: `results/MAIN_TABLE.md`, `results/model_zoo/MODEL_ZOO.md`, `results/figures/fig_reader_vs_zoo.pdf`.

**Answer, stated as it is.** READER minus BiTE (re-run): 2a +0.60 [−0.80, +1.94], 2b −0.89 [−1.84, +0.10],
HGD +0.77 [+0.03, +1.60], SD-SSVEP +1.94 [+0.72, +3.39]. At the endpoint READER is at **parity with BiTE on motor
imagery** (both intervals span zero, 2b leans to BiTE) and ahead of BiTE on SD-SSVEP (Holm p .047 over the eleven
SD-SSVEP comparisons). It beats every other re-run baseline on 2a and 2b, and is level with DeepConvNet on SD-SSVEP
(−0.33 [−1.61, +0.67]). READER ranks 1 of 12 on 2a and HGD and 2 of 12 on 2b and
SD-SSVEP, and is first on the corpus-balanced mean (90.90 vs BiTE 90.29). Against the published single-seed bars it is −0.63 (2a), −1.96 (2b), +0.37 (HGD), +0.67
(SD-SSVEP). The paper's claim is therefore not "higher endpoint accuracy everywhere" but endpoint parity or
better **plus** a legal decision at every deadline from one model (sections 5–6).

**Open.** On HGD the strongest re-run baseline is DMSANet (96.11), not BiTE, and READER − DMSANet is
+0.19 [−1.69, +1.79], 8/0/6 — rank 1 on HGD is not a significant margin and is not claimed as one.

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

**Done, 3 seeds.** A defect was found and fixed while doing this: the first BiTE cross-subject seed was trained
with gradient clip 5 (our trainer's default), while BiTE's release does not clip and every within-subject BiTE run
uses `--clip 0`. All three BiTE seeds were re-run at `--clip 0`; the clip-5 runs stay in the table in italics as
superseded. Result over 3 seeds: READER − BiTE is −0.23 [−2.58, +2.01] on 2a, −0.32 [−1.53, +1.02] on 2b and
+1.63 [−0.15, +3.50] on SD-SSVEP — **parity, every interval spanning zero**. READER − Compact is +0.98
[+0.20, +1.70] on 2a and +0.93 [−0.46, +2.46] on SD-SSVEP. The prefix-reversed branch is a within-subject result
and is not claimed to transfer across subjects. Evidence: `results/CROSS_SUBJECT.md`.

**Open.** BiTE's ten baselines were not run cross-subject. A LOSO run costs 1.15 GPU-h against 0.32 within-subject,
so that zoo is about 965 GPU-h against the 290 already spent; the cross-subject table is presented as a
three-model comparison rather than implied to be a zoo.

## 10. GPU nondeterminism

**Done, all 147 pairs.** H100 80GB and H200 NVL reproduce each other bit-for-bit (45/45). MIG-partitioned slices do
not. Every one of the 147 runs of a comparison of record that had run on a MIG slice was re-run on an H200 and the
re-run is now the record; the MIG original is kept under `runs/mig/` so the pair can be compared. MIG minus full
GPU, by arm: within-subject BiTE +0.04 pp (se 0.06, 48 pairs), BiTE anytime banks −0.13 / +0.26 / −0.02 (se ≤ 0.22,
23 pairs each), LOSO Compact −0.00 (se 0.41, 8 pairs) — **noise without bias**.

**The exception, stated plainly:** READER's prefix-supervised anytime arm moved +1.83 pp (se 1.35) over its 22
pairs, with 21 of 22 differing and a mean absolute difference of 3.08 pp. READER's 2a training is the least
reproducible thing in this repository (see also the seed spread in `results/ERROR_ANALYSIS.md`: 1.91 pp SD across
seeds against BiTE's 1.34). This is why the record uses the full-GPU re-runs and why the anytime intervals are
reported with subject-level CIs rather than point estimates. Evidence: `results/REPRODUCIBILITY.md`,
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

HGD has the full endpoint comparison (READER, Compact, BiTE and all ten baselines, 3 seeds) and the gate
interventions, but no
FF-Control, Compact-Mean, exact-duration, prefix-supervision or bank runs: its 14 subjects make each READER run
about one GPU-hour. Stated as a limitation.

## 16. Per-class behaviour, calibration, and are the errors even different?

**Done, without a GPU.** Every run of record stores its final-epoch test logits, and those logits reproduce the
accuracy in the run's `summary.json` exactly (checked for all 2,647 runs; a mismatch aborts the script). So the
class-level and confidence-level analyses are the same runs at the same epoch as the main table, not a second
measurement. Evidence: `results/ERROR_ANALYSIS.md`, `results/error_analysis.json`,
`results/figures/fig_confusion.pdf`, `fig_complementarity.pdf`, `fig_calibration.pdf`.

- **No collapsed class.** READER's per-class recall spread is 5.9 pp on 2a and 8.7 on SD-SSVEP, against BiTE's
  6.6 and 17.3 (BiTE's twelfth SSVEP class falls to 81.3 where READER holds 90.7).
- **Calibration.** Every arm, including the baselines, is **under**-confident on every corpus (READER 2a
  −15.4 pp, SD-SSVEP −33.2). That is the expected direction for label smoothing 0.1 and grows with class count,
  so it is a property of the shared recipe, not of the architecture. The consequence is real anyway: a confidence
  threshold tuned on one corpus does not transfer, which matters for any deployment that stops early when sure.
- **READER and BiTE fail on different trials.** On 2a they disagree on 13.2% of test trials (6.9 only READER,
  6.3 only BiTE) and both miss only 9.0%. An oracle over the two is +6.30 pp [+4.19, +8.55] above READER, 9/9
  subjects; averaging the two softmax outputs is +1.79 [+0.95, +2.80]. This is reported as a diagnostic and not
  as a proposed system: it needs both models at inference, which is exactly the cost this paper removes. It does
  say that the endpoint parity in section 1 is not two models making the same predictions.

## 17. Why that prefix weight, and why that pooling factor? Were they chosen to flatter READER?

**Two sweeps, in progress, 1008 runs.** Design, values and reasoning: `docs/SENSITIVITY_ANALYSIS.md`;
numbers as they land: `results/SENSITIVITY.md`; recipe: `scripts/reproduce_sensitivity.sh`.

Both settings were declared, not tuned, and neither has ever been selected on a result — but "declared" is
not evidence, so both are now swept with the forward-only control re-trained at **every** setting, because
the quantity that matters is the paired difference and not either arm's level.

- **Prefix-supervision weight.** λ ∈ {0, 0.1, 0.3, 1.0} at the default pooling factor. λ = 0 and λ = 0.3
  are the paper's existing arms, so the sweep spans no supervision to strong supervision. Across the largest
  change it already contains (λ = 0 → 0.3) the nAUC difference stays positive on all three corpora:
  +30.72 / +11.06 / +42.43 at λ = 0 against +9.31 / +1.13 / +12.30 at λ = 0.3.
- **Token resolution.** `p` giving ≈16 / 32 / 64 tokens on every corpus (2a/2b `p` ∈ {64, 32, 16};
  SD-SSVEP `p` ∈ {16, 8, 4}), at λ = 0 and λ = 0.3. Every condition stays inside the reader's 71-token
  receptive field, so a difference between them is about resolution and not about unreachable history.
  `p` = 8 on SD-SSVEP is BiTE's own released setting; `p` = 16 there puts the 16 Hz token rate below twice
  the 9.25–14.75 Hz carriers, so a loss at that point is a prediction of the sweep, not a surprise.

A setting where the declared value is not the best cell is reported as such. **Nothing in the paper's main
tables is re-selected from these runs**; the sweep can only tell you whether the reverse reader's advantage
survives moving the knob.

**One defect this exposed, reported rather than quietly fixed.** In the development harness, the forward-only
arm's builder silently dropped unrecognised options, so a misspelt `--option pool=16` would have trained the
corpus default and been reported as a pooling result. It now raises. No published number was affected: the
defect was found in the pre-flight, before any sweep cell was trained.

## 18. Which integration range do the nAUC numbers use?

The nAUC values quoted in the paper (+9.31 / +1.13 / +12.30 for READER+PS − no-reverse+PS) integrate from
**1.0 s** on 2a/2b and **0.25 s** on SD-SSVEP, not from the first evaluated point (0.5 s / 0.125 s). Over the
full evaluated grid the same comparison is **+10.88 / +1.77 / +13.15**. Both are computed by
`reader/sensitivity.py` (`--full-range` selects the second), and `results/SENSITIVITY_FULL_RANGE.md` carries
the full-range table so the two can be compared directly.
