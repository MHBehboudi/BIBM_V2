# Prefix-bidirectional reading for anytime EEG decoding

One model that emits a legal decision at every token boundary. Given only part of each trial, endpoint-trained
READER holds 70-82% accuracy on 2a at 1-3 s where the same forward-only parent checkpoint collapses to 31-50% (same
checkpoints, exact truncated input), and the advantage survives giving both models the identical prefix loss
(+19.2 points at 1 s on 2a, +22.2 at 0.25 s on SD-SSVEP). Against a bank of separately retrained per-deadline
specialists, one prefix-supervised READER wins on 2a (+2.01 at 1 s, +1.80 at 2 s) but not on 2b (a flat −1.5, the
endpoint gap) or SD-SSVEP (−3.3 at 0.5 s against retrained Compact specialists).

Bidirectional context in EEG decoders is bought at the cost of deployability. BiTE's BiTCN obtains
backward context by reversing the **complete trial** — its backward branch's last step corresponds
to the trial's first moment, and its STFT front end uses `center=True`, reading 128 ms into the
future. No output exists until the trial ends, so a deployed system must train and hold a separate
model per decision deadline.

**READER** applies the backward branch to the **observed prefix** instead. At deadline *t* a second
causal TCN reads tokens *t−1 … 0* and is averaged with the forward reading (a convex per-feature gate that
stays at its equal-weight initialisation; pinning it there is free). Reversed over the trial is non-causal;
reversed over the prefix is not.

**Where everything is.** `results/README.md` maps every paper table and figure to its file and script;
`docs/REVIEWER_QUESTIONS.md` answers the questions a reviewer will ask, with evidence and open items;
`results/raw/runs.csv` holds every run so any number can be recomputed without a GPU.
**`results/dashboard.html`** is a single self-contained page over all of it — open it in a browser, no server and
no network: pick a corpus and see where READER sits among twelve models, which subjects the mean is made of, what
it does before the trial ends, and which trials it gets wrong.

## Results

### Within-subject, against every baseline re-run with 3 seeds (`results/MAIN_TABLE.md`)

Official roles (session 1 train, session 2 test), 600 epochs, fixed final epoch, no validation set, no model
selection, seeds 2025-2027. BiTE and its ten released baselines were **re-run under the identical protocol**, so
the comparison is paired and multi-seed; BiTE's published single-seed numbers are shown beside. Mean ± SD across
subjects. Strongest re-run baselines shown; all eleven are in `results/MAIN_TABLE.md` and
`results/model_zoo/MODEL_ZOO.md`.

| model | params (2a) | 2a | 2b | HGD | SD-SSVEP |
|---|---:|---:|---:|---:|---:|
| ATCNet (re-run) | 113.7K | 81.29 ± 8.46 | 84.07 ± 8.81 | 95.67 ± 2.99 | 85.33 ± 17.57 |
| MBCNNEATCFNet (re-run) | 29.5K | 81.76 ± 7.87 | 84.40 ± 7.91 | 93.72 ± 4.00 | 94.22 ± 7.29 |
| DeepConvNet (re-run) | 102.3K | 71.26 ± 15.23 | 84.95 ± 9.84 | 93.68 ± 3.23 | **96.50** ± 7.77 |
| BiTE (re-run) | 16.3K | 84.10 ± 8.04 | **87.30** ± 7.14 | 95.53 ± 3.41 | 94.22 ± 8.69 |
| BiTE (published, 1 seed) | 14.5K | 85.34 | 88.37 | 95.93 | 94.16 |
| Compact (READER without the reversed branch) | 17.8K | 82.33 ± 10.06 | 85.32 ± 8.06 | 95.69 ± 2.73 | 94.11 ± 11.13 |
| **READER** | 21.0K | **84.71** ± 8.27 | 86.41 ± 7.30 | **96.30** ± 2.79 | 96.17 ± 7.89 |

READER minus BiTE (re-run), subject-level with 95% bootstrap CI: 2a +0.60 [−0.80, +1.94], 2b −0.89 [−1.84, +0.10],
HGD +0.77 [+0.03, +1.60], SD-SSVEP +1.94 [+0.72, +3.39]. **At the endpoint READER is at parity with BiTE on motor
imagery and ahead of it on SD-SSVEP**; it is above every other re-run baseline on 2a and 2b, and level with the
re-run DeepConvNet on SD-SSVEP (READER − DeepConvNet −0.33 [−1.56, +0.67], 3/4/3 subjects). Against the published
single-seed bars: 2a −0.63, 2b −1.96, HGD +0.37, SD-SSVEP +0.67 (DeepConvNet's 95.50). **READER ranks 1 of 12 on 2a
and HGD and 2 of 12 on 2b and SD-SSVEP, and is first on the corpus-balanced mean (90.90 against BiTE's 90.29).**
On HGD the strongest re-run baseline is DMSANet (96.11), not BiTE, and READER − DMSANet is +0.19 [−1.69, +1.79]:
rank 1 there is not a significant margin.

### Cross-subject (`results/CROSS_SUBJECT.md`)

Leave-one-subject-out as BiTE defines it; BiTE publishes no HGD row. **Three seeds, complete**, BiTE trained at
`--clip 0` as released:

| model | 2a | 2b | SD-SSVEP |
|---|---:|---:|---:|
| BiTE (re-run, 3 seeds) | **61.32** ± 12.83 | 76.17 ± 6.29 | 78.94 ± 20.79 |
| Compact | 60.11 ± 14.44 | **76.48** ± 6.48 | 79.65 ± 21.69 |
| **READER** | 61.10 ± 14.36 | 75.86 ± 6.86 | **80.57** ± 21.41 |
| BiTE (published, 1 seed) | 64.56 | 76.44 | 79.72 |

**Parity on all three, and none of the three architectures transfers across subjects the way it does within one.**
READER − BiTE: 2a −0.23 [−2.58, +2.01], 2b −0.32 [−1.53, +1.02], SD-SSVEP +1.63 [−0.15, +3.50] — every CI spans
zero. The prefix-reversed branch is a within-subject result; we do not claim a cross-subject one. The first BiTE
seed was trained at gradient clip 5 (our trainer's default) whereas BiTE's release does not clip; all three seeds
were re-run at `--clip 0` and the clip-5 numbers (61.11 / 76.08 / 79.44) are marked superseded in the table.

The ten baselines were **not** run cross-subject: at the measured 1.15 GPU-h per LOSO run that zoo is ~965 GPU-h,
against 290 for the three models above. The cross-subject comparison of record is therefore READER, Compact and
BiTE only, and it is stated that way rather than implied to be a zoo.

### Anytime: one model against retrained per-deadline specialists (`results/anytime_*.md`)

**One** READER read at each deadline vs a **bank of separately retrained** specialists (one model per deadline,
trained from scratch on truncated trials). Subject-level difference against the strongest bank at each deadline,
9 subjects × 3 seeds:

| deadline | READER (1 model) | BiTE bank | Compact bank | READER − strongest bank [95% CI], W/T/L |
|---|---:|---:|---:|---|
| 1.0 s | 76.84 | 74.27 | 74.83 | **+2.01** [+0.76, +3.09] 7/0/2 |
| 2.0 s | 83.37 | 81.49 | 81.57 | **+1.80** [+0.48, +2.89] 7/1/1 |
| 3.0 s | 84.10 | 82.93 | 83.35 | +0.76 [−0.72, +2.42] 5/1/3 |
| 4.0 s | 84.84 | 84.08 | 82.33 | +0.76 [+0.01, +1.63] 6/1/2 |

With 9 subjects, Holm over the three early deadlines gives p = .082 / .082 / .53: the 1 s and 2 s intervals exclude
zero but do not survive correction. An earlier version of this table headlined the delta against the *weaker* bank
(+2.57 / +1.88 / +1.17).

**It does not replicate on the other two corpora.** SD-SSVEP (second paradigm, 10 subjects × 3 seeds,
`results/anytime_sdssvep.md`):

| deadline | READER (1 model) | BiTE bank | Compact bank | READER − strongest bank [95% CI], W/T/L, Holm p |
|---|---:|---:|---:|---|
| 0.25 s | 65.33 | 66.11 | 67.17 | −1.83 [−5.17, +2.06] 2/0/8, .28 |
| 0.5 s | 84.28 | 84.11 | 87.56 | **−3.28** [−4.89, −1.83] 0/0/10, .006 |
| 0.75 s | 88.00 | 89.00 | 91.72 | −3.72 [−6.44, −1.06] 2/2/6, .094 |
| 1.0 s | 94.33 | 94.22 | 94.11 | +0.11 [−1.33, +1.56] 5/3/2 |

READER is level with BiTE's SSVEP bank but Compact models retrained at each deadline beat it. On 2b READER trails
BiTE's bank by a flat −1.50 / −1.62 / −1.44 / −1.59 at 1 / 2 / 3 / 4 s, which is BiTE's endpoint lead
(`results/anytime_2b.md`). **One READER replacing the bank without losing accuracy is therefore a 2a result.**

### Beyond accuracy: classes, confidence and whose errors (`results/ERROR_ANALYSIS.md`)

Computed from the final-epoch test logits stored with every run, which reproduce each run's reported accuracy
exactly — same runs, same epoch, no GPU.

- **Per-class recall.** No arm has a collapsed class. READER's worst-to-best spread is 5.9 pp on 2a and 8.7 on
  SD-SSVEP, against BiTE's 6.6 and 17.3; BiTE's twelfth SSVEP class falls to 81.3 where READER holds 90.7.
- **Calibration.** Every arm is **under**-confident on every corpus (2a −15.4 pp, SD-SSVEP −33.2), the expected
  direction for label smoothing 0.1 rather than a property of the architecture. A confidence threshold tuned on
  one corpus will not transfer to another — which matters for any system that stops early when it is sure.
- **READER and BiTE make different mistakes.** On 2a they disagree on 13.2% of trials (6.9 only READER, 6.3 only
  BiTE) and both miss only 9.0%. The oracle over the two is +6.30 pp [+4.19, +8.55] above READER, 9/9 subjects;
  averaging their softmax outputs is +1.79 [+0.95, +2.80]. That is reported as a diagnostic, not a system: it
  needs both models at inference, which is the cost this paper set out to remove.
- **Re-run noise.** READER's 2a accuracy moves 1.91 pp SD across seeds within a subject (max 5.54), against BiTE's
  1.34. Every delta in this repository is a difference of numbers with that much spread underneath it.

### Ablations (`results/ABLATION.md`)

| variant (minus READER, subject-level) | 2a full | 2b full | SD-SSVEP full | 2a at 1 s | SD-SSVEP at 0.25 s |
|---|---:|---:|---:|---:|---:|
| − reversed branch (Compact) | −2.38 | −1.09 | −2.06 | −39.17 | −24.67 |
| reversed → forward second branch (FF-Control, same size and init) | −1.63 | −1.26 | −2.00 | −39.85 | −23.72 |
| second branch → running-mean readout (Compact-Mean) | −12.35 | −1.36 | −29.72 | −29.66 | −6.33 |
| forward route only, g = 0 (inference, same weights) | −0.60 | −0.70 | −1.83 | −37.17 | −24.06 |
| reversed route only, g = 1 (inference, same weights) | −2.12 | −0.77 | −8.83 | +1.38 | +4.89 |
| equal average, g = 0.5 (inference, same weights) | −0.06 | +0.02 | +0.11 | +0.19 | +0.50 |
| + prefix supervision (loss) | +0.13 | −0.67 | −1.83 | +6.73 | +30.72 |

Early columns are the same checkpoints given only the first quarter of each trial. CIs, W/T/L and HGD are in
`results/ABLATION.md`.

**Same loss, different architecture.** Prefix supervision is a loss change, so Compact was retrained with the
identical prefix loss. READER+PS minus Compact+PS on the same truncated inputs, subject-level [95% CI]:

| | 2a 1 s | 2a 2 s | 2b 1 s | 2b 2 s | SD-SSVEP 0.25 s | SD-SSVEP 0.5 s |
|---|---:|---:|---:|---:|---:|---:|
| early decision | **+19.20** [+15.74, +23.30] 9/0/0 | **+10.12** [+6.31, +14.99] 9/0/0 | +3.89 [+0.08, +7.10] 7/0/2 | +1.42 [−2.11, +5.02] 5/0/4 | **+22.22** [+16.89, +28.11] 10/0/0 | **+13.11** [+6.78, +19.94] 9/0/1 |

At the full trial the same comparison is 2a +2.74 [−0.08, +6.46], 2b −0.75 [−1.42, −0.17], SD-SSVEP +0.22
[−1.67, +2.44]. The prefix-reversed branch, not the loss, carries the early-decision advantage of a single model;
what it does not do is match specialists retrained for each deadline on 2b and SD-SSVEP (section above).

### Efficiency (`results/EFFICIENCY.md`)

One READER replaces a bank of four per-deadline BiTE specialists (e.g. 21.0K vs 65.2K parameters on 2a). A single
READER has 1.3–2.2× BiTE's parameters, so no single-model efficiency claim is made. **CPU latency is provisional**:
it was timed on a shared node and repeated measurements of the same configuration differed by up to 3×, so no
latency comparison is claimed until it is re-measured on an idle node.

### Scope of the anytime claim

Three limits, all of them load-bearing:

1. **It requires prefix supervision, which is a different arm.** The table above is the reader
   trained with `--prefix-weight 0.3` (deep supervision over deadlines) — a *loss* change, not the
   architecture. The endpoint-supervised reader from the within-subject table above *loses* **−4.55
   at 1 s** against the strongest bank and is then within ±1.3 points of it (+0.59 / −1.22 / +0.63 at
   2/3/4 s) — it does not carry the anytime-vs-bank claim on its own
   (`results/anytime_2a_endpoint_supervised.md`). Prefix supervision is free on 2a
   (+0.13 endpoint) and costs −1.83 on SD-SSVEP, so it is not a global default and the
   within-subject table is not built on it. Which arm produced an anytime number is therefore part
   of that number; `reader/anytime.py --reader` selects it and the generated tables name it.
2. **It is a 2a claim.** On 2b the same comparison is **−1.50 / −1.62 / −1.44 / −1.59** at
   1/2/3/4 s (`results/anytime_2b.md`) — a flat offset, statistically indistinguishable across
   deadlines and equal to BiTE's standing 2b *endpoint* advantage. On SD-SSVEP, Compact specialists
   retrained at each deadline beat the single READER at 0.5 s (−3.28, Holm p .006) and 0.75 s (−3.72)
   (`results/anytime_sdssvep.md`).
3. **No bank exists on HGD**, so no comparison is available there.

### Ablation: does supervising the intermediate decisions help? (`results/ablation_prefix_supervision.md`)

READER produces a decision at every deadline whether or not anything supervises it. This is the
one focused ablation of that: endpoint-only loss vs **one predeclared** prefix weight of 0.3. No
sweep — 0.3 is the only weight ever trained on this model, so nothing is selected on the test set.
Same model, subjects, seeds, epochs and protocol; the loss is the only difference, and
`reader/ablation.py` aborts if the two arms differ in anything else. Paired per (subject, seed).

| corpus | early deadline | endpoint-only → prefix | paired Δ (se, W/L) | endpoint Δ (se, W/L) |
|---|---|---:|---:|---:|
| 2a | 1 s | 70.28 → 76.84 | **+6.56** (0.95, 24/3) | +0.13 (0.54, 9/15) |
| 2b | 1 s | 70.79 → 75.92 | **+5.14** (1.81, 18/8) | −0.67 (0.42, 7/18) |
| SD-SSVEP | 0.25 s | 34.61 → 65.33 | **+30.72** (2.00, 29/1) | −1.83 (0.48, 1/14) |
| SD-SSVEP | 0.5 s | 62.00 → 84.28 | **+22.28** (2.43, 30/0) | — |

**Deep supervision of the intermediate decisions buys early accuracy on every corpus tested, and
the endpoint is preserved only on 2a.** On 2b it costs −0.67 and on SD-SSVEP −1.83, so it is a
trade, not a free improvement, and the within-subject table above is deliberately *not* built on
it. The effect is largest where the endpoint-only model is worst early: an SSVEP decoder trained
only on the trial end is barely above chance at 250 ms (34.61 against a 12-class chance of 8.33)
because nothing ever asked it for an early answer.

SD-SSVEP matters here for a second reason: it is a **different paradigm** from the motor imagery of
2a/2b, and its 1 s trial on a 15.625 ms grid is a different deadline regime. It carries no
specialist bank, so it cannot enter the comparison above, but it can enter this one.

## Is the prefix-reversed branch doing anything? (`results/gate_intervention.md`)

The forward and prefix-reversed readings are fused by a convex per-feature gate
`g = sigmoid(gamma)`, `gamma` initialised at 0 so both routes start live at 0.5. Pinning `g` at
inference on already-trained weights separates two questions that are easy to conflate. Paired per
cell against that cell's own learned-gate accuracy (se, cells made worse):

| corpus | n | learned | g=0 forward only | g=1 reversed only | g=0.5 (init, unlearned) | learned gate |
|---|---:|---:|---:|---:|---:|---:|
| 2a | 27 | 84.71 | −0.60 (0.35, 19/27) | −2.12 (0.49, 21/27) | −0.06 (0.05, 8/27) | 0.4943 ± 0.0179 |
| 2b | 27 | 86.41 | −0.70 (0.23, 16/27) | −0.77 (0.29, 15/27) | +0.02 (0.07, 4/27) | 0.4969 ± 0.0156 |
| HGD | 42 | 96.30 | −0.44 (0.14, 22/42) | −0.30 (0.14, 17/42) | +0.00 (0.03, 2/42) | 0.5003 ± 0.0142 |
| SD-SSVEP | 30 | 96.17 | −1.83 (0.57, 12/30) | −8.83 (1.61, 26/30) | +0.11 (0.11, 1/30) | 0.4484 ± 0.0378 |

**The fusion is load-bearing: deleting either route costs accuracy on all four corpora**, so the
prefix-reversed reading carries information the forward reading does not, and vice versa — the
reversed branch is not decorative. On SD-SSVEP the forward reading is doing most of the work
(−8.83 to drop it, −1.83 to drop the reversed one); on 2a it is the other way round.

**The gate's *learning*, however, contributes nothing.** It sits within a few thousandths of its
0.5 initialisation on every corpus, and pinning it exactly there is free (−0.06 to +0.11, all
inside noise). We therefore describe the fusion as a **fixed equal-weight convex average**, not as
a learned adaptive gate; `gamma` is kept only because it costs 64 parameters and leaves the door
open on corpora we have not tried.

## Does reverse reading improve the SAME checkpoint's intermediate predictions? (`results/subject_stats/`)

The question that matters for the mechanism is not whether READER beats separately retrained models, but
whether, with **no retraining**, it predicts better from partial input than its own forward-only parent.
Each endpoint-trained checkpoint is given **only the first n samples** of every test trial: the last pooling
window is partial and rescaled exactly as at deployment, and the prediction is that truncated input's
endpoint — never a value read off the full-trial curve (`reader/exact_duration.py`, 420 checkpoints; every
one reproduces its logged full-length accuracy exactly). Subject-level means (seeds averaged per subject):

| corpus | observed | READER | Compact | FF-Control | Compact-Mean | BiTE (same ckpt) |
|---|---|---:|---:|---:|---:|---:|
| 2a | 1 s / 2 s / 3 s / 4 s | **70.1 / 82.1 / 82.0 / 84.7** | 30.9 / 45.1 / 50.4 / 82.3 | 30.2 / 50.8 / 58.4 / 83.1 | 40.4 / 65.2 / 71.1 / 72.4 | 59.9 / 76.9 / 70.4 / 84.1 |
| 2b | 1 s / 2 s / 3 s / 4 s | **70.9 / 83.0 / 85.4 / 86.4** | 54.3 / 71.6 / 73.8 / 85.3 | 54.6 / 70.1 / 75.7 / 85.2 | 62.7 / 81.7 / 84.2 / 85.1 | 68.1 / 83.2 / 82.3 / 87.3 |
| SD-SSVEP | .25 / .5 / .75 / 1 s | **34.6 / 62.0 / 69.9 / 96.2** | 9.9 / 13.8 / 14.1 / 94.1 | 10.9 / 16.7 / 16.1 / 94.2 | 28.3 / 48.6 / 60.9 / 66.4 | 18.8 / 29.4 / 32.9 / 94.2 |

READER minus Compact on the same checkpoints, subject-level mean, 95% bootstrap CI, subjects W/T/L; exact
sign-flip Wilcoxon, Holm-adjusted over the 12 pre-declared prefix tests:

| corpus | 1/4 of the trial | 1/2 | 3/4 | full |
|---|---|---|---|---|
| 2a | +39.2 [+34.6, +44.4] 9/0/0, Holm p .043 | +37.0 9/0/0, .043 | +31.6 9/0/0, .043 | +2.4 [+0.2, +4.7] 6/0/3, .125 |
| 2b | +16.5 [+9.5, +23.3] 8/0/1, .043 | +11.5 8/0/1, .047 | +11.6 9/0/0, .043 | +1.1 [+0.4, +1.8] 7/0/2, .094 |
| SD-SSVEP | +24.7 [+14.6, +34.6] 9/0/1, .043 | +48.2 9/0/1, .043 | +55.8 10/0/0, .023 | +2.1 [+0.3, +4.4] 5/5/0, .125 |

With 9 subjects the smallest attainable exact p is 0.0039, which bounds how small these can get after Holm.

**Read this with its main caveat.** Compact classifies its *last* token, whose position moves with the
deadline and whose classifier was only ever trained at the final position; READER's reversed branch ends at
the trial's *first* token at every deadline. Part of the gap is therefore **anchoring**, not reversal as
such. The gate interventions agree: reversed-only (g = 1) is *better* than the learned gate early (2a +1.4 at
1 s, SD-SSVEP up to +11.7) and worse at the endpoint, and forward-only (g = 0) collapses like Compact.

## Beyond a second branch, or a better readout? (FF-Control, Compact-Mean)

Two controls, each trained from scratch with the identical recipe on all 28 subjects x 3 seeds:

- **FF-Control** (`--model ff_control`): READER's second branch with the *same* constructor, forked seed,
  gate, initialisation and objective, reading the prefix in **original** order. Parameter count and initial
  weights are identical to READER's (tested).
- **Compact-Mean** (`--model compact_mean`): the parent with its classifier on the causal running mean of the
  forward TCN outputs, (1/t) sum f_i, **trained** with that readout. Parameters identical to Compact (tested).

Endpoint, READER minus comparator, subject-level (95% bootstrap CI, W/T/L, Holm over 12 endpoint tests):

| corpus | vs Compact | vs FF-Control | vs Compact-Mean | vs BiTE (this harness) |
|---|---|---|---|---|
| 2a | +2.38 [+0.23, +4.63] 6/0/3 | +1.63 [+0.27, +3.07] 6/0/3 | **+12.35** [+6.94, +18.58] 9/0/0, Holm .043 | +0.63 [−0.82, +1.98] |
| 2b | +1.09 [+0.39, +1.79] 7/0/2 | +1.26 [+0.34, +2.19] 8/0/1 | +1.36 [−0.79, +3.57] 6/0/3 | −0.92 [−1.91, +0.06] |
| SD-SSVEP | +2.06 [+0.33, +4.39] 5/5/0 | +2.00 [+0.50, +4.17] 6/4/0 | **+29.72** [+19.00, +41.06] 10/0/0, Holm .023 | +1.94 [+0.78, +3.44] |

- **A second forward branch adds nothing measurable**: FF-Control minus Compact is +0.75 [−0.18, +1.79] (2a),
  −0.16 [−0.82, +0.51] (2b), +0.06 [−0.72, +1.00] (SD-SSVEP). READER's endpoint gain over Compact is therefore
  not explained by capacity or by ensembling two branches; READER minus FF-Control is about as large as READER
  minus Compact on every corpus, with every interval above zero. None survives Holm at 12 comparisons.
- **Compact-Mean is not a READER substitute at the endpoint** (72.4 / 85.1 / 66.4): uniform averaging gives
  early, not-yet-informative tokens the same weight as late ones and generalises poorly (it still fits the
  training set to 100%). Its position-free readout does recover much of Compact's prefix collapse (2a at 2 s
  45.1 -> 65.2), which confirms that anchoring matters; READER still leads it at every duration on 2a and
  SD-SSVEP, while on 2b beyond 1.5 s the lead (+1.2 to +2.2) has intervals that include zero.
- FF-Control is end-anchored like Compact, so it does not separate *direction* from *start-anchoring*.

Full tables, per-subject differences and the exact tests: `results/subject_stats/SUBJECT_STATS.md`.

## Is it actually causal?

Yes, and it is tested rather than asserted, on random **and trained** weights.

**On trained checkpoints** (`reader/exact_duration.py`, `results/exact_duration.json`): for every READER,
FF-Control and Compact-Mean checkpoint on 2a / 2b / SD-SSVEP (252 checkpoints, 10,782 pooling-boundary checks in
total, all test trials), the maximum |logit| difference between the truncated input and the full-trial curve
is **0.0**, and replacing every sample after a boundary with N(0, 1e3^2) noise changes the earlier decisions
by **0.0**. All BatchNorm layers track running statistics, and eval-mode outputs do not depend on batch
composition.

**Why random-init tests are not enough** (`tests/test_controls_and_causality.py`). Every residual TCN block
zero-initialises its second convolution, so at initialisation each TCN is exactly the identity and the
reverse branch's last output is token 0 whatever it reads. A deliberately wrong implementation that reverses
the **complete** trial at every deadline is therefore indistinguishable from READER at init — a test pins
this. The causality tests run on an **activated** configuration instead (second convolutions, BatchNorm
statistics and gate randomised), where that mutant is detected and all four causal arms pass:

- **truncation equivalence** — feeding only the first *t* tokens reproduces entry *t* of the anytime
  curve to <1e-4. This is the property deployment needs.
- **no future leakage** — replacing everything after the deadline with 1e3-scale noise leaves early
  deadlines bit-identical.
- **deadline labels** — the reported token must *end at or after* the wall-clock deadline
  (`reader.anytime.deadline_index` uses `ceil`; `round` put "3.0 s" on a token ending at 2944 ms).

One caveat stated plainly: BatchNorm uses batch statistics during **training**, which pool over time
and across the batch. At inference it uses fixed running statistics, so deployment is causal;
training is not strictly online. The BiTE baseline is non-causal at *inference* by construction.

## Quick start

```bash
pip install -r requirements.txt
export READER_DATA=/path/to/prepared/data      # see docs/DATA.md
export PYTHONPATH=$PWD
python scripts/fetch_baseline.py               # BiTE, for the baseline arms and the LOSO gate
pytest tests/ -q                               # 144 gates

python reader/train.py --model reader --dataset 2b --subject 4 --seed 2025 --epochs 600 \
       --out runs/demo/reader/2b_S4_seed2025
```

Full reproductions (SLURM; edit the partition in `scripts/slurm/pack.sbatch`):

```bash
./scripts/reproduce_within.sh     #   378 runs: READER, Compact, BiTE x 42 subjects x 3 seeds
./scripts/reproduce_zoo.sh        # 1,230 runs: BiTE's 10 released baselines x 42 subjects x 3 seeds
./scripts/reproduce_loso.sh       #   252 runs: cross-subject, 3 arms x 28 subjects x 3 seeds, no HGD row
./scripts/reproduce_anytime.sh    # prefix-supervised READER + Compact, specialist banks on 2a / 2b / SD-SSVEP
./scripts/reproduce_controls.sh   #   168 runs: FF-Control + Compact-Mean x 28 subjects x 3 seeds
./scripts/rebuild_results.sh      # every file in results/ from runs/, CPU only
python scripts/efficiency.py      # parameters + CPU latency (run on an idle CPU)
```

## Reproducing from scratch

What ships here is **code**, not cached results. `results/` holds our scored outputs so the numbers
can be checked without a GPU; re-running `scripts/score.py` **overwrites them** from your own runs,
so a regenerated table is your table, not ours.

Verified on a clean clone: `pytest` (the gates at the time: 77), a single `reader/train.py` run, the
manifest -> `sbatch` -> `scripts/score.py` pipeline, and the BiTE baseline after
`scripts/fetch_baseline.py`.

Requirements that are easy to miss:

- **`READER_DATA` must point at a prepared corpus tree.** The EEG is not shipped (~2.7 GB, and
  each corpus has its own terms), but the raw-download -> `.npz` conversion now is:
  `scripts/prepare_data.py` drives **BiTE's own** `preprocess_*` functions with the settings from
  their `config.yaml`, so both models see the authors' preprocessing rather than our reading of
  their paper. It needs `scripts/fetch_baseline.py` first and `pip install -r
  requirements-prepare.txt` (mne, scipy; braindecode for HGD only).

  ```bash
  python scripts/prepare_data.py --raw /path/to/downloads --out data --corpus 2a 2b sdssvep hgd
  python scripts/prepare_data.py --verify --out data     # against results/data_checksums.json
  ```

  `docs/DATA.md` gives the layout, the per-corpus preprocessing and where each raw download comes
  from. Starting from an already-prepared tree, everything below runs unchanged.
- **The clone must live on a filesystem the compute nodes can see.** A clone under node-local
  `/tmp` fails immediately with no log, because SLURM cannot reach the working directory.
- `scripts/fetch_baseline.py` needs network access, and is required for the `bite` arm and for the
  LOSO alignment gate (which skips without it).

Measured cost of the full within-subject cohort (378 runs, 600 epochs each, H100/H200):

| arm | 2a | 2b | HGD | SD-SSVEP |
|---|---:|---:|---:|---:|
| BiTE | 9.3 h | 5.4 h | 22.0 h | 2.2 h |
| compact | 5.9 h | 3.6 h | 25.1 h | 1.9 h |
| READER | 9.1 h | 6.1 h | 42.9 h | 3.3 h |

**~137 GPU-hours total**, about 12 wall-clock hours on 12 GPUs packed 4 runs per GPU. READER's HGD
cost is the quadratic prefix recomputation: the anytime curve is O(T^2) in the token count. Add
~20 GPU-hours for the cross-subject cohort, and **42.9** measured for everything
`reproduce_anytime.sh` launches: the specialist banks (2a 24.2 h over 162 runs, 2b 4.5 h over 81)
plus the prefix-supervised reader (2a 10.8 h, 2b 3.4 h). The two controls took 20.3 summed run-hours
(FF-Control 2a 6.7 / 2b 3.0 / SD 1.0; Compact-Mean 6.4 / 2.4 / 0.7) with 8 runs sharing each H200.

**Hardware reproducibility.** A run repeated on an H100 80GB and on an H200 NVL is bit-identical (45/45
READER runs). MIG-partitioned slices (H100 NVL MIG 3g.47gb) are not: 11/12 READER reruns differed from the
full-GPU run, by up to 5.56 pp on one 2a run. Keep paired arms off MIG partitions.

## Layout

```
reader/          data roles, model, trainer, diagnostics, anytime scoring
  data.py        SPECS, official within-subject roles, BiTE-matched LOSO roles + alignment
  models/
    compact.py   the forward-only parent decoder
    reader.py    READER: prefix-bidirectional reading  <- the contribution
    baseline.py  BiTE adapter + its exact STFT input (fetched, not vendored)
  train.py       one run: Adam 2e-3/2e-3, cosine, LS .1, batch 64, clip 5, 600 epochs, final epoch
  diagnostics.py per-layer/per-epoch instrument: weight & gradient norms, activation
                 distributions, effective rank, train->test probes, calibration
  anytime.py     one model vs the per-deadline specialist bank (strongest bank is the comparison of record)
  exact_duration.py  same checkpoint on truncated input + gate interventions + causality on trained weights
  subject_stats.py   subject-level paired CIs, exact sign-flip Wilcoxon, Holm over declared families
  ablation.py    prefix supervision vs endpoint-only loss, with a matched-arms guard
  gate_intervention.py  pin the fusion gate at 0 / 1 / 0.5 on trained weights
  zoo_report.py  BiTE's baselines re-run here: per-subject tables, READER minus each model
  models/zoo.py  BiTE's 10 released baselines through BiTE's own get_model, code unchanged
scripts/
  reproduce_*.sh         SLURM launchers for every cohort in the paper
  rebuild_results.sh     all of results/ from runs/, in dependency order
  link_record_runs.py    runs of record from the development harness (MIG -> full-GPU re-run rule)
  export_runs.py         results/raw/*.csv: every run, flat, with kappa
  paper_tables.py        MAIN_TABLE, CROSS_SUBJECT, DEVELOPMENT_DISCLOSURE, REPRODUCIBILITY
  ablation_table.py      ABLATION: architecture, inference interventions, loss, matched loss
  efficiency.py          EFFICIENCY: parameters and single-thread CPU latency per decision
  make_figures.py        results/figures/*.pdf, *.png
analysis/        model-free input probes (not reported in the paper)
results/         every paper table and figure, checkable without a GPU (index: results/README.md)
docs/            DATA.md, CONTROLS.md, REVIEWER_QUESTIONS.md
tests/           causality (activated + leaky mutant), controls, compact parity, LOSO roles, zoo, results pipeline
```

## Model-free probes (development evidence, not reported in the paper)

`analysis/` reproduces the input-level measurements that rule out three families before any
training, each with the same ridge protocol (fit on the training role, score the test role):

- `field_information_probe.py` — inter-electrode differential-field features carry **no** class
  information beyond the channels: the edge residual after regressing channels out is at chance on
  2a and HGD, and adding edges makes those probes *worse*.
- `spectral_headroom_probe.py` — time-resolved spectral structure adds nothing on 2b
  (band power .7609 vs windowed spectrum .6822), so the 2b deficit is not an input-information gap.
- `reference_admittance_oracle.py` — the best amount of common-mode removal is neither CAR nor raw:
  SD-SSVEP's optimum is 0.25, beating CAR by +1.66.

## Honest limitations

- 2a (−0.63) and 2b (−1.96) remain below the published bars; only HGD and SD-SSVEP clear them. Against BiTE
  re-run with 3 seeds READER is at parity on motor imagery (2b leans to BiTE, −0.92 [−1.88, +0.08]).
- DeepConvNet re-run on SD-SSVEP (96.50) is level with READER (96.17): −0.33 [−1.56, +0.67].
- The advantage of one READER over retrained per-deadline banks holds on 2a only: 2b is a flat −1.5 (the endpoint
  gap) and on SD-SSVEP retrained Compact specialists win at 0.5 s and 0.75 s. It needs prefix supervision, which costs
  −0.67 (2b) and −1.83 (SD-SSVEP) at the endpoint; with 9 subjects the 2a 1 s and 2 s intervals exclude zero but do
  not survive Holm correction.
- The fusion gate does not learn (it stays at 0.5 and pinning it is free); the fusion is a fixed average.
- Cross-subject is parity rather than a win outside SD-SSVEP; seeds 2026-2027 are still training.
- READER is 1.3–2.2× BiTE's parameter count, so no efficiency claim is made.
- **Development was test-informed.** Architecture screens scored the official test session, and READER was
  chosen among eight arms that way. What bounds the bias: on the 34 subjects the screen never used, READER
  minus Compact is +1.43 corpus-balanced, against +1.92 on the screen subjects. No claim is made that design
  choices were independent of test data.
- The prefix gain over Compact is partly start-anchoring (see above); FF-Control does not separate direction
  from anchoring.
- Every model loses accuracy from its own test-curve peak to the reported final epoch (READER 1.89, BiTE 2.06,
  zoo baselines 2.20–4.54 pp; `results/REPRODUCIBILITY.md`). This is BiTE's recipe (600 epochs, no validation
  set, final-epoch reporting), matched deliberately; it does not favour READER over BiTE, and it costs the zoo
  baselines somewhat more.
- Zoo baselines are trained with this repository's recipe (float32, clip 5), not BiTE's trainer (AMP, no clip);
  their seed-2025 re-runs track BiTE's published table within about 1 pp on average.
- 147 runs of comparisons of record first ran on MIG-partitioned GPU slices, which do not reproduce full-GPU runs
  bit-for-bit; they are being re-run on H200 and replace the originals (`results/REPRODUCIBILITY.md`).
- HGD has endpoint and gate-intervention results only: no FF-Control, Compact-Mean, exact-duration or anytime runs.
- CPU latency in `results/EFFICIENCY.md` is provisional: shared-node timings of the same configuration varied up to 3×.
- The BiTE repository ships no license. It is fetched, never redistributed here.
