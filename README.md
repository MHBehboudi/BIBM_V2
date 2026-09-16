# Prefix-bidirectional reading for anytime EEG decoding

One model that emits a legal decision every 128 ms and, trained with prefix supervision, beats a
bank of separately retrained per-deadline specialists at every deadline on 2a — by +2.57 at 1 s,
where a bank is most expensive to maintain and hardest to beat.

Bidirectional context in EEG decoders is bought at the cost of deployability. BiTE's BiTCN obtains
backward context by reversing the **complete trial** — its backward branch's last step corresponds
to the trial's first moment, and its STFT front end uses `center=True`, reading 128 ms into the
future. No output exists until the trial ends, so a deployed system must train and hold a separate
model per decision deadline.

**READER** applies the backward branch to the **observed prefix** instead. At deadline *t* a second
causal TCN reads tokens *t−1 … 0* and is fused with the forward reading by a convex per-feature
gate. Reversed over the trial is non-causal; reversed over the prefix is not.

## Results

Within-subject, 42 subjects × 3 seeds = 126 runs per arm, official roles, fixed final epoch, no
model selection (`results/within_subject.json`):

| model | params | 2a | 2b | HGD | SD-SSVEP | corpus-bal |
|---|---:|---:|---:|---:|---:|---:|
| EEGNet | — | 71.14 | 82.70 | 93.97 | 64.83 | 78.16 |
| DeepConvNet | — | 72.38 | 85.51 | 93.08 | **95.50** | 86.62 |
| ATCNet | — | 81.02 | 84.25 | 95.89 | 84.50 | 86.42 |
| BiTE (published) | 14.5K | **85.34** | **88.37** | 95.93 | 94.16 | 90.95 |
| BiTE (this harness) | 14.5–17.9K | 84.08 | 87.33 | 95.57 | 94.22 | 90.30 |
| compact (forward-only parent) | 17.8K | 82.33 | 85.32 | 95.69 | 94.11 | 89.36 |
| **READER** | 19.0–32.4K | 84.71 | 86.41 | **96.30** | **96.17** | **90.90** |

Two of four published bars cleared: HGD **+0.37**, SD-SSVEP **+0.67** over DeepConvNet (+2.01 over
BiTE). 2a is **−0.63** and 2b **−1.96** short. Paired against BiTE in this harness (126 pairs):
**+0.59** corpus-balanced (2a +0.63, 2b −0.92, HGD +0.73, SD +1.94).

Cross-subject, leave-one-subject-out, seed 2025 (`results/cross_subject.json`). BiTE publishes no
HGD LOSO row:

| model | 2a | 2b | SD-SSVEP |
|---|---:|---:|---:|
| BiTE (published) | **64.56** | **76.44** | 79.72 |
| BiTE (this harness) | 61.11 | 76.08 | 79.44 |
| compact | 59.66 | **76.65** | 80.11 |
| **READER** | 60.36 | 75.83 | **81.22** |

Parity, with one win: SD-SSVEP +1.78 paired, 7/10. Our BiTE reproduction is within 0.4 of the
published 2b/SD bars but 3.45 under on 2a, so the 2a shortfall is in the reproduction, not the arm.
**One seed** — the 2a/2b signs are not established.

Anytime on 2a: **one** READER read at each deadline vs a **bank of separately retrained**
specialists, paired by subject and seed, 9 subjects × 3 seeds (`results/anytime_2a.md`):

| deadline | READER (1 model) | bite bank | compact bank | paired vs bite |
|---|---:|---:|---:|---:|
| 1.0 s | 76.84 | 74.27 | 74.83 | **+2.57** |
| 2.0 s | 83.37 | 81.49 | 81.57 | **+1.88** |
| 3.0 s | 84.10 | 82.93 | 83.35 | **+1.17** |
| 4.0 s | 84.84 | 84.08 | 82.33 | +0.76 |

### Scope of the anytime claim

Three limits, all of them load-bearing:

1. **It requires prefix supervision, which is a different arm.** The table above is the reader
   trained with `--prefix-weight 0.3` (deep supervision over deadlines) — a *loss* change, not the
   architecture. The endpoint-supervised reader from the within-subject table above *loses* **−3.99
   at 1 s** against the same bank and is then within ±1 point of it (+0.67 / −0.81 / +0.63 at
   2/3/4 s) — it does not carry the anytime claim on its own
   (`results/anytime_2a_endpoint_supervised.md`). Prefix supervision is free on 2a
   (+0.13 endpoint) and costs −1.83 on SD-SSVEP, so it is not a global default and the
   within-subject table is not built on it. Which arm produced an anytime number is therefore part
   of that number; `reader/anytime.py --reader` selects it and the generated tables name it.
2. **It is a 2a claim.** On 2b the same comparison is **−1.50 / −1.62 / −1.44 / −1.59** at
   1/2/3/4 s (`results/anytime_2b.md`) — a flat offset, statistically indistinguishable across
   deadlines and equal to BiTE's standing 2b *endpoint* advantage. That is an endpoint deficit
   showing up at every deadline, not an anytime effect in either direction.
3. **No bank exists on HGD or SD-SSVEP**, so no comparison is available there.

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

## Is it actually causal?

Yes, and it is tested rather than asserted (`tests/test_deployment_causality.py`):

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
pytest tests/ -q                               # 36 gates

python reader/train.py --model reader --dataset 2b --subject 4 --seed 2025 --epochs 600 \
       --out runs/demo/reader/2b_S4_seed2025
```

Full reproductions (SLURM; edit the partition in `scripts/slurm/pack.sbatch`):

```bash
./scripts/reproduce_within.sh     # 378 runs: 3 arms x 42 subjects x 3 seeds
./scripts/reproduce_loso.sh       #  84 runs: cross-subject, no HGD row
./scripts/reproduce_anytime.sh    # the per-deadline specialist bank on 2a
python scripts/score.py --runs runs/cohort --name within_subject --reference bite
```

## Reproducing from scratch

What ships here is **code**, not cached results. `results/` holds our scored outputs so the numbers
can be checked without a GPU; re-running `scripts/score.py` **overwrites them** from your own runs,
so a regenerated table is your table, not ours.

Verified on a clean clone: `pytest` (36 gates), a single `reader/train.py` run, the
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
plus the prefix-supervised reader (2a 10.8 h, 2b 3.4 h).

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
  anytime.py     one model vs the per-deadline specialist bank
  ablation.py    prefix supervision vs endpoint-only loss, with a matched-arms guard
  gate_intervention.py  pin the fusion gate at 0 / 1 / 0.5 on trained weights
analysis/        model-free input probes behind the paper's negative results
results/         scored tables + probe outputs, checkable without a GPU
tests/           causality, compact parity, LOSO role gates
```

## Model-free probes (the paper's negative results)

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

- 2a (−0.63) and 2b (−1.96) remain below the published bars; only HGD and SD-SSVEP clear them.
- Cross-subject is one seed, and is parity rather than a win outside SD-SSVEP.
- READER is 1.3–2.2× BiTE's parameter count, so no efficiency claim is made.
- Every arm, ours and BiTE's, loses 1.5–3.2 pp from its own test-curve peak to the reported final
  epoch. This is a property of BiTE's recipe (600 epochs, no validation set, final-epoch reporting)
  which we match deliberately; it is **not** differential between arms (reader−BiTE decay is
  −0.14/+0.04/+0.03/−0.67 pp, all inside their standard errors).
- The BiTE repository ships no license. It is fetched, never redistributed here.
