# Prefix-bidirectional reading for anytime EEG decoding

One model that emits a legal decision every 128 ms, and matches or beats a bank of separately
retrained per-deadline specialists from 2 s on.

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
specialists (`results/anytime_2a.md`). With prefix supervision the single model beats the bank at
every deadline: 1 s **+2.57**, 2 s **+1.88**, 3 s **+1.17**.

### Scope of the anytime claim

It is a 2a claim. On 2b the same comparison is roughly **−1.5 at every deadline** — a flat offset
equal to BiTE's standing 2b endpoint advantage, not an anytime effect. No specialist bank exists on
HGD or SD-SSVEP, so no comparison is available there. Prefix supervision is a *loss* change
(deep supervision of the architecture); it is free on 2a (+0.13 endpoint) and costs −1.83 on
SD-SSVEP, so it is not a global default.

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
pytest tests/ -q                               # 16 gates

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
