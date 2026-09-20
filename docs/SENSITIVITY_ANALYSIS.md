# Sensitivity of the reverse reader to its two declared hyper-parameters

The paper declares two settings without evidence that they matter: the prefix-supervision weight
`--prefix-weight 0.3` and the pooling factor `p` (32 on BCICIV-2A/2B, 4 on SD-SSVEP). A reviewer can
reasonably ask whether the reverse reader's advantage is an artefact of either choice — in
particular whether `p` was chosen because it flattered READER's early accuracy. It was not, and this
document is the evidence and the recipe.

**Design rule.** One factor moves at a time around the declared operating point, and at every
setting **both** arms are trained: READER and its forward-only control (`compact`), same seeds, same
subjects, same recipe, shared initialisation. The reported quantity is the paired difference, never
either arm's level — a level that moves with λ or `p` says nothing about the mechanism.

| sweep | varied | held | cells that already existed |
|---|---|---|---|
| Q1 prefix-supervision weight | λ ∈ {0, 0.1, 0.3, 1.0} | `p` at the corpus default | λ = 0 (`runs/cohort`), λ = 0.3 (`runs/anytime`) |
| Q2 token resolution | `p` ≈ 16 / 32 / 64 tokens | λ ∈ {0, 0.3} | the default `p` at both λ |

1008 new runs (336 + 336 + 336), 3 seeds × 9/9/10 subjects, 600 epochs, fixed final epoch, official
session roles. The full λ × `p` grid is deliberately **not** run: neither question needs it and it
would quadruple the cost.

## Why these values

`p` is chosen to give the same three token counts on every corpus, which is what makes the sweep
explainable in one sentence instead of "we tried some pooling factors":

| corpus | T | `p` = | tokens `⌈T/p⌉` | token duration |
|---|---:|---|---:|---|
| 2a / 2b | 1000 | 64 / 32 / 16 | 16 / 32 / 63 | 256 / 128 / 64 ms |
| SD-SSVEP | 256 | 16 / 8 / 4 | 16 / 32 / 64 | 62.5 / 31.25 / 15.6 ms |

Three facts make these the right three, and all three are checked in code before launch
(`tests/test_sensitivity.py`, and the pre-flight in the run card):

- **Every condition stays inside the reader's receptive field.** R = 1 + 2(6−1)(1+2+4) = 71 tokens,
  and the largest condition is 64. No condition is limited by history the reader cannot reach, so a
  difference between conditions is about resolution, not truncation.
- **`p` = 8 on SD-SSVEP is BiTE's own released pooling factor** for that corpus, so one point of the
  sweep is a setting an independent group chose.
- **`p` = 16 on SD-SSVEP aliases the stimulus.** The token rate is 16 Hz, below twice the 9.25–14.75 Hz
  carriers, so accuracy is expected to fall there. That makes the coarsest SSVEP condition a
  prediction the sweep can confirm rather than an arbitrary grid point.

λ ∈ {0, 0.1, 0.3, 1.0} spans endpoint-only training, weak supervision, the declared setting and
strong supervision. λ = 0 and λ = 0.3 are the paper's existing arms, so only 0.1 and 1.0 are new.

## The objective the sweep moves

With `K` token boundaries per trial and `t_K` the endpoint,

```
L = L_end + λ * (1/(K-1)) * sum_{i=1..K-1} L_CE(t_i)
```

implemented in `reader/train.py::task_loss` as a single cross-entropy over all non-endpoint token
decisions. Because the sum is **averaged** over the `K−1` early boundaries, λ is the total weight of
early supervision relative to the endpoint and does not silently change when `p` changes the token
count — which is what makes Q1 and Q2 separable.

## What is measured

`nAUC`, the paper's normalised accuracy–duration area, computed by `reader/sensitivity.py` from the
exact-duration records: the trained checkpoint is given only the first *n* samples of each test trial
and its endpoint prediction is scored, on a grid **in samples** that does not depend on `p`. This is
the only summary that is comparable across Q2's arms; a token-indexed curve would compare different
wall-clock durations across pooling factors.

Two integration ranges are reported:

- `nauc` — the manuscript's range: 1.0–4.0 s on 2a/2b, 0.25–1.0 s on SD-SSVEP.
- `nauc_full` — the whole evaluated grid: 0.5–4.0 s and 0.125–1.0 s.

**Note for the manuscript.** The published values (+9.31 / +1.13 / +12.30 for REACT+PS − no-reverse+PS)
are the *first* of these, but the Anytime Evaluation subsection describes the grid as starting at
0.5 s / 0.125 s. Over the full grid the same comparison is **+10.88 / +1.77 / +13.15**. The text and
the numbers must be made to agree; both ranges are defensible and the full range is the larger.

Endpoint accuracy is reported beside nAUC, from each run's own `summary.json`, so the λ = 0 /
default-`p` row reproduces the within-subject table exactly.

Uncertainty is the paper's: 95% percentile bootstrap over **subjects** as paired units (10,000
resamples, fixed seed), seeds averaged within subject first.

## Reading the result

The claim under test is that the paired difference stays positive across both sweeps. A setting
where the default `p` or λ = 0.3 is *not* the best cell is reported as such: this is a robustness
check, not a selection, and **nothing in the paper's main tables is re-selected from these runs**.

## Reproducing it

`scripts/reproduce_sensitivity.sh` writes the manifests and submits the training; then

```
python scripts/sensitivity_collect.py          # exact-duration records + the scorer's plan
python reader/sensitivity.py --plan results/sensitivity_plan.json
python reader/sensitivity.py --plan results/sensitivity_plan.json --full-range \
       --out results/SENSITIVITY_FULL_RANGE.md
```

`sensitivity_collect.py` holds an exclusive lock, so a second copy refuses to start; two collectors
evaluating the same cells waste the machine and can race on one record. Records are written through
a temp file and renamed, so a reader never sees a partial one. Run the collection on a compute node
(`scripts/slurm/collect_sensitivity.sbatch`) — READER's forward pass recomputes every prefix state,
so scoring a 64-token checkpoint is O(T²) on CPU and does not belong on a login node.

## Code this required, and one defect it exposed

- `reader/models/reader.py`: `pool` is an option instead of a corpus constant.
- `reader/exact_duration.py`: `--option` passthrough, `--label` so one arm directory can hold several
  pooling factors, the record now stores `pool` / `prefix_weight` / `options`, and the trained-weight
  causality check uses the **checkpoint's** pool rather than the corpus default.
- `reader/sensitivity.py`, `tests/test_sensitivity.py` (16 gates): the quadrature, the integration
  range, the paired bootstrap, and four guards that refuse a cell whose record came from a different
  run, whose pool disagrees with the plan, whose λ disagrees with the plan, or whose accuracy is read
  in the wrong units.
- **Defect found:** in the development harness the forward-only arm's builder *silently dropped*
  unrecognised options, so a misspelt `--option pool=16` would have trained the corpus default and
  reported it as a pooling result. It now raises. Nothing in the paper was affected — the defect was
  found before any sweep cell was trained — but it is the kind of thing that turns into a retraction.

## Pairing, verified rather than assumed

At every pooling factor, the 55 tensors READER shares with its forward-only control are bit-identical
at the same seed; only the reverse branch (3,072 weights) and its gate (64) are extra. The paper's
paired-initialisation claim therefore holds in the new conditions too, not just at the default `p`.
