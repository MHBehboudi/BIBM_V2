# Gate interventions: is the prefix-reversed branch load-bearing?

Inference-time interventions on already-trained weights; the gate is pinned and the model re-evaluated. Paired per cell against that cell's own learned-gate accuracy; se is of the paired delta, and `worse` counts cells the intervention hurts.

| corpus | n | learned | g=0 (forward only) | g=1 (prefix-reversed only) | g=0.5 (the init, unlearned) | learned gate |
|---|---:|---:|---:|---:|---:|---:|
| 2a | 27 | 84.71 | -0.60 (0.35, 19/27) | -2.12 (0.49, 21/27) | -0.06 (0.05, 8/27) | 0.4943 ± 0.0179 |
| 2b | 27 | 86.41 | -0.70 (0.23, 16/27) | -0.77 (0.29, 15/27) | +0.02 (0.07, 4/27) | 0.4969 ± 0.0156 |
| hgd | 42 | 96.30 | -0.44 (0.14, 22/42) | -0.30 (0.14, 17/42) | +0.00 (0.03, 2/42) | 0.5003 ± 0.0142 |
| sdssvep | 30 | 96.17 | -1.83 (0.57, 12/30) | -8.83 (1.61, 26/30) | +0.11 (0.11, 1/30) | 0.4484 ± 0.0378 |

Read the last two columns together. Removing either route costs accuracy on every corpus, so the fusion is load-bearing and neither reading subsumes the other. But the learned gate sits within a few thousandths of its 0.5 initialisation everywhere, and pinning it there costs nothing — so the gate's LEARNING contributes no accuracy, and the mechanism is honestly described as a fixed equal-weight convex average of the two readings.
