# Figures

Drawn by `scripts/make_figures.py` from committed files only. Every figure's numbers are in a table.

| figure | shows | numbers |
|---|---|---|
| `fig_anytime` | Accuracy at each decision deadline. Blue lines: ONE READER model read at every token boundary (solid: trained with prefix supervision; dashed: endpoint loss only). Markers: separately retrained specialists, one per deadline (BiTE aqua triangles, Compact orange squares), mean ± 1 SE across subjects; the full-trial marker is the within-subject cohort run. | `results/anytime_2a.md`, `anytime_2b.md`, `anytime_sdssvep.md` |
| `fig_exact_duration` | The SAME endpoint-trained checkpoints given only the first part of every test trial (exact truncated input, partial last pooling window kept). | `results/subject_stats/SUBJECT_STATS.md`, `results/exact_duration.json` |
| `fig_reader_vs_zoo` | READER minus every baseline re-run with 3 seeds under the same protocol: subject-level mean (dot) and 95% paired bootstrap CI (bar). Right of zero favours READER. "pending" = that baseline's runs are not complete. | `results/MAIN_TABLE.md`, `results/model_zoo/MODEL_ZOO.md` |
| `fig_landscape` | Accuracy against parameter count, one panel per corpus. Hollow circles are BiTE's ten released baselines re-run here; READER, BiTE and Compact are filled and named. Up is more accurate, left is smaller. | `results/MAIN_TABLE.md`, `results/EFFICIENCY.md` |
| `fig_subject_dumbbell` | One row per subject: BiTE's accuracy and READER's, joined by a line coloured by which of the two won. Shows who the mean is made of. | `results/subject_stats/SUBJECT_STATS.md` |
| `fig_subject_heatmap` | Every model against every subject. Rows ordered by mean accuracy over corpora; dark columns are the subjects every model finds hard. | `results/model_zoo/MODEL_ZOO.md`, `results/raw/runs.csv` |
| `fig_confusion` | READER's row-normalised confusion matrix per corpus, pooled over subjects and seeds. Class names are BiTE's own (2a/2b from the official `classlabel`, HGD from its event mapping); SD-SSVEP's twelve stimuli are numbered. | `results/ERROR_ANALYSIS.md` |
| `fig_complementarity` | Of every test trial: both models right, only READER, only BiTE, neither. The `oracle` figure at the right is the ceiling a perfect selector between the two would reach. | `results/ERROR_ANALYSIS.md` |
| `fig_calibration` | Reliability: accuracy against confidence, 15 equal-width bins. Above the diagonal means under-confident, which every arm is here (label smoothing 0.1). | `results/ERROR_ANALYSIS.md` |

Colour follows the model, never its rank: READER blue, Compact orange, BiTE aqua, FF-Control yellow,
Compact-Mean magenta, and marker shape repeats the identity so the figures survive greyscale printing
and colour-vision deficiency. Heatmaps use a single-hue blue ramp (magnitude, not identity).
