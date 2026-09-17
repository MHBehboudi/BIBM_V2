# Figures

Drawn by `scripts/make_figures.py` from committed files only. Every figure's numbers are in a table.

| figure | shows | numbers |
|---|---|---|
| `fig_anytime` | Accuracy at each decision deadline. Blue lines: ONE READER model read at every token boundary (solid: trained with prefix supervision; dashed: endpoint loss only). Markers: separately retrained specialists, one per deadline (BiTE aqua triangles, Compact orange squares), mean ± 1 SE across subjects; the full-trial marker is the within-subject cohort run. | `results/anytime_2a.md`, `anytime_2b.md`, `anytime_sdssvep.md` |
| `fig_exact_duration` | The SAME endpoint-trained checkpoints given only the first part of every test trial (exact truncated input, partial last pooling window kept). | `results/subject_stats/SUBJECT_STATS.md`, `results/exact_duration.json` |
| `fig_reader_vs_zoo` | READER minus every baseline re-run with 3 seeds under the same protocol: subject-level mean (dot) and 95% paired bootstrap CI (bar). Right of zero favours READER. "pending" = that baseline's runs are not complete. | `results/MAIN_TABLE.md`, `results/model_zoo/MODEL_ZOO.md` |
