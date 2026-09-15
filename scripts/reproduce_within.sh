#!/usr/bin/env bash
# Within-subject cohort: 3 arms x 42 subjects x 3 seeds = 378 runs.
# ~8 h on 12 H100s packed 4-per-GPU. Set READER_DATA first (see docs/DATA.md).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_manifest.py \
  --arm reader:reader --arm compact:compact --arm bite:bite \
  --study cohort --cells all --seeds 2025,2026,2027 \
  --output runs/manifests/cohort.txt
sbatch --array=0-11 scripts/slurm/pack.sbatch runs/manifests/cohort.txt 32 4
