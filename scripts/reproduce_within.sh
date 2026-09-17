#!/usr/bin/env bash
# Within-subject cohort: 3 arms x 42 subjects x 3 seeds = 378 runs.
# BiTE is trained with --clip 0 because its release does not clip gradients; our arms use the recipe's clip 5.
# ~8 h on 12 H100s packed 4-per-GPU. Set READER_DATA first (see docs/DATA.md).
set -euo pipefail
cd "$(dirname "$0")/.."
# SLURM cannot create its own log directory; make it before submitting.
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/make_manifest.py \
  --arm reader:reader --arm compact:compact --arm bite:bite:--clip,0 \
  --study cohort --cells all --seeds 2025,2026,2027 \
  --output runs/manifests/cohort.txt
sbatch --array=0-11 scripts/slurm/pack.sbatch runs/manifests/cohort.txt 32 4

cat <<'MSG'
when complete:
  python scripts/score.py --runs runs/cohort --name within_subject --reference bite
  python reader/gate_intervention.py --arm runs/cohort/reader
the gate interventions are recorded during training, so scoring them needs no GPU.
MSG
