#!/usr/bin/env bash
# Cross-subject (leave-one-subject-out). BiTE publishes no HGD LOSO row, so HGD is omitted.
# The bite arm on 2a precomputes an 8.1 GB STFT over the pooled training role: give it its own GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
# SLURM cannot create its own log directory; make it before submitting.
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# 3 arms x 28 subjects x 3 seeds = 252 runs, ~120 GPU-hours. BiTE runs with --clip 0 (its release does not clip).
python scripts/make_manifest.py \
  --arm reader:reader:--protocol,loso --arm compact:compact:--protocol,loso \
  --arm bite:bite:--protocol,loso,--clip,0 \
  --study loso --cells all_no_hgd --seeds 2025,2026,2027 \
  --output runs/manifests/loso.txt
sbatch --array=0-20 scripts/slurm/pack.sbatch runs/manifests/loso.txt 12 3

cat <<'MSG'
when complete:
  python scripts/export_runs.py && python scripts/paper_tables.py     # -> results/CROSS_SUBJECT.md
MSG
