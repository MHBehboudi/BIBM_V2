#!/usr/bin/env bash
# Cross-subject (leave-one-subject-out). BiTE publishes no HGD LOSO row, so HGD is omitted.
# The bite arm on 2a precomputes an 8.1 GB STFT over the pooled training role: give it its own GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
# SLURM cannot create its own log directory; make it before submitting.
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/make_manifest.py \
  --arm reader:reader:--protocol,loso --arm compact:compact:--protocol,loso \
  --arm bite:bite:--protocol,loso \
  --study loso --cells all_no_hgd --seeds 2025 \
  --output runs/manifests/loso.txt
sbatch --array=0-11 scripts/slurm/pack.sbatch runs/manifests/loso.txt 9 2
