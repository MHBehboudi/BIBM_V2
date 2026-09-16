#!/usr/bin/env bash
# Capacity and readout controls: FF-Control + Compact-Mean x 28 subjects (2a, 2b, SD-SSVEP) x 3 seeds = 168 runs.
# 20.3 summed run-hours with 8 runs sharing each H200. Set READER_DATA first (see docs/DATA.md).
# Keep paired arms OFF MIG-partitioned GPUs: MIG slices do not reproduce full-GPU runs bit-for-bit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/make_manifest.py \
  --arm ff_control:ff_control --arm compact_mean:compact_mean \
  --study controls --cells all_no_hgd --seeds 2025,2026,2027 \
  --output runs/manifests/controls.txt
sbatch --array=0-20 scripts/slurm/pack.sbatch runs/manifests/controls.txt 8 8

cat <<'MSG'
when complete (and after scripts/reproduce_within.sh + reproduce_anytime.sh for the cohort and the bank):
  for ds in 2a 2b sdssvep; do
    for arm in reader compact bite; do python reader/exact_duration.py --dataset $ds --arm $arm --runs runs/cohort/$arm; done
    for arm in ff_control compact_mean; do python reader/exact_duration.py --dataset $ds --arm $arm --runs runs/controls/$arm; done
  done
  python reader/subject_stats.py --cohort runs/cohort --controls runs/controls --bank runs/bank
exact_duration.py is CPU-only (a few seconds per checkpoint; READER is slowest because of the gate interventions).
MSG
