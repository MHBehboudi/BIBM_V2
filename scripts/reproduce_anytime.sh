#!/usr/bin/env bash
# The anytime comparison: ONE reader read at each deadline against a BANK of separately
# retrained per-deadline specialists. Deadline models see a truncated trial end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
: > runs/manifests/bank.txt
for d in 1 2 3; do
  for m in bite compact; do
    python scripts/make_manifest.py --arm "${m}_${d}s:${m}:--window-seconds,${d}" \
      --study bank --cells 2a --seeds 2025,2026,2027 --output /tmp/bank_part.txt
    cat /tmp/bank_part.txt >> runs/manifests/bank.txt
  done
done
sbatch --array=0-11 scripts/slurm/pack.sbatch runs/manifests/bank.txt 14 4
echo "when complete: python reader/anytime.py --cohort runs/cohort --bank runs/bank --dataset 2a"
