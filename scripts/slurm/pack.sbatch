#!/usr/bin/env bash
# One array task = one GPU running PARALLEL runs from a manifest chunk.
#SBATCH --job-name=fusion
#SBATCH --partition=h200,h100,a30,a30-2.12gb   # site-specific: edit for your cluster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --error=runs/slurm/%x_%A_%a.err
set -uo pipefail
MANIFEST="${1:?manifest}"; PER_TASK="${2:?lines per task}"; PARALLEL="${3:-4}"
# Override with READER_PYTHON=/path/to/python and READER_DATA=/path/to/data.
PY="${READER_PYTHON:-python}"
export READER_DATA="${READER_DATA:?set READER_DATA to the prepared-data directory}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p runs/slurm
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK + 1 )); END=$(( START + PER_TASK - 1 ))
printf 'job=%s task=%s partition=%s node=%s lines=%s-%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_PARTITION" "$SLURMD_NODENAME" "$START" "$END"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${MEM:-0}" -lt 40000 ] && [ "$PARALLEL" -gt 3 ]; then PARALLEL=3; echo "small GPU: parallel -> $PARALLEL"; fi
sed -n "${START},${END}p" "$MANIFEST" | xargs -P "$PARALLEL" -I{} bash -c '
  set -- {}; OUT=""; prev=""
  for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done
  mkdir -p "$OUT"
  '"$PY"' reader/train.py "$@" > "$OUT/log.txt" 2>&1 && echo "done $OUT" || echo "FAILED $OUT"
'
