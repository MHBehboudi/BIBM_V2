#!/usr/bin/env bash
# Exact-duration evaluation of the sensitivity cells. CPU only, no GPU: it is inference on stored
# checkpoints. It runs HERE and not on the login node, which has 4 cores and is shared.
#SBATCH --job-name=senscollect
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/mxb190076/BIBM2026/prefix-reader-eeg
#SBATCH --output=/work/mxb190076/BIBM2026/prefix-reader-eeg/runs/collect_%j.out
#SBATCH --error=/work/mxb190076/BIBM2026/prefix-reader-eeg/runs/collect_%j.err
set -uo pipefail
PY=/work/mxb190076/conda_envs/sleepfm/bin/python
export READER_DATA=/work/mxb190076/BIBM2026/data OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$PY" scripts/sensitivity_collect.py --python "$PY" --workers 16
