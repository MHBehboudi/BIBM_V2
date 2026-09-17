#!/usr/bin/env bash
# BiTE's ten released baselines, re-run under this repository's protocol so the comparison is multi-seed and
# paired: 10 models x 42 subjects x 3 seeds = 1,260 runs (FACTNet skips SD-SSVEP: its released attention
# hard-codes 1000 samples, so 1,230 run). Models are built by BiTE's own get_model, code unchanged
# (reader/models/zoo.py); run scripts/fetch_baseline.py first. Recipe = this repository's trainer
# (Adam 2e-3, cosine, label smoothing .1, batch 64, float32, gradient clip 5, 600 epochs, final epoch).
# ~180 GPU-hours on H200s, 10 runs per GPU. Keep off MIG-partitioned GPUs (see results/REPRODUCIBILITY.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ARMS=()
for m in ATCNet DMSANet DeepConvNet EEGNeX EEGNet EEGTCNet EISATC FACTNet MBCNNEATCFNet ShallowConvNet; do
  ARMS+=(--arm "${m}:zoo_${m}:--no-diag")
done
python scripts/make_manifest.py "${ARMS[@]}" --study zoo --cells all --seeds 2025,2026,2027 \
  --output runs/manifests/zoo.txt
# FACTNet cannot run on SD-SSVEP as released
grep -v -- "--model zoo_FACTNet --dataset sdssvep" runs/manifests/zoo.txt > runs/manifests/_zoo.txt
mv runs/manifests/_zoo.txt runs/manifests/zoo.txt
LINES=$(wc -l < runs/manifests/zoo.txt)
sbatch --array=0-40 scripts/slurm/pack.sbatch runs/manifests/zoo.txt $(( (LINES + 40) / 41 )) 10

cat <<'MSG'
when complete (and after scripts/reproduce_within.sh for READER / Compact / BiTE):
  scripts/rebuild_results.sh     # -> results/MAIN_TABLE.md, results/model_zoo/MODEL_ZOO.md, figures
MSG
