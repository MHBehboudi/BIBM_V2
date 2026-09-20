#!/usr/bin/env bash
# Hyper-parameter sensitivity of the reverse reader: 1008 runs (see docs/SENSITIVITY_ANALYSIS.md).
#   Q1  prefix-supervision weight  lambda in {0.1, 1.0}   at the corpus default pooling factor
#   Q2  token resolution           p ~ 16 / 32 / 64 tokens at lambda in {0, 0.3}
# lambda 0 and lambda 0.3 at the default p already exist as runs/cohort and runs/anytime and are NOT re-run.
# Both arms are trained at every setting: the quantity of interest is READER minus its forward-only control.
# ~190 summed run-hours with 12 runs sharing each H200. Set READER_DATA first (see docs/DATA.md).
# Keep paired arms OFF MIG-partitioned GPUs: MIG slices do not reproduce full-GPU runs bit-for-bit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---------------------------------------------------------------- Q1: the prefix-supervision weight
# The corpus default pooling factor, so only the loss changes. READER returns its per-deadline
# sequence on every forward pass, so --prefix-weight alone is enough here.
for lam in 0.1 1.0; do
  for ds in 2a 2b; do
    python scripts/make_manifest.py --study sensitivity --cells "$ds" --seeds 2025,2026,2027 \
      --arm "reader_p32_ps${lam}:reader:--prefix-weight,${lam}" \
      --arm "compact_p32_ps${lam}:compact:--prefix-weight,${lam}" \
      --output "runs/manifests/sens_lambda_${ds}_ps${lam}.txt"
  done
  python scripts/make_manifest.py --study sensitivity --cells sdssvep --seeds 2025,2026,2027 \
    --arm "reader_p4_ps${lam}:reader:--prefix-weight,${lam}" \
    --arm "compact_p4_ps${lam}:compact:--prefix-weight,${lam}" \
    --output "runs/manifests/sens_lambda_sdssvep_ps${lam}.txt"
done

# ---------------------------------------------------------------- Q2: the token resolution
# p is corpus-specific, so the manifests are written per corpus. Token counts: 2a/2b 64->16, 16->63;
# SD-SSVEP 16->16, 8->32. All inside the 71-token receptive field.
for lam in 0 0.3; do
  # the cell directory name carries the setting, so the collector and the scorer can find it
  weight=""
  [ "$lam" != "0" ] && weight=",--prefix-weight,${lam}"
  for p in 64 16; do
    for ds in 2a 2b; do
      python scripts/make_manifest.py --study sensitivity --cells "$ds" --seeds 2025,2026,2027 \
        --arm "reader_p${p}_ps${lam}:reader:--option,pool=${p}${weight}" \
        --arm "compact_p${p}_ps${lam}:compact:--option,pool=${p}${weight}" \
        --output "runs/manifests/sens_pool_${ds}_p${p}_ps${lam}.txt"
    done
  done
  for p in 16 8; do
    python scripts/make_manifest.py --study sensitivity --cells sdssvep --seeds 2025,2026,2027 \
      --arm "reader_p${p}_ps${lam}:reader:--option,pool=${p}${weight}" \
      --arm "compact_p${p}_ps${lam}:compact:--option,pool=${p}${weight}" \
      --output "runs/manifests/sens_pool_sdssvep_p${p}_ps${lam}.txt"
  done
done

# ---------------------------------------------------------------- submit
# 12 runs per GPU: measured 1.6x per-run slowdown for 12x concurrency (7.5x effective). The 63-token
# READER cells are the heavy ones; give them a full-memory GPU rather than a MIG slice.
for m in runs/manifests/sens_*.txt; do
  n=$(wc -l < "$m"); tasks=$(( (n + 55) / 56 ))
  sbatch --array=0-$((tasks - 1)) scripts/slurm/pack.sbatch "$m" 56 12
done

cat <<'MSG'
when the runs are complete:
  sbatch scripts/slurm/collect_sensitivity.sbatch        # exact-duration records, 32 CPU cores
  python reader/sensitivity.py --plan results/sensitivity_plan.json
  python reader/sensitivity.py --plan results/sensitivity_plan.json --full-range \
         --out results/SENSITIVITY_FULL_RANGE.md

sensitivity_collect.py takes an exclusive lock: a second copy refuses to start rather than racing the
first on the same records. Do not run it on a login node -- READER recomputes every prefix state on
each forward pass, so one 64-token checkpoint is O(T^2) of CPU work.
MSG
