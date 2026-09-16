#!/usr/bin/env bash
# The anytime comparison: ONE reader read at each deadline against a BANK of separately
# retrained per-deadline specialists. Deadline models see a truncated trial end to end.
#
# Two arms are trained here, and the distinction matters:
#   reader_anytime  the reader with PREFIX SUPERVISION (--prefix-weight 0.3), which is the arm the
#                   headline anytime numbers come from. Prefix supervision is a LOSS change, free
#                   on 2a (+0.13 endpoint) and -1.83 on SD-SSVEP, so it is not a global default and
#                   is not the arm reported in the within-subject table.
#   {bite,compact}_{1,2,3}s  the specialist bank: one model per deadline, trained from scratch on
#                   trials truncated to that deadline. The 4 s row is the full-trial cohort run.
# Scoring the within-subject cohort's endpoint-supervised `reader` against this bank instead is a
# different (and weaker at 1 s) comparison -- pass --reader explicitly to pick the arm.
set -euo pipefail
cd "$(dirname "$0")/.."
# SLURM cannot create its own log directory; make it before submitting.
mkdir -p runs/manifests runs/slurm
: "${READER_DATA:?set READER_DATA to the prepared-data directory (see docs/DATA.md)}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- the prefix-supervised reader: 2a and 2b, 9 subjects x 3 seeds each -----------------------
python scripts/make_manifest.py --arm 'reader_anytime:reader:--prefix-weight,0.3' \
  --study anytime --cells '2a' --seeds 2025,2026,2027 --output runs/manifests/anytime.txt
python scripts/make_manifest.py --arm 'reader_anytime:reader:--prefix-weight,0.3' \
  --study anytime --cells '2b' --seeds 2025,2026,2027 --output runs/manifests/_part.txt
cat runs/manifests/_part.txt >> runs/manifests/anytime.txt

# --- the specialist bank ----------------------------------------------------------------------
# compact is banked on 2a only; on 2b the published BiTE endpoint is the reference that matters.
: > runs/manifests/bank.txt
for d in 1 2 3; do
  for m in bite compact; do
    python scripts/make_manifest.py --arm "${m}_${d}s:${m}:--window-seconds,${d}" \
      --study bank --cells 2a --seeds 2025,2026,2027 --output runs/manifests/_part.txt
    cat runs/manifests/_part.txt >> runs/manifests/bank.txt
  done
  python scripts/make_manifest.py --arm "bite_${d}s:bite:--window-seconds,${d}" \
    --study bank --cells 2b --seeds 2025,2026,2027 --output runs/manifests/_part.txt
  cat runs/manifests/_part.txt >> runs/manifests/bank.txt
done
rm -f runs/manifests/_part.txt

READER_LINES=$(wc -l < runs/manifests/anytime.txt)
BANK_LINES=$(wc -l < runs/manifests/bank.txt)
sbatch --array=0-11 scripts/slurm/pack.sbatch runs/manifests/anytime.txt $(( (READER_LINES + 11) / 12 )) 4
sbatch --array=0-11 scripts/slurm/pack.sbatch runs/manifests/bank.txt    $(( (BANK_LINES + 11) / 12 )) 4

cat <<'EOF'
when complete, regenerate both shipped tables:
  python reader/anytime.py --dataset 2a --cohort runs/cohort --bank runs/bank \
         --reader runs/anytime/reader_anytime --out results/anytime_2a.md
  python reader/anytime.py --dataset 2b --cohort runs/cohort --bank runs/bank \
         --reader runs/anytime/reader_anytime --out results/anytime_2b.md
the 4 s row is read from runs/cohort, so scripts/reproduce_within.sh must have finished first.
EOF
