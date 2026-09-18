#!/usr/bin/env bash
# Rebuild every file in results/ from the runs of record, in dependency order. CPU only.
#
#   runs/  (from scripts/reproduce_*.sh, or linked from the development harness by scripts/link_record_runs.py)
#     -> results/raw/*.csv                     every run, flat (the only input of the paper tables below)
#     -> results/MAIN_TABLE.md, CROSS_SUBJECT.md, DEVELOPMENT_DISCLOSURE.md, REPRODUCIBILITY.md
#     -> results/anytime_{2a,2b,sdssvep}.md    one model vs retrained per-deadline banks
#     -> results/exact_duration.json, subject_stats/, ABLATION.md, ablation_prefix_supervision.md
#     -> results/model_zoo/, gate_intervention.md, ERROR_ANALYSIS.md, figures/, dashboard.html
# EFFICIENCY.md is measured separately (scripts/efficiency.py): it times models, so run it on an idle CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="${READER_PYTHON:-python}"

if [ "${1:-}" = "--from-harness" ]; then
  "$PY" scripts/link_record_runs.py ${2:+--harness "$2"}
fi

# exact-duration evaluation of every checkpoint of record that lacks it (recomputes replaced runs)
for ds in 2a 2b sdssvep; do
  for arm in reader compact bite; do
    "$PY" reader/exact_duration.py --dataset $ds --arm $arm --runs runs/cohort/$arm
  done
  for arm in ff_control compact_mean; do
    [ -d runs/controls/$arm ] && "$PY" reader/exact_duration.py --dataset $ds --arm $arm --runs runs/controls/$arm
  done
  [ -d runs/anytime/reader_anytime ] && \
    "$PY" reader/exact_duration.py --dataset $ds --arm reader --runs runs/anytime/reader_anytime --accuracy-only
  [ -d runs/anytime/compact_anytime ] && \
    "$PY" reader/exact_duration.py --dataset $ds --arm compact --runs runs/anytime/compact_anytime --accuracy-only
done
"$PY" scripts/collect_exact_duration.py

"$PY" scripts/export_runs.py
"$PY" scripts/score.py --runs runs/cohort --name within_subject --reference bite
"$PY" scripts/paper_tables.py
for ds in 2a 2b sdssvep; do
  "$PY" reader/anytime.py --dataset $ds --reader runs/anytime/reader_anytime
done
"$PY" reader/anytime.py --dataset 2a --reader runs/cohort/reader --out results/anytime_2a_endpoint_supervised.md
"$PY" reader/ablation.py --endpoint runs/cohort/reader --prefix runs/anytime/reader_anytime
"$PY" reader/gate_intervention.py --arm runs/cohort/reader
"$PY" reader/subject_stats.py
"$PY" scripts/ablation_table.py
"$PY" reader/zoo_report.py
"$PY" reader/error_analysis.py
"$PY" scripts/make_figures.py
"$PY" scripts/make_dashboard.py          # reads the files above; must run last
echo "results/ rebuilt"
