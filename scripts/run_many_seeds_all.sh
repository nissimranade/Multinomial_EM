#!/bin/bash
# Launch run_many_seeds.py for all five tumors in parallel, one background
# process per tumor. Per-run progress goes to logs/many_seeds_<tumor>.out,
# the split trace goes to logs/em_many_seeds_<tumor>.log (written by the
# Python script itself).
#
# Usage:
#   bash scripts/run_many_seeds_all.sh            # 500 runs per tumor
#   N_RUNS=10 bash scripts/run_many_seeds_all.sh  # override run count
#
# Output files get a run-count suffix (e.g. t57_em_many_seeds_500runs_assignments.csv)
# so the 500-run batch does not overwrite the earlier 100-run results.
#
# Monitor:   tail -f logs/many_seeds_*.out
# Stop all:  pkill -f run_many_seeds.py

cd "$(dirname "$0")/.." || exit 1

N_RUNS="${N_RUNS:-500}"
SUFFIX="_${N_RUNS}runs"

for t in t24 t25 t31 t46 t57; do
    python scripts/run_many_seeds.py --tumor "$t" --n-runs "$N_RUNS" \
        --out-suffix "$SUFFIX" \
        > "logs/many_seeds_${t}${SUFFIX}.out" 2>&1 &
    echo "Started tumor $t (pid $!)"
done

wait
echo "All tumors finished."
