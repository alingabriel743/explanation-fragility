#!/bin/sh
# Reproduces every result of the paper. The grids take several hours on a 14-core machine.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python experiments/tune.py 0 10
for grid in main fresh mixed mixedfresh relational noise; do
    python experiments/run_grid.py "$grid"
done
python experiments/run_synthetic.py
python experiments/run_lime.py
python experiments/run_smote_factorial.py
python experiments/run_scale_check.py
python analysis/analyze.py
python analysis/analyze_monitoring.py
