#!/bin/sh
# Run a pipeline script with the fixed thread settings used for the reported results.
# Thread counts change the order of floating-point sums (in LightGBM and in the maths library behind PCA),
# which shifts LightGBM-based scores by ~0.001-0.002; fixing them makes every run reproduce exactly.
#   usage: ./run.sh src/10_models.py --feature-set full ...
export LGBM_THREADS=6 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$(dirname "$0")/src" || exit 1
script="$(basename "$1")"; shift
exec ../.venv/bin/python "$script" "$@"
