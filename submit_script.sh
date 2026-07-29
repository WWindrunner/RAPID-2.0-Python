#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --job-name=rapid_python

set -e
cd RAPID-2.0-Python

conda activate RAPID_python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Optional explicit Dask worker limit. If omitted, RAPID uses 90% of the
# memory allocated by SLURM; outside SLURM it defaults to 100GB.
# export RAPID_MEMORY_LIMIT=108GB

# Keep Dask spill and all RAPID temporary files off the home filesystem.
# Sites without SLURM_TMPDIR should replace /tmp with their HPC scratch path.
scratch_base="${SLURM_TMPDIR:-/tmp}"
run_id="${SLURM_JOB_ID:-$$}"
export RAPID_SCRATCH="${scratch_base%/}/rapid_${run_id}"
mkdir -p "$RAPID_SCRATCH"
export TMPDIR="$RAPID_SCRATCH"
export DASK_TEMPORARY_DIRECTORY="$RAPID_SCRATCH"

cleanup() {
    if [[ -n "${RAPID_SCRATCH:-}" && "$(basename "$RAPID_SCRATCH")" == "rapid_${run_id}" ]]; then
        rm -rf -- "$RAPID_SCRATCH"
    fi
}
trap cleanup EXIT

python3 -u RAPID.py 'S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/Gamma0_VV.tif' 'S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/Gamma0_VH.tif' output/20210829 --inc_angle_path 'S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/ellipsoid_incidence_angle.tif' --scratch_dir "$RAPID_SCRATCH"
