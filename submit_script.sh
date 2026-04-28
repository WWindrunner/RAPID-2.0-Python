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
export SLURM_MEM_PER_NODE=120 # IMPORTANT: this sets the memory limit, and if SLURM_MEM_PER_NODE is not found, a default of 100G will be used

python3 -u RAPID.py 'S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/Gamma0_VV.tif' 'S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/Gamma0_VH.tif' output/20210829
