#!/bin/bash

#SBATCH --output=/tank/data/SFS/xinyis/shared/src/RAPID_python/RAPID_2.0_Python/logs/log-%j.out
#SBATCH --partition=768g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --job-name=rapid_python

set -e
cd /tank/data/SFS/xinyis/shared/src/RAPID_python/RAPID_2.0_Python/RAPID-2.0-Python

source /home/uwm/maopuxu/.bashrc
conda activate RAPID_python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 -u RAPID.py '/tank/data/SFS/xinyis/shared/src/RAPID_python/RAPID_2.0_Python/data/20210829/S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/Gamma0_VV.tif' '/tank/data/SFS/xinyis/shared/src/RAPID_python/RAPID_2.0_Python/data/20210829/S1A_IW_GRDH_1SDV_20210829T000202_20210829T000227_039437_04A8B3_BDF7/Gamma0_VH.tif' /tank/data/SFS/xinyis/shared/src/RAPID_python/RAPID_2.0_Python/output/20210829


