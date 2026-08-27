#!/bin/bash

#SBATCH --output=rapid-%j.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rapid_pipeline

set -euo pipefail

# Site-specific setup: update these two paths.
project_root="/path/to/RAPID-2.0-Python"
source "/path/to/miniconda3/etc/profile.d/conda.sh"
cd "$project_root"

export S1PRO_ENV="s1pro"
export RAPID_ENV="RAPID_python"
export OUTPUT_ROOT="/path/to/results"
# Optional: keep downloaded/normalized Gamma0 source data outside OUTPUT_ROOT.
# export PREPROCESS_ROOT="/path/to/preprocessed-data"
export DESERT_MASK_VRT="/path/to/global_desert_mask.vrt"
export RAPID_SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"

# Automatic download mode. Export CDSE_USERNAME and CDSE_PASSWORD in the
# submitting shell; Slurm normally carries exported variables into the job.
export DOWNLOAD_S1=true
: "${CDSE_USERNAME:?Export CDSE_USERNAME before sbatch}"
: "${CDSE_PASSWORD:?Export CDSE_PASSWORD before sbatch}"

scenes=(
    "S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_ORBIT_DATATAKE_ID"
)

bash "$project_root/run_scenes.sh" "${scenes[@]}"

# For read-only local inputs instead, replace the download settings above with:
# export DOWNLOAD_S1=false
# export INPUT_ROOT="/path/to/local/scenes"
# export PREPROCESS_GAMMA=true
