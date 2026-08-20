#!/bin/bash

# Process named Sentinel-1 scene folders in the order supplied.
#
# Required environment variables:
#   INPUT_ROOT   root containing one folder per S1 product
#   OUTPUT_ROOT  RAPID output root
#
# Optional environment variables:
#   S1PRO_ENV=s1pro
#   RAPID_ENV=RAPID_python
#   PREPROCESS_GAMMA=true
#   VV_NAME=Gamma0_VV.tif
#   VH_NAME=Gamma0_VH.tif
#   INC_ANGLE_NAME=incidenceAngleFromEllipsoid.tif
#   NORMALIZED_DIR_NAME=normalized
#   RAPID_SCRATCH_ROOT=${SLURM_TMPDIR:-/tmp}
#
# When PREPROCESS_GAMMA=true, inputs are normalized before RAPID and
# --inc_angle_path is deliberately omitted. When false, the original Gamma0
# files are passed to RAPID together with --inc_angle_path so RAPID performs
# the same normalization internally. Never use an incidence angle with inputs
# that have already been normalized.

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
input_root=${INPUT_ROOT:-}
output_root=${OUTPUT_ROOT:-}
s1pro_env=${S1PRO_ENV:-s1pro}
rapid_env=${RAPID_ENV:-RAPID_python}
preprocess_gamma=${PREPROCESS_GAMMA:-true}
vv_name=${VV_NAME:-Gamma0_VV.tif}
vh_name=${VH_NAME:-Gamma0_VH.tif}
inc_angle_name=${INC_ANGLE_NAME:-incidenceAngleFromEllipsoid.tif}
normalized_dir_name=${NORMALIZED_DIR_NAME:-normalized}
scratch_root=${RAPID_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp}}

if [[ -z "$input_root" || -z "$output_root" ]]; then
    echo "Set INPUT_ROOT and OUTPUT_ROOT before running this script." >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: INPUT_ROOT=... OUTPUT_ROOT=... $0 <S1_NAME> [S1_NAME ...]" >&2
    exit 2
fi

case "${preprocess_gamma,,}" in
    true|1|yes) preprocess_gamma=true ;;
    false|0|no) preprocess_gamma=false ;;
    *) echo "PREPROCESS_GAMMA must be true or false." >&2; exit 2 ;;
esac

mkdir -p -- "$output_root"

successful=()
failed=()

for scene in "$@"; do
    scene_dir="$input_root/$scene"
    vv="$scene_dir/$vv_name"
    vh="$scene_dir/$vh_name"
    incidence="$scene_dir/$inc_angle_name"
    scene_scratch="$scratch_root/rapid_${SLURM_JOB_ID:-$$}_$scene"

    echo "[$scene] starting"

    if [[ ! -f "$vv" || ! -f "$vh" || ! -f "$incidence" ]]; then
        echo "[$scene] missing VV, VH, or incidence-angle input; skipping." >&2
        failed+=("$scene:input")
        continue
    fi

    rapid_vv="$vv"
    rapid_vh="$vh"
    rapid_incidence_args=(--inc_angle_path "$incidence")

    if [[ "$preprocess_gamma" == true ]]; then
        normalized_dir="$scene_dir/$normalized_dir_name"
        if ! conda run -n "$s1pro_env" python \
            "$script_dir/S1_download/normalize_gamma0.py" \
            "$vv" "$vh" "$incidence" "$normalized_dir"; then
            echo "[$scene] Gamma0 preprocessing failed; continuing." >&2
            failed+=("$scene:normalization")
            continue
        fi
        rapid_vv="$normalized_dir/Gamma0_VV.tif"
        rapid_vh="$normalized_dir/Gamma0_VH.tif"
        rapid_incidence_args=()
    fi

    rapid_mask_args=()
    lia="$scene_dir/${scene}_LIA.tif"
    ice="$scene_dir/${scene}_ice.tif"
    desert="$scene_dir/${scene}_desert.tif"
    [[ -f "$lia" ]] && rapid_mask_args+=(--LIA_path "$lia")
    [[ -f "$ice" ]] && rapid_mask_args+=(--ice_mask_path "$ice")
    [[ -f "$desert" ]] && rapid_mask_args+=(--desert_mask_path "$desert")

    mkdir -p -- "$scene_scratch"
    if conda run -n "$rapid_env" python -u "$script_dir/RAPID.py" \
        "$rapid_vv" "$rapid_vh" "$output_root" \
        --granules "$scene" \
        --scratch_dir "$scene_scratch" \
        "${rapid_incidence_args[@]}" \
        "${rapid_mask_args[@]}"; then
        successful+=("$scene")
        echo "[$scene] completed"
    else
        failed+=("$scene:RAPID")
        echo "[$scene] RAPID failed; continuing." >&2
    fi
done

echo "Completed scenes: ${#successful[@]}"
if [[ ${#successful[@]} -gt 0 ]]; then
    printf '  %s\n' "${successful[@]}"
fi
echo "Failed scenes: ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
