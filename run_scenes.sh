#!/bin/bash

# Process named Sentinel-1 scene folders in the order supplied.
#
# Required environment variables:
#   OUTPUT_ROOT  RAPID output root
#   INPUT_ROOT   only when DOWNLOAD_S1=false
#
# Optional environment variables:
#   DOWNLOAD_S1=true
#   CDSE_USERNAME / CDSE_PASSWORD (required when DOWNLOAD_S1=true)
#   SNAP_BIN=                    optional ESA SNAP bin directory
#   S1PRO_ENV=s1pro
#   RAPID_ENV=RAPID_python
#   PREPROCESS_GAMMA=true
#   GENERATE_MASKS=true
#   ORBIT_DIRECTION=             optional LIA fallback: ASCENDING/DESCENDING
#   DESERT_MASK_VRT=             fixed global desert-mask VRT
#   VV_NAME=Gamma0_VV.tif
#   VH_NAME=Gamma0_VH.tif
#   INC_ANGLE_NAME=incidenceAngleFromEllipsoid.tif
#   RAPID_SCRATCH_ROOT=${SLURM_TMPDIR:-/tmp}
#
# When PREPROCESS_GAMMA=true, inputs are normalized before RAPID and
# --inc_angle_path is deliberately omitted. When false, the original Gamma0
# files are passed to RAPID together with --inc_angle_path so RAPID performs
# the same normalization internally. Never use an incidence angle with inputs
# that have already been normalized.
# DESERT_MASK_VRT should point to the fixed global desert-mask dataset. If it
# is omitted, each scene must already contain <S1_NAME>_desert.tif.

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
input_root=${INPUT_ROOT:-}
output_root=${OUTPUT_ROOT:-}
s1pro_env=${S1PRO_ENV:-s1pro}
rapid_env=${RAPID_ENV:-RAPID_python}
preprocess_gamma=${PREPROCESS_GAMMA:-true}
download_s1=${DOWNLOAD_S1:-true}
generate_masks=${GENERATE_MASKS:-true}
orbit_direction=${ORBIT_DIRECTION:-}
desert_mask_vrt=${DESERT_MASK_VRT:-}
vv_name=${VV_NAME:-Gamma0_VV.tif}
vh_name=${VH_NAME:-Gamma0_VH.tif}
inc_angle_name=${INC_ANGLE_NAME:-incidenceAngleFromEllipsoid.tif}
scratch_root=${RAPID_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp}}

if [[ -z "$output_root" ]]; then
    echo "Set OUTPUT_ROOT before running this script." >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: OUTPUT_ROOT=... $0 <S1_NAME> [S1_NAME ...]" >&2
    exit 2
fi

case "${preprocess_gamma,,}" in
    true|1|yes) preprocess_gamma=true ;;
    false|0|no) preprocess_gamma=false ;;
    *) echo "PREPROCESS_GAMMA must be true or false." >&2; exit 2 ;;
esac
case "${download_s1,,}" in
    true|1|yes) download_s1=true ;;
    false|0|no) download_s1=false ;;
    *) echo "DOWNLOAD_S1 must be true or false." >&2; exit 2 ;;
esac
case "${generate_masks,,}" in
    true|1|yes) generate_masks=true ;;
    false|0|no) generate_masks=false ;;
    *) echo "GENERATE_MASKS must be true or false." >&2; exit 2 ;;
esac

output_root=$(realpath -m -- "$output_root")
if [[ "$download_s1" == false ]]; then
    if [[ -z "$input_root" ]]; then
        echo "Set INPUT_ROOT when DOWNLOAD_S1=false." >&2
        exit 2
    fi
    if ! input_root=$(realpath -e -- "$input_root"); then
        echo "INPUT_ROOT does not exist: $input_root" >&2
        exit 2
    fi
    case "$output_root/" in
        "$input_root/"|"$input_root/"*)
            echo "OUTPUT_ROOT must not be the input directory or one of its subdirectories." >&2
            exit 2
            ;;
    esac
else
    if [[ -z "${CDSE_USERNAME:-}" || -z "${CDSE_PASSWORD:-}" ]]; then
        echo "Set CDSE_USERNAME and CDSE_PASSWORD when DOWNLOAD_S1=true." >&2
        exit 2
    fi
fi
mkdir -p -- "$output_root"

successful=()
failed=()

for scene in "$@"; do
    scene_output="$output_root/$scene"
    preprocess_dir="$scene_output/preprocessing"
    scene_scratch="$scratch_root/rapid_${SLURM_JOB_ID:-$$}_$scene"
    generated_desert=false

    echo "[$scene] starting"

    if [[ "$download_s1" == true ]]; then
        mkdir -p -- "$preprocess_dir"
        rapid_vv="$preprocess_dir/Gamma0_VV.tif"
        rapid_vh="$preprocess_dir/Gamma0_VH.tif"
        incidence=$(find "$preprocess_dir" -maxdepth 1 -type f \
            -name '*incidenceAngleFromEllipsoid.tif' -print -quit)
        if [[ ! -f "$rapid_vv" || ! -f "$rapid_vh" || -z "$incidence" ]]; then
            if ! conda run -n "$s1pro_env" python \
                "$script_dir/S1_download/Sentinel_1_specific_name_download_process.py" \
                "$scene" "$preprocess_dir"; then
                echo "[$scene] Sentinel-1 download/preprocessing failed; continuing." >&2
                failed+=("$scene:S1-download")
                continue
            fi
            incidence=$(find "$preprocess_dir" -maxdepth 1 -type f \
                -name '*incidenceAngleFromEllipsoid.tif' -print -quit)
        fi
        if [[ ! -f "$rapid_vv" || ! -f "$rapid_vh" || -z "$incidence" ]]; then
            echo "[$scene] downloaded preprocessing outputs are incomplete." >&2
            failed+=("$scene:S1-preprocessing")
            continue
        fi
        scene_dir="$preprocess_dir"
        rapid_incidence_args=()
    else
        scene_dir="$input_root/$scene"
        vv="$scene_dir/$vv_name"
        vh="$scene_dir/$vh_name"
        incidence="$scene_dir/$inc_angle_name"
        if [[ ! -f "$vv" || ! -f "$vh" || ! -f "$incidence" ]]; then
            echo "[$scene] missing VV, VH, or incidence-angle input; skipping." >&2
            failed+=("$scene:input")
            continue
        fi
        rapid_vv="$vv"
        rapid_vh="$vh"
        rapid_incidence_args=(--inc_angle_path "$incidence")
    fi

    if [[ "$download_s1" == false && "$preprocess_gamma" == true ]]; then
        normalized_dir="$preprocess_dir"
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

    if [[ "$generate_masks" == true ]]; then
        mkdir -p -- "$preprocess_dir"
        if ! conda run -n "$s1pro_env" python \
            "$script_dir/S1_download/Snow_detect.py" \
            "$scene" "$rapid_vv" "$preprocess_dir"; then
            echo "[$scene] snow/cloud mask generation failed; continuing." >&2
            failed+=("$scene:snow-cloud")
            continue
        fi

        lia_direction_args=()
        if [[ -n "$orbit_direction" ]]; then
            lia_direction_args=(--orbit-direction "$orbit_direction")
        fi
        if ! conda run -n "$s1pro_env" python \
            "$script_dir/S1_download/cal_LIA.py" "$scene" "$preprocess_dir" \
            --incidence-angle "$incidence" \
            --metadata-dir "$scene_dir" \
            "${lia_direction_args[@]}"; then
            echo "[$scene] LIA mask generation failed; continuing." >&2
            failed+=("$scene:LIA")
            continue
        fi

        desert="$preprocess_dir/raw_desert_mask.tif"
        if [[ -n "$desert_mask_vrt" ]]; then
            if [[ ! -f "$desert_mask_vrt" ]]; then
                echo "[$scene] DESERT_MASK_VRT does not exist: $desert_mask_vrt" >&2
                failed+=("$scene:desert-input")
                continue
            fi
            if ! conda run -n "$s1pro_env" python \
                "$script_dir/S1_download/Desert_mask.py" \
                "$rapid_vv" "$desert_mask_vrt" "$desert"; then
                echo "[$scene] desert-mask clipping failed; continuing." >&2
                failed+=("$scene:desert")
                continue
            fi
            generated_desert=true
        elif [[ ! -f "$desert" ]]; then
            echo "[$scene] set DESERT_MASK_VRT or provide $desert; continuing." >&2
            failed+=("$scene:desert-input")
            continue
        fi
    fi

    rapid_mask_args=()
    lia="$preprocess_dir/lia_mask.tif"
    ice="$preprocess_dir/snow_mask.tif"
    cloud="$preprocess_dir/cloud_mask.tif"
    desert="$preprocess_dir/raw_desert_mask.tif"
    if [[ "$generate_masks" == false ]]; then
        [[ -f "$scene_dir/${scene}_LIA.tif" ]] && lia="$scene_dir/${scene}_LIA.tif"
        [[ -f "$scene_dir/${scene}_ice.tif" ]] && ice="$scene_dir/${scene}_ice.tif"
        [[ -f "$scene_dir/${scene}_cloud.tif" ]] && cloud="$scene_dir/${scene}_cloud.tif"
        [[ -f "$scene_dir/${scene}_desert.tif" ]] && desert="$scene_dir/${scene}_desert.tif"
    fi
    [[ -f "$lia" ]] && rapid_mask_args+=(--LIA_path "$lia")
    [[ -f "$ice" ]] && rapid_mask_args+=(--ice_mask_path "$ice")
    [[ -f "$cloud" ]] && rapid_mask_args+=(--cloud_mask_path "$cloud")
    [[ -f "$desert" ]] && rapid_mask_args+=(--desert_mask_path "$desert")

    mkdir -p -- "$scene_scratch"
    if conda run -n "$rapid_env" python -u "$script_dir/RAPID.py" \
        "$rapid_vv" "$rapid_vh" "$output_root" \
        --granules "$scene" \
        --scratch_dir "$scene_scratch" \
        "${rapid_incidence_args[@]}" \
        "${rapid_mask_args[@]}"; then
        successful+=("$scene")
        if [[ "$generated_desert" == true ]]; then
            rm -f -- "$desert"
        fi
        if [[ "$download_s1" == true ]]; then
            rm -f -- \
                "$preprocess_dir"/*.zip \
                "$preprocess_dir"/*incidenceAngleFromEllipsoid.tif \
                "$preprocess_dir"/*localIncidenceAngle.tif \
                "$preprocess_dir"/*_manifest.safe \
                "$preprocess_dir"/*_proc.xml \
                "$preprocess_dir"/*_gamma0-elp.tif
        fi
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
