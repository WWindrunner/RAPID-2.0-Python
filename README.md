# RAPID
## v2.0, Apr-14-2026.


## Overview

This project is a Python implementation of the RAPID (RAdar-Produced Inundation Diary) version 2.0 flood mapping workflow, converted from an original MATLAB-based system. It processes SAR imagery to generate flood extent maps.

The workflow is designed for large-scale processing and supports parallel execution on HPC systems (e.g., SLURM). It integrates multi-step processing including preprocessing, threshold estimation, classification, and post-processing.

This repository aims to:

1. Replace the MATLAB workflow with a fully Python-based pipeline

2. Improve scalability and maintainability

3. Enable flexible batch processing and automation


## Requirements
Create the RAPID processing environment:

```
conda env create -f requirements.yml
```

The complete download/preprocessing pipeline keeps its geospatial dependencies
in a separate environment:

```
conda env create -f S1_download/s1pro.yml
```

The automatic-download environment is expected to have ESA SNAP configured.
The pipeline runner uses Bash and standard GNU utilities such as `realpath`.


## Data input requirements

The required SAR inputs are linear Gamma0 power values, not amplitude and not
dB values. VV and VH must be separate, local, single-band GeoTIFF files on
exactly the same grid.

Incidence-angle normalization is required by the algorithm, but an
incidence-angle raster is not always required as an input:

- If the Gamma0 rasters are already incidence-angle normalized, do not provide
  `inc_angle_path`.
- If they are not normalized, provide a local, single-band ellipsoid
  incidence-angle GeoTIFF in degrees through `inc_angle_path`; RAPID will
  perform the normalization internally.

RAPID uses the following algorithm-specific normalization:

```
Gamma0_normalized = Gamma0 / cos(theta_ellipsoid)
```

where `theta_ellipsoid` is the ellipsoid incidence angle.

`inc_angle_path` is different from `LIA_path`. The former is a degree raster
used for incidence-angle normalization, while the latter is an optional binary
terrain/mountain-shadow exclusion mask (`1` where LIA is greater than 50
degrees).

Other required auxiliary data, including Water Occurrence and land-cover
maps, are downloaded automatically.

## Usage

RAPID currently processes one scene per run. It can be run either with
command-line arguments or with a JSON configuration file. All parameters are
equivalent between the two forms. Command-line options use a leading `--`,
while JSON keys do not; for example, `--inc_angle_path` corresponds to
`"inc_angle_path"`.

### Command line

The first SAR input is co-polarization (VV or HH), and the second is
cross-polarization (VH or HV).

```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> [options]
```

For input data that still requires incidence-angle normalization:

```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> --inc_angle_path '<Ellipsoid_Incidence_Angle>'
```

To also apply terrain/mountain-shadow exclusion:

```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> --inc_angle_path '<Ellipsoid_Incidence_Angle>' --LIA_path '<Binary_LIA_Mask>'
```

If no exclusion-mask options are supplied, exclusion masks are not applied.
Important optional inputs also include `--ice_mask_path` for a binary
snow/ice mask and `--desert_mask_path` for a binary desert/dark-area mask. In
both masks, a value of 1 marks pixels to exclude.

For HPC runs, temporary data can be directed to scratch storage with
`--scratch_dir`, for example `--scratch_dir "$SLURM_TMPDIR"`.

### JSON configuration

Run RAPID with a JSON file:

```
python -u RAPID.py <PATH_JSON>
```

Example:

```json
{
  "OUT_PATH": "/path/to/output/",
  "filepath_list": [
    [
      "/path/to/scene_001/Gamma0_VV.tif",
      "/path/to/scene_001/Gamma0_VH.tif"
    ]
  ],
  "inc_angle_path": "/path/to/scene_001/ellipsoid_incidence_angle.tif",
  "LIA_path": "/path/to/scene_001/lia_mask.tif",
  "ice_mask_path": "/path/to/scene_001/snow_ice_mask.tif",
  "cloud_mask_path": "/path/to/scene_001/cloud_reference_mask.tif",
  "desert_mask_path": "/path/to/scene_001/desert_dark_area_mask.tif",
  "granules": ["scene_001"]
}
```

Omit `inc_angle_path` or set it to `null` when the Gamma0 inputs are already
incidence-angle normalized. All JSON parameters use the same names and values
as their command-line counterparts, without the leading `--`. `granules` sets
the scene output-folder name; if it is empty, the name is derived from the
input path.

### SLURM template

`submit_script.sh` provides an `sbatch` example:

```
export CDSE_USERNAME='username'
export CDSE_PASSWORD='password'
sbatch submit_script.sh
```

Edit the project, Conda, output, desert-mask, partition, and product-name
placeholders in the template before submission. Credentials remain in the
submitting environment rather than being stored in the script.

### Complete pipeline from S1 product names

`run_scenes.sh` processes product names in the exact order listed. A failed
scene is reported and does not stop later scenes.

Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OUTPUT_ROOT` | required | Root for every generated file and final result |
| `DOWNLOAD_S1` | `true` | Download by exact product name or use local inputs |
| `INPUT_ROOT` | none | Required only when `DOWNLOAD_S1=false` |
| `CDSE_USERNAME`, `CDSE_PASSWORD` | none | Required only for automatic S1 download |
| `DESERT_MASK_VRT` | none | Fixed global desert-mask input |
| `S1PRO_ENV` | `s1pro` | Download and mask-generation Conda environment |
| `RAPID_ENV` | `RAPID_python` | Flood-processing Conda environment |
| `GENERATE_MASKS` | `true` | Generate snow/cloud, LIA, and desert inputs |
| `PREPROCESS_GAMMA` | `true` | Pre-normalize local Gamma0; local mode only |
| `ORBIT_DIRECTION` | automatic | Optional `ASCENDING`/`DESCENDING` LIA fallback |

Automatic download is the default. It needs no `INPUT_ROOT`; provide
Copernicus credentials through environment variables:

```bash
DOWNLOAD_S1=true \
CDSE_USERNAME='username' \
CDSE_PASSWORD='password' \
OUTPUT_ROOT=/path/to/results \
DESERT_MASK_VRT=/path/to/global_desert_mask.vrt \
bash run_scenes.sh S1_PRODUCT_1 S1_PRODUCT_2
```

Downloaded S1 data, preprocessing products, normalized Gamma0, and mask intermediates
are all written under `<OUTPUT_ROOT>/<S1_PRODUCT>/preprocessing/`. After the
scene completes successfully, the downloaded ZIP and disposable preprocessing products
are removed; failed-scene intermediates are retained for diagnosis.

For existing local scenes, set `DOWNLOAD_S1=false`. Each read-only input folder
must contain `Gamma0_VV.tif`, `Gamma0_VH.tif`, and
`incidenceAngleFromEllipsoid.tif` (names can be overridden with environment
variables). Generated data is written under the corresponding output scene's
`preprocessing/` directory. `OUTPUT_ROOT` must be separate from `INPUT_ROOT`
and cannot be one of its subdirectories:

```bash
DOWNLOAD_S1=false \
INPUT_ROOT=/path/to/scenes \
OUTPUT_ROOT=/path/to/results \
DESERT_MASK_VRT=/path/to/global_desert_mask.vrt \
bash run_scenes.sh S1_PRODUCT_1 S1_PRODUCT_2
```

To skip the separate normalization output and let RAPID normalize the raw
Gamma0 inputs internally with `--inc_angle_path`, use:

```bash
PREPROCESS_GAMMA=false \
DOWNLOAD_S1=false \
INPUT_ROOT=/path/to/scenes \
OUTPUT_ROOT=/path/to/results \
bash run_scenes.sh S1_PRODUCT_1 S1_PRODUCT_2
```

Do not supply an incidence-angle normalization input for Gamma0 rasters that
have already been normalized. When `GENERATE_MASKS=false`, existing masks
named `<S1_PRODUCT>_LIA.tif`, `<S1_PRODUCT>_ice.tif`,
`<S1_PRODUCT>_cloud.tif`, and `<S1_PRODUCT>_desert.tif` are read without being
modified and passed to RAPID when present.

By default, the runner regenerates snow/cloud and binary LIA masks before
RAPID. Set `GENERATE_MASKS=false` to reuse existing masks. Snow is set when any
valid Sentinel-2 observation day in the preceding 15-day window sees snow or
ice. Cloud is set only when every valid observation on every observed day is
cloud or cloud shadow. Pixels without a valid observation are nodata (255).
The cloud mask is aligned and retained by RAPID as a reference but is not used
for exclusion.

Set `DESERT_MASK_VRT` to the fixed global desert-mask input when masks are
generated. RAPID intersects the clipped desert mask with ESA WorldCover class
60 (bare/sparse vegetation), uses only that effective mask for exclusion, and
saves the four aligned uint8 products under each result scene:

```text
<OUTPUT_ROOT>/<S1_PRODUCT>/masks/
├── snow_mask.tif
├── cloud_mask.tif
├── lia_mask.tif
└── desert_mask.tif
```

Values are `0` (false), `1` (true), and `255` (nodata). The saved desert mask
is the effective WorldCover intersection, not the full input desert layer.
Generated normalized Gamma0 and pre-RAPID masks are kept under
`<OUTPUT_ROOT>/<S1_PRODUCT>/preprocessing/`; user-provided input files are
never overwritten or deleted.


## Main Output
1. `TempPol_VV/Cat_init_VV.tif`: Result of classification of the VV polarization.
2. `TempPol_VH/Cat_init_VH.tif`: Result of classification of the VH polarization.
3. `Morph/Morph_final.tif`: Final result after the morphological process
4. `quicklook.png` and `quicklook_map.png`: An overview of the final result with LCC map, Google Earth map, and the original SAR as reference.


## Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.

## Contact
Maopu Xu`maopuxu@uwm.edu`
