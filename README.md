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
Run the following command to install the conda environment for RAPID v2.0.
```
conda env create -f requirements.yml
```


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

`inc_angle_path` is different from `LIA_path`. The former is used for
incidence-angle normalization, while the latter is an optional local incidence
angle raster used for terrain/mountain-shadow exclusion.

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
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> --inc_angle_path '<Ellipsoid_Incidence_Angle>' --LIA_path '<Local_Incidence_Angle>'
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
  "LIA_path": "/path/to/scene_001/local_incidence_angle.tif",
  "ice_mask_path": "/path/to/scene_001/snow_ice_mask.tif",
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
sbatch submit_script.sh
```


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
