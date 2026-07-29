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


## Usage
### RAPID.py
This runs RAPID in single-scene mode. VV and VH must be local, single-band
GeoTIFF files. The first SAR input is co-polarization (VV or HH).
```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> [other possible parameters]
```
If `--inc_angle_path` is omitted, the input Gamma0 power rasters are assumed
to have already been incidence-angle normalized. To perform normalization in
RAPID, provide an ellipsoid incidence-angle raster:
```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> --inc_angle_path '<Ellipsoid_Incidence_Angle>'
```
Without specifying other parameters, the RAPID v2 default mode will NOT apply exclusion masks.

If a terrain/mountain shadow mask is wanted, please run by the `--LIA_path` parameter, as shown below.
```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> --inc_angle_path "path_to_ellipsoid_incidence_angle.tif" --LIA_path "path_to_lia.tif" [other possible parameters]
```

For HPC runs, direct Dask spill and other temporary data to node-local or
high-capacity scratch storage:
```
python -u RAPID.py '<Input_Gamma0_CoPol>' '<Input_Gamma0_CrossPol>' <Output_Folder> --inc_angle_path "path_to_ellipsoid_incidence_angle.tif" --scratch_dir "$SLURM_TMPDIR"
```
If `--scratch_dir` is omitted, RAPID checks `RAPID_SCRATCH`, then
`SLURM_TMPDIR`. Outside a configured HPC environment it uses `cache` beneath
the output directory, falling back to the system temporary directory only if
the output cache cannot be created. Each run creates and cleans its own
job-specific subdirectory.

In the default Dask thread mode, RAPID automatically assigns 90% of the memory
reported by SLURM to the Dask worker, leaving 10% for Python, GDAL, plotting,
and other unmanaged allocations. Outside SLURM the default is 100GB. An
explicit worker limit can be supplied when needed:
```
export RAPID_MEMORY_LIMIT=108GB
```

This will run RAPID with the input parameters that specified in <PATH_JSON>.
```
python -u RAPID.py <PATH_JSON>
```

### config.json
Specify input files, output folder, and parameters in .json file. 

Example of file input is shown here. 
```
"filepath_list": [
    [
      "/path/to/scene_001/Gamma0_VV.tif",
      "/path/to/scene_001/Gamma0_VH.tif"
    ]
  ],
```
`granules` specifies the output folder name. If left empty, it will be extracted from the input path.
```
"granules": [
    "scene_001"
  ],
```
Besides the file paths and granules, other parameters can be added same as the command line mode.
```
"inc_angle_path": "/path/to/scene_001/ellipsoid_incidence_angle.tif",
"LIA_path": "/path/to/scene_001/local_incidence_angle.tif"
```
Note that the parameters will affect the running results, and should not be modified at will.
They can be left empty in the config file, and their default value will be used.

### Data input requirements

The required SAR inputs are linear Gamma0 power values, not amplitude and not
dB values. VV and VH must be separate, local, single-band GeoTIFF files on
exactly the same grid.

The optional ellipsoid incidence-angle input must be a local, single-band
GeoTIFF. Its values must be in degrees. When it is supplied, RAPID aligns this
continuous raster to the SAR grid with bilinear resampling and applies:

```
Gamma0_normalized = Gamma0 / cos(theta_ellipsoid)
```

Invalid incidence-angle pixels are kept at zero and excluded from
classification. `inc_angle_path` is distinct from `LIA_path`: the former is
the ellipsoid incidence angle used for radiometric normalization, while the
latter is an optional local incidence angle used for terrain exclusion.
When `inc_angle_path` is omitted, RAPID assumes VV and VH are already
incidence-angle-normalized Gamma0 power and does not modify them.

Other required auxiliary data, including Water Occurrence and land-cover
maps, are downloaded automatically.

### Running template
Please check the file `submit_script.sh` for an example. It is designed to be used by `sbatch` command for a Slurm scheduler.
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
