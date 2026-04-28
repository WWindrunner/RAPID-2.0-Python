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
```
python -u RAPID.py '<Input_SAR_CoPol>' '<Input_SAR_CrossPol>' <Output_Folder>
```
This will run RAPID in single mode with direct input from command line. Note that by default the first input file should be cross polarization (VV or HH).
```
python -u RAPID.py <PATH_JSON>
```
This will run RAPID with the input parameters that specified in <PATH_JSON>.

### config.json
Specify input files, output folder, and parameters in .json file.

### Data input requirements
SAR Data in GeoTIFF format, without speckle filter, preprocessed to sigma0. Other required auxiliary data include Water Occurrence Data and LCC maps will be downloaded automatically.

### Running template
Please check the file `submit_script.sh` for an example. It is designed to be used by `sbatch` command for a Slurm scheduler.
```
sbatch submit_script.sh
```


## Output
1. `TempPol_VV/Cat_init_VV.tif`: Result of classification of the VV polarization.
2. `TempPol_VH/Cat_init_VH.tif`: Result of classification of the VH polarization.
3. `Morph/Morph_final.tif`: Final result after the morphological process
4. `quicklook.png` and `quicklook_map.png`: An overview of the final result with LCC map, Google Earth map, and the original SAR as reference.


## Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.
