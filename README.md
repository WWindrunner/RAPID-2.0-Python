# RAPID
## v2.0, Apr-14-2026.

## Overview

This project is a Python implementation of the RAPID (RAdar-Produced Inundation Diary) flood mapping workflow, converted from an original MATLAB-based system. It processes Sentinel-1 SAR imagery to generate flood extent maps using both traditional threshold-based (0D) and improved 1D metrics.

The workflow is designed for large-scale processing and supports parallel execution on HPC systems (e.g., SLURM). It integrates multi-step processing including preprocessing, threshold estimation, classification, and post-processing.

This repository aims to:

1. Replace the MATLAB workflow with a fully Python-based pipeline

2. Improve scalability and maintainability

3. Enable flexible batch processing and automation


## Requirements
```
TODO: export python library
```


## Usage
### RAPID.py
```
python -u RAPID.py '<Input_SAR_Pol1>' '<Input_SAR_Pol2>' <Output_Folder>
```
This will run RAPID in single mode with direct input from command line.
```
python -u RAPID.py <PATH_JSON>
```
This will run RAPID with the input parameters that specified in <PATH_JSON>.

### config.json
Specify input files, output folder, and parameters in .json file.


### Data input requirements
SAR Data in GeoTIFF format, preprocessed to sigma0. Other required auxiliary data include Water Occurrence Data and LCC maps will be downloaded automatically.

### Running template
```
python -u RAPID.py 'S1A_IW_GRDH_1SDV_20220501T001528_20220501T001603_043010_05229E_121A/Sigma0_VV.tif' 'S1A_IW_GRDH_1SDV_20220501T001528_20220501T001603_043010_05229E_121A/Sigma0_VH.tif' ./OUTPUT
```
or
```
python -u RAPID.py ./config.json
```

## Output
1. 'TempPol_VV/Cat_init_VV.tif': Result of classification of the VV polarization.
2. 'TempPol_VH/Cat_init_VH.tif': Result of classification of the VH polarization.
3. 'Morph/Morph_final.tif': Final result after the morphological process

## Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.
