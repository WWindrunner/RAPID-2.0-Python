## Updated on May-15-2023: Adaptive Initialization (AdaI)

# RAPID
RAPID python version.<br>
Please note that the code is under development, so performance is not guaranteed.


## 1 Requirements
```
conda install gdal rasterio dask scipy pandas scikit-learn dask-ml matplotlib jupyter tqdm -c conda-forge
```

## 2 Usage
### 2.1 RAPID.py
```
python -u RAPID.py '<Input_SAR_Pol1>' '<Input_SAR_Pol2>' <Output_Folder>
```
This will run RAPID in single mode with direct input from command line.
```
python -u RAPID.py <PATH_JSON>
```
This will run RAPID with the input parameters that specified in <PATH_JSON>.

### 2.2 config.json
Specify the mode, input files, and output folder in .json file.
* "OUT_PATH": Folder path for output.
* "MODE": RAPID working mode.
  * * (1) 'single': Single image mode, for cases with only one in-event image.
  * * (2) 'normal': Normal mode with one in-event image and one or multiple pre-event images(i.e., dry reference).
  * * (3) 'desert': Desert mode run NDFI based algorithm for desert or arid areas.
  * * (4) 'swd': Self-supervised Water Detection, an alternative algorithm for test purpose.
* "filepath_list": Input file list.
  * * (1) Raster files links or paths: [["file1_pol1","file1_pol2"],["file2_pol1","file2_pol2"],...,["fileN_pol1","fileN_pol2"]]. The first group should be the in-event image, and the rest are pre-event images.
  * * (2) ASF RTC urls: [["ASF_url1.zip"],["ASF_url2.zip"],...,["ASF_urlN.zip"]].
* "granules" (optional): Granules name for each image, will extract from input file path if not provided. Refer to 'config.json' for more info.
* 'config.json' shows a demo using ASF RTC30 urls. The urls are valid until 05/27/2023.

### 2.3 Data input requirements
1. Dual polarizations.
   * Currently, only input with dual polarizations is supported, regardless of the polarization mode.
2. Amplitude or sigma0.
   * If input with amplitude, ensure that 'AMP' or 'amp' is included in the input file name.(Current pipline from Earth Big Data)
   * Otherwise the input should be in sigma0. (Current pipline from ASF RTC service)

### 2.4 Running template
```
python -u RAPID.py 'https://ebd-clients-w.s3.amazonaws.com/NOAA/RCM/COG/RCM2_OK2535409_PK2538063_SC30MCPB_20230503_124750UTC-EBD_CH_AMP.tif?AWSAccessKeyId=AKIAJFTC5FRGKQV5C5KQ&Signature=j73oGsBfQsOscPBLlpCvv%2FVk6W0%3D&Expires=1769538603' \
'https://ebd-clients-w.s3.amazonaws.com/NOAA/RCM/COG/RCM2_OK2535409_PK2538063_SC30MCPB_20230503_124750UTC-EBD_CV_AMP.tif?AWSAccessKeyId=AKIAJFTC5FRGKQV5C5KQ&Signature=b8s8CaOBDjQIlF%2FjZX5Qo6dLhc0%3D&Expires=1769538603' \
./OUTPUT
```
or
```
python -u RAPID.py ./config.json
```

## 3 Output
### 3.1 Output files
1. 'RAPID_2classes_RGB_<granuele_name>.tif': RGB tif result labeling persistent and flood water, recommend for web display.
2. 'RAPID_3classes_<granuele_name>.tif': RGB tif result labeling persistent, pre-event and flood water, when running normal mode with dry reference.
3. 'RAPID_RiverIce_2classes_RGB_<granuele_name>.tif': RGB tif result labeling persistent, river ice and flood water, under development.
4. 'RAPID_raw_<granuele_name>.tif': Raw binary raster, all water pixels are labeled as 1.
5. '*.png, *.jpg': Quick viualization of the rasters.
6. Folder named by <granuele_name>: Contain intermediate result files.

### 3.2 Legend for RGB tif
{0 92 230 Persistent water<br>
0 197 255 Pre-event water<br>
219 0 0 Flood water<br>
255 215 0 River ice
}

## 4 Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.
