## Updated on May-15-2023: Adaptive Initialization (AdaI)

# RAPID
RAPID python version.<br>
Please note that the code is under development, so its performance is not guaranteed.


## 1 Requirements
```
conda install gdal rasterio dask scipy pandas scikit-learn dask-ml matplotlib jupyter tqdm -c conda-forge
```

## 2 Usage
### 2.1 Template
```
python -u RAPID.py '<Input_SAR_Pol1>' '<Input_SAR_Pol2>' <Output_Folder>
```
This will run RAPID in single mode with direct input from command line.
```
python -u RAPID.py <PATH_JSON>
```
This will run RAPID with the input parameters that specified in <PATH_JSON>.

### 2.2 config.json
Specify the mode, input files, and output folder in the config.json file.
* "OUT_PATH": Folder path for output results:
* "MODE": RAPID working mode.
  * * (1) 'single': Single image mode, for cases with only one in-event image.
  * * (2) 'normal': Normal mode with one in-event image and one or multiple pre-event images(i.e., dry reference).
  * * (3) 'desert': Desert mode run NDFI based algorithm for desert or arid areas.
  * * (4) 'swd': Self-supervised Water Detection, an alternative algorithm for test purpose.
* "filepath_list": Input file list.
  * * (1) Raster files links or paths: [["file1_pol1","file1_pol2"],["file2_pol1","file2_pol2"],...,["fileN_pol1","fileN_pol2"]]. The first group should be the in-event image, and the rest are pre-event images. Now only support input with dual polarizations, regardless the polarizaton mode.
  * * (2) ASF RTC urls: [["ASF_url1.zip"],["ASF_url2.zip"],...,["ASF_urlN.zip"]].
* "granules" (optional): Granules name for each image, see Demo in 'config.json'.
* Demo is case using ASF RTC30 urls.

### 2.2 Data source
1. ASF RTC service.
   * Please query and submit Sentinel-1 RTC request through ASF website: https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/.
   * Choose Sigma0 in power, better to include incidence angle(optional).
   * Once done, copy the RTC .zip url into config.json, no need to pre-download.
    
2. Local rasters. Simply specify the SAR image files in config.json.


## 3 Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.
