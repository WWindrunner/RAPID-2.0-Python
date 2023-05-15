## Updated on May-15-2023: Adaptive Initialization (AdaI)

# RAPID
RAPID system in python version. Please note that the code is still under development, so its performance is not guaranteed.


## 1 Requirements
```
conda install gdal rasterio dask scipy pandas h5py jupyter tqdm -c conda-forge
```

## 2 Usage
### 2.1 Data source
1. ASF RTC service.
   * Please query and submit Sentinel-1 RTC request through ASF website: https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/.
   * Choose Sigma0 in power, better to include incidence angle(optional).
   * Once done, copy the RTC .zip url into config.json, no need to pre-download.
    
2. Local rasters. Simply specify the SAR image files in config.json.

### 2.2 config.json
Specify the output folder, mode, and input files in the config.json file.
* "OUT_PATH": Folder path for output results:
* "MODE": RAPID working mode.
  * * (1) 'normal': Normal mode with one in-event image and one or multiple pre-event images.
  * * (2) 'desert': Desert mode run NDFI based algorithm for desert or arid areas.
  * * (3) 'single': Single image mode, for cases with only one in-event image.
* "filepath_list": Input file list.
  * * (1) ASF RTC urls: [["ASF_url1.zip"],["ASF_url2.zip"],...,["ASF_urlN.zip"]]
  * * (2) local raster files: ["VV_file1","VH_file1"],["VV_file2","VH_file2"],...,["VV_fileN","VH_fileN"]
* "granules" (optional): Granules name for each image, see Demo.
* Demo is using ASF RTC30 urls.

### 2.3 Run RAPID.py
```
python -u RAPID.py
```
This will use the config.json file located in codes' folder.
```
python -u RAPID.py PATH_JSON
```
This will use .jsin file specify in PATH_JSON.

## 3 Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.
