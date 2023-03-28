# RAPID
RAPID system in python version. It includes an auto ancillary data preparation and three working mode (Normal, desert and single). Please note that the code is still under development, so its performance is not guaranteed.


## 1 Requirements
```
conda install gdal rasterio dask scipy pandas h5py tqdm jupyter -c conda-forge
```

## 2 Usage
### 2.1 config.json
Specify the input files and output folder. Demo run is a case using ASF RTC30 images.

### 2.2 RAPID.py
```
python -u RAPID.py
```

## 3 Reference
1. Xinyi Shen, et al. (2019), Nearreal-time non-obstructed flood inundation mapping using synthetic aperture radar. Remote Sens. Environ., 221, 302–315, https://doi.org/10.1016/j.rse.2018.11.008.
2. Qing Yang, et al. (2021), A high-resolution flood inundation archive (2016–the present) from sentinel-1 SAR imagery over CONUS. Bull. Amer. Meteor. Soc., 102, E1064–E1079, https://doi.org/10.1175/BAMS-D-19-0319.1.
