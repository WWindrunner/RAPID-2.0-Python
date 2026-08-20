# Sentinel-1 Download and Processing

This project downloads specified Sentinel-1 GRD products and generates the
SAR and ancillary raster products used by the downstream workflow.

## Processing workflow

For each Sentinel-1 product name, the workflow runs the following steps:

1. `Sentinel_1_specific_name_download_process.py`
   - Searches for and downloads the Sentinel-1 GRD product from Copernicus
     Data Space.
   - Uses SNAP through pyroSAR to process VV and VH backscatter.
   - Produces the incidence-angle rasters required by the next step.
2. `Desert_mask.py`
   - Clips and reprojects the external global desert-mask VRT to the
     Sentinel-1 grid.
   - Preserves the mask classes using nearest-neighbour resampling.
3. `cal_LIA.py`
   - Downloads Copernicus DEM tiles from Microsoft Planetary Computer.
   - Resolves orbit direction from a manifest, ZIP, public catalogue query, or
     explicit fallback argument.
   - Writes a uint8 mask with 1 where LIA is greater than 50 degrees.
4. `Snow_detect.py`
   - Searches for Sentinel-2 L2A products from the 15 days before the
     Sentinel-1 acquisition.
   - Gets the date from the S1 name and the extent from the local Gamma0, so
     the original S1 ZIP is not required.
   - Uses the SCL downloads once to generate both snow/ice and persistent-cloud
     masks on the Sentinel-1 grid.

The repository-level multi-scene runner controls stage ordering. Temporary
ZIP, DEM, SNAP, and Sentinel-2 files should only be cleaned after the relevant
scene succeeds; failed-scene intermediates are retained for inspection or
retry.

## Environment

The current Conda environment was exported to `s1pro.yml`:

```bash
conda env create -f s1pro.yml
conda activate s1pro
```

The workflow also requires ESA SNAP. Make sure the SNAP `bin` directory is
available on `PATH`. The current server-specific SNAP path is set in
`Sentinel_1_specific_name_download_process.py` and may need to be changed on a
different system.

Copernicus Data Space credentials are only required when the optional
Sentinel-1 download script is used. They are passed to that script as command
line arguments. Existing local Gamma0 and incidence-angle inputs do not need
Copernicus credentials.

## Running the utilities

These utilities are part of the RAPID repository. Use the repository-level
`run_scenes.sh` for ordered multi-scene processing. Sentinel-1 downloading is
optional; when it is needed, the individual stages can be run manually:

```bash
python Sentinel_1_specific_name_download_process.py \
  <S1_PRODUCT_NAME> <OUTPUT_FOLDER> <USERNAME> <PASSWORD>
python Desert_mask.py <S1_PRODUCT_NAME> <OUTPUT_FOLDER> <DESERT_MASK_VRT>
python cal_LIA.py <S1_PRODUCT_NAME> <OUTPUT_FOLDER>
python Snow_detect.py <S1_PRODUCT_NAME> <OUTPUT_FOLDER>
```

If LIA orbit direction cannot be obtained automatically, provide it directly:

```bash
python cal_LIA.py <S1_PRODUCT_NAME> <OUTPUT_FOLDER> \
  --incidence-angle /path/to/incidence_angle.tif \
  --orbit-direction ASCENDING
```

For already-downloaded, unnormalized linear Gamma0 rasters, normalization can
be run without downloading Sentinel-1 again:

```bash
python normalize_gamma0.py \
  <RAW_GAMMA0_VV> <RAW_GAMMA0_VH> \
  <ELLIPSOID_INCIDENCE_ANGLE> <OUTPUT_FOLDER>
```

This writes `Gamma0_VV.tif` and `Gamma0_VH.tif` as
`Gamma0 / cos(ellipsoid incidence angle)`. These outputs must be passed to
RAPID without `--inc_angle_path`. Alternatively, pass the original
unnormalized rasters and `--inc_angle_path` directly to RAPID and omit this
separate normalization step.

## Output structure

Each Sentinel-1 product is written to its own subdirectory:

```text
<OUTPUT_FOLDER>/
+-- <S1_PRODUCT_NAME>/
    +-- Gamma0_VV.tif
    +-- Gamma0_VH.tif
    +-- <S1_PRODUCT_NAME>_desert.tif
    +-- <S1_PRODUCT_NAME>_LIA.tif
    +-- <S1_PRODUCT_NAME>_ice.tif
    +-- <S1_PRODUCT_NAME>_cloud.tif
```

Intermediate SNAP output files are removed after successful completion.
