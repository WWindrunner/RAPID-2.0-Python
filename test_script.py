"""TEST SCRIPT for rioxarray use with cloud urls

Josef Kellndorfer, 2022-11-06

"""

import os
import rioxarray as xr
import numpy as np

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

TESTCASES = ("red_river_20200416", "hurricane_ian_20221002")
varnames = ['season_lp', 'season_cp', 'month_lp', 'month_cp', 'img_lp', 'img_cp']


for TESTCASE in TESTCASES:

    print(f'\n****TESTCASE: {TESTCASE}', flush=True)

    # Get the URLS

    TESTURLS = f"{TESTCASE}_urls_presigned.txt"
    URLS = os.path.join(os.environ['REPO_HOME'], 'data', 'urls', TESTURLS)

    with open(URLS) as f:
        urls = f.readlines()

    # Make a rioxarray data set of the baseline and flood image variables

    var_dict = {varname: urls[i].strip() for i, varname in enumerate(varnames)}

    print(f"Open with rasterio: {var_dict[varnames[0]]}")

    da = xr.open_rasterio(var_dict[varnames[0]], chunks=(1, 512, 512))

    ds = da.sel({'band': 1}, drop=True).to_dataset(name=varnames[0])  

    for varname in varnames[1:]:
        print(f"Open with rasterio: {var_dict[varname]}")
        ds[varname] = xr.open_rasterio(var_dict[varname], chunks=(1, 512, 512)).sel({'band': 1}, drop=True)

    # Apply a simple log_ratio classifier for flood detection

    ratio = ds.img_lp / ds.season_lp

    logratio = ratio.reduce(np.log)

    flood = (logratio < -0.4).astype(np.uint8)

    # Write the classified image to disk

    outdir = os.path.join(os.environ['HOME'], 'junk', 'rapid_test')
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, TESTCASE + '_flood.tif')

    flood.rio.to_raster(outname)

    print(f"Wrote output {outname}")
