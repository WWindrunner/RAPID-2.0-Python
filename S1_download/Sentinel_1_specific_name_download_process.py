import argparse
import datetime
import glob
import os
from sys import stdout

import numpy as np
import requests
from osgeo import gdal
from pyroSAR.snap.util import geocode

snap_bin = os.environ.get("SNAP_BIN")
if snap_bin:
    os.environ["PATH"] = snap_bin + os.pathsep + os.environ.get("PATH", "")


def log_in(username, password):
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password
    }
    response = requests.post(auth_url, data=auth_data, timeout=60)
    response.raise_for_status()
    access_token = response.json().get("access_token")

    if not access_token:
        raise RuntimeError("Copernicus authentication returned no access token")
    return access_token

def search_sentinel_with_S1name(S1name):
    S1name_with_safe = S1name + ".SAFE"
    query_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Name eq '{S1name_with_safe}'"
    )
    response = requests.get(query_url, timeout=60)
    response.raise_for_status()
    response_json = response.json()
    return response_json.get("value", [])


def download_Sentinel_with_ids_names(ids, name, output_dir, access_token):
    download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({ids})/$value"
    download_headers = {
        "Authorization": f"Bearer {access_token}"
    }
    output_filename = f"{name}.zip"
    print("Downloading Sentinel-1 product...", end="", flush=True)
    stdout.flush()
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    partial_path = f"{output_path}.part"
    try:
        with requests.get(
            download_url, headers=download_headers, stream=True, timeout=300
        ) as response:
            response.raise_for_status()
            with open(partial_path, 'wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        os.replace(partial_path, output_path)
        print(" completed.")
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)

def process_snentinel_images(file, processed_path):
    target_resolution = 20
    terrain_flat_bool = False
    remove_therm_noise_bool = True

    geocode(
        infile=file,
        outdir=processed_path,
        spacing=int(target_resolution),
        polarizations=['VV','VH'],
        refarea='gamma0',
        t_srs=4326,
        scaling='linear',
        clean_edges=True,
        terrainFlattening=terrain_flat_bool,
        removeS1ThermalNoise=remove_therm_noise_bool,
        export_extra=['incidenceAngleFromEllipsoid', 'localIncidenceAngle'],
        demName='ACE30',
        nodataValueAtSea=False,
    )


def incidence_process(VV_VH_incidence_path):
    ds1 = gdal.Open(glob.glob(f"{VV_VH_incidence_path}/*VV_gamma0-elp.tif")[0])
    band1 = ds1.GetRasterBand(1).ReadAsArray().astype(float)

    ds2 = gdal.Open(glob.glob(f"{VV_VH_incidence_path}/*VH_gamma0-elp.tif")[0])
    band2 = ds2.GetRasterBand(1).ReadAsArray().astype(float)

    ds3 = gdal.Open(glob.glob(f"{VV_VH_incidence_path}/*incidenceAngleFromEllipsoid.tif")[0])
    band3 = ds3.GetRasterBand(1).ReadAsArray().astype(float)

    # Normalize backscatter using the ellipsoid incidence angle.
    cos_band = np.cos(np.deg2rad(band3))

    result1 = band1 / cos_band

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(f"{VV_VH_incidence_path}/Gamma0_VV.tif",
                           ds1.RasterXSize,
                           ds1.RasterYSize,
                           1,
                           gdal.GDT_Float32)

    out_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_ds.SetProjection(ds1.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result1)
    out_band.SetNoDataValue(np.nan)

    result2 = band2 / cos_band
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(f"{VV_VH_incidence_path}/Gamma0_VH.tif",
                           ds1.RasterXSize,
                           ds1.RasterYSize,
                           1,
                           gdal.GDT_Float32)

    out_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_ds.SetProjection(ds1.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result2)
    out_band.SetNoDataValue(np.nan)

    out_ds.FlushCache()
    out_ds = None
    ds1 = None
    ds2 = None

parser = argparse.ArgumentParser(
    description="Download and preprocess one Sentinel-1 product by exact name."
)
parser.add_argument("s1_product_name")
parser.add_argument("work_dir", help="Exact writable preprocessing directory.")
args = parser.parse_args()

S1name = args.s1_product_name.removesuffix(".SAFE")
workfolder = os.path.abspath(os.path.expanduser(args.work_dir))
username = os.environ.get("CDSE_USERNAME", "")
password = os.environ.get("CDSE_PASSWORD", "")

if not username or not password:
    print("Copernicus Data Space username and password are required.")
    sys.exit(1)

os.makedirs(workfolder, exist_ok=True)
products = search_sentinel_with_S1name(S1name)
if not products:
    raise RuntimeError(f"Sentinel-1 product not found: {S1name}.SAFE")
if len(products) > 1:
    raise RuntimeError(f"Multiple catalogue products matched exact name: {S1name}")

product = products[0]
started = datetime.datetime.now()
product_id = product["Id"]
name = product["Name"]
archive_path = os.path.join(workfolder, f"{name}.zip")
if not os.path.isfile(archive_path):
    access_token = log_in(username, password)
    download_Sentinel_with_ids_names(
        product_id, name, workfolder, access_token
    )

print("Preprocessing SAR imagery...", end="", flush=True)
process_snentinel_images(archive_path, workfolder)
incidence_process(workfolder)
print(f" completed in {datetime.datetime.now() - started}.")
