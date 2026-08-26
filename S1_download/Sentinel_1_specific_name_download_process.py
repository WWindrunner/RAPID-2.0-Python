import argparse
import datetime
import glob
import os
import sys
import zipfile
from sys import stdout

import numpy as np
import rasterio
import requests
from pyroSAR.snap.util import geocode
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

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
    def exactly_one(pattern, label):
        matches = sorted(glob.glob(os.path.join(VV_VH_incidence_path, pattern)))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {label}, found {len(matches)}"
            )
        return matches[0]

    vv_path = exactly_one("*VV_gamma0-elp.tif", "raw VV Gamma0 raster")
    vh_path = exactly_one("*VH_gamma0-elp.tif", "raw VH Gamma0 raster")
    angle_path = exactly_one(
        "*incidenceAngleFromEllipsoid.tif", "ellipsoid incidence-angle raster"
    )
    output_paths = {
        "VV": os.path.join(VV_VH_incidence_path, "Gamma0_VV.tif"),
        "VH": os.path.join(VV_VH_incidence_path, "Gamma0_VH.tif"),
    }
    temporary_paths = {
        key: f"{path}.partial.tif" for key, path in output_paths.items()
    }

    try:
        with (
            rasterio.open(vv_path) as vv,
            rasterio.open(vh_path) as vh,
            rasterio.open(angle_path) as angle,
        ):
            if (
                vv.count != 1
                or vh.count != 1
                or angle.count != 1
                or vv.crs is None
                or vh.crs is None
                or angle.crs is None
            ):
                raise ValueError(
                    "VV, VH, and incidence angle must be georeferenced "
                    "single-band rasters"
                )
            if (
                vv.width != vh.width
                or vv.height != vh.height
                or vv.crs != vh.crs
                or not vv.transform.almost_equals(vh.transform)
            ):
                raise ValueError("Raw VV and VH rasters do not use the same grid")

            profile = vv.profile.copy()
            profile.update(
                driver="GTiff", count=1, dtype="float32", nodata=-9999.0,
                compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
            )
            with (
                WarpedVRT(
                    angle,
                    crs=vv.crs,
                    transform=vv.transform,
                    width=vv.width,
                    height=vv.height,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                ) as aligned_angle,
                rasterio.open(temporary_paths["VV"], "w", **profile) as out_vv,
                rasterio.open(temporary_paths["VH"], "w", **profile) as out_vh,
            ):
                for _, window in vv.block_windows(1):
                    vv_values = vv.read(1, window=window, masked=True).astype(
                        np.float32
                    ).filled(np.nan)
                    vh_values = vh.read(1, window=window, masked=True).astype(
                        np.float32
                    ).filled(np.nan)
                    angle_values = aligned_angle.read(
                        1, window=window, masked=True
                    ).astype(np.float32).filled(np.nan)
                    cosine = np.cos(np.deg2rad(angle_values)).astype(np.float32)
                    valid = (
                        np.isfinite(vv_values)
                        & np.isfinite(vh_values)
                        & (vv_values >= 0)
                        & (vh_values >= 0)
                        & np.isfinite(angle_values)
                        & (angle_values > 0)
                        & (angle_values < 90)
                        & np.isfinite(cosine)
                        & (cosine > 0)
                    )
                    normalized_vv = np.full(vv_values.shape, -9999.0, np.float32)
                    normalized_vh = np.full(vh_values.shape, -9999.0, np.float32)
                    np.divide(vv_values, cosine, out=normalized_vv, where=valid)
                    np.divide(vh_values, cosine, out=normalized_vh, where=valid)
                    out_vv.write(normalized_vv, 1, window=window)
                    out_vh.write(normalized_vh, 1, window=window)

        for key, final_path in output_paths.items():
            os.replace(temporary_paths[key], final_path)
    finally:
        for temporary_path in temporary_paths.values():
            if os.path.isfile(temporary_path):
                os.remove(temporary_path)


def valid_single_band_raster(path):
    if not os.path.isfile(path):
        return False
    try:
        with rasterio.open(path) as src:
            return (
                src.count == 1
                and src.width > 0
                and src.height > 0
                and src.crs is not None
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def preprocessing_complete(workfolder):
    angle_paths = sorted(
        glob.glob(os.path.join(workfolder, "*incidenceAngleFromEllipsoid.tif"))
    )
    paths = [
        os.path.join(workfolder, "Gamma0_VV.tif"),
        os.path.join(workfolder, "Gamma0_VH.tif"),
        *angle_paths,
    ]
    return len(angle_paths) == 1 and all(valid_single_band_raster(p) for p in paths)


def remove_incomplete_derivatives(workfolder):
    patterns = (
        "Gamma0_VV.tif", "Gamma0_VH.tif", "*.partial.tif",
        "*VV_gamma0-elp.tif", "*VH_gamma0-elp.tif",
        "*incidenceAngleFromEllipsoid.tif", "*localIncidenceAngle.tif",
        "*_manifest.safe", "*_proc.xml",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(workfolder, pattern)):
            if os.path.isfile(path):
                os.remove(path)

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
if preprocessing_complete(workfolder):
    print("Sentinel-1 preprocessing outputs already complete; reusing them.")
    raise SystemExit(0)

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
remove_incomplete_derivatives(workfolder)
if os.path.isfile(archive_path) and not zipfile.is_zipfile(archive_path):
    print("Existing Sentinel-1 archive is incomplete; downloading it again.")
    os.remove(archive_path)
if not os.path.isfile(archive_path):
    access_token = log_in(username, password)
    download_Sentinel_with_ids_names(
        product_id, name, workfolder, access_token
    )

print("Preprocessing SAR imagery...", end="", flush=True)
process_snentinel_images(archive_path, workfolder)
incidence_process(workfolder)
if not preprocessing_complete(workfolder):
    raise RuntimeError("Sentinel-1 preprocessing did not produce complete outputs")
print(f" completed in {datetime.datetime.now() - started}.")
