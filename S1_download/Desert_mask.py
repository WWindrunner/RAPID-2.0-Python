import argparse
import os
import sys
import time

from osgeo import gdal


gdal.UseExceptions()
start_time = time.perf_counter()


parser = argparse.ArgumentParser(
    description="Clip a fixed global desert mask to a read-only SAR reference."
)
parser.add_argument("reference_raster")
parser.add_argument("desert_mask_vrt")
parser.add_argument("output_path")
args = parser.parse_args()

reference_path = os.path.abspath(os.path.expanduser(args.reference_raster))
desert_mask_vrt = os.path.abspath(os.path.expanduser(args.desert_mask_vrt))
output_path = os.path.abspath(os.path.expanduser(args.output_path))
if not os.path.isfile(reference_path):
    raise FileNotFoundError(f"Reference raster not found: {reference_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)


def create_nodata_desert_raster(reference_ds):
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(
        output_path,
        reference_ds.RasterXSize,
        reference_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    if output_ds is None:
        raise RuntimeError(f"Failed to create empty desert mask: {output_path}")
    output_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    output_ds.SetProjection(reference_ds.GetProjection())
    band = output_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.Fill(-9999)
    band.FlushCache()
    output_ds = None


# The reference raster defines the output grid and fallback raster.
reference_ds = gdal.Open(reference_path)
if reference_ds is None:
    raise RuntimeError(f"Failed to open reference raster: {reference_path}")

failed = False
try:
    if not os.path.isfile(desert_mask_vrt):
        raise FileNotFoundError(f"Desert mask VRT not found: {desert_mask_vrt}")

    reference_gt = reference_ds.GetGeoTransform()
    reference_projection = reference_ds.GetProjection()
    reference_cols = reference_ds.RasterXSize
    reference_rows = reference_ds.RasterYSize

    if not reference_projection:
        raise ValueError(f"Reference raster has no CRS: {reference_path}")

    if reference_gt[2] != 0 or reference_gt[4] != 0:
        raise ValueError("Rotated reference rasters are not supported")

    x_end = reference_gt[0] + reference_cols * reference_gt[1]
    y_end = reference_gt[3] + reference_rows * reference_gt[5]
    reference_bounds = [
        min(reference_gt[0], x_end),
        min(reference_gt[3], y_end),
        max(reference_gt[0], x_end),
        max(reference_gt[3], y_end),
    ]

    desert_ds = gdal.Open(desert_mask_vrt)
    if desert_ds is None:
        raise RuntimeError(f"Failed to open desert mask: {desert_mask_vrt}")
    if desert_ds.RasterCount != 1:
        raise ValueError(
            "Desert mask must contain exactly one band, "
            f"found {desert_ds.RasterCount}"
        )

    source_nodata = desert_ds.GetRasterBand(1).GetNoDataValue()
    warp_kwargs = {
        "format": "GTiff",
        "dstSRS": reference_projection,
        "outputBounds": reference_bounds,
        "width": reference_cols,
        "height": reference_rows,
        "resampleAlg": gdal.GRA_NearestNeighbour,
        "multithread": True,
        "creationOptions": ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    }
    if source_nodata is not None:
        warp_kwargs["srcNodata"] = source_nodata
        warp_kwargs["dstNodata"] = source_nodata

    # print(f"Reference raster: {reference_path}")
    # print(f"Desert mask: {desert_mask_vrt}")
    # print(f"Output raster: {output_path}")

    output_ds = gdal.Warp(
        output_path,
        desert_ds,
        options=gdal.WarpOptions(**warp_kwargs),
    )
    if output_ds is None:
        raise RuntimeError(f"Failed to create desert mask: {output_path}")

    output_ds.FlushCache()
    output_ds = None
    desert_ds = None
    elapsed = time.perf_counter() - start_time
    print(f"Desert mask completed in {elapsed:.2f} seconds.")
except Exception:
    failed = True
    print(
        "ERROR: Desert mask calculation failed. Creating an empty nodata mask ",
        "for diagnostics.",
        file=sys.stderr,
    )
    create_nodata_desert_raster(reference_ds)
    print("Empty diagnostic desert mask created.")
finally:
    reference_ds = None

if failed:
    sys.exit(1)
