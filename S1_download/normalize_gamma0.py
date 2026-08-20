"""Normalize linear Gamma0 rasters with ellipsoid incidence angle."""

import argparse
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


OUTPUT_NODATA = -9999.0


def same_grid(left, right):
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and left.transform.almost_equals(right.transform)
    )


def normalize_pair(vv_path, vh_path, angle_path, output_dir):
    """Write VV/VH divided by cos(ellipsoid incidence angle)."""
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {
        "VV": os.path.join(output_dir, "Gamma0_VV.tif"),
        "VH": os.path.join(output_dir, "Gamma0_VH.tif"),
    }

    with (
        rasterio.open(vv_path) as vv,
        rasterio.open(vh_path) as vh,
        rasterio.open(angle_path) as angle,
    ):
        for label, src in (("VV", vv), ("VH", vh), ("incidence angle", angle)):
            if src.count != 1:
                raise ValueError(f"{label} raster must contain exactly one band")
            if src.crs is None:
                raise ValueError(f"{label} raster has no CRS")

        if not same_grid(vv, vh):
            raise ValueError("VV and VH rasters must use exactly the same grid")

        profile = vv.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=OUTPUT_NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

        angle_vrt_options = {
            "crs": vv.crs,
            "transform": vv.transform,
            "width": vv.width,
            "height": vv.height,
            "resampling": Resampling.bilinear,
            "nodata": np.nan,
        }

        temporary_paths = {
            pol: f"{path}.partial.tif" for pol, path in output_paths.items()
        }
        try:
            with (
                WarpedVRT(angle, **angle_vrt_options) as aligned_angle,
                rasterio.open(temporary_paths["VV"], "w", **profile) as out_vv,
                rasterio.open(temporary_paths["VH"], "w", **profile) as out_vh,
            ):
                for _, window in vv.block_windows(1):
                    vv_data = vv.read(1, window=window, masked=True).astype(np.float32)
                    vh_data = vh.read(1, window=window, masked=True).astype(np.float32)
                    angle_data = aligned_angle.read(
                        1, window=window, masked=True
                    ).astype(np.float32)

                    vv_values = vv_data.filled(np.nan)
                    vh_values = vh_data.filled(np.nan)
                    angle_values = angle_data.filled(np.nan)
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

                    normalized_vv = np.full(vv_values.shape, OUTPUT_NODATA, np.float32)
                    normalized_vh = np.full(vh_values.shape, OUTPUT_NODATA, np.float32)
                    np.divide(vv_values, cosine, out=normalized_vv, where=valid)
                    np.divide(vh_values, cosine, out=normalized_vh, where=valid)
                    out_vv.write(normalized_vv, 1, window=window)
                    out_vh.write(normalized_vh, 1, window=window)

            for pol, final_path in output_paths.items():
                os.replace(temporary_paths[pol], final_path)
        finally:
            for temporary_path in temporary_paths.values():
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize local linear Gamma0 VV/VH rasters as "
            "Gamma0 / cos(ellipsoid incidence angle)."
        )
    )
    parser.add_argument("vv_path")
    parser.add_argument("vh_path")
    parser.add_argument("incidence_angle_path")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    try:
        outputs = normalize_pair(
            args.vv_path,
            args.vh_path,
            args.incidence_angle_path,
            args.output_dir,
        )
    except Exception as exc:
        print(f"Gamma0 normalization failed: {exc}", file=sys.stderr)
        return 1

    print(f"Normalized VV: {outputs['VV']}")
    print(f"Normalized VH: {outputs['VH']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
