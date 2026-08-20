import os
import numpy as np
from utils import *
from AncillaryData import *
import dask.array as da
import rasterio
from scipy import ndimage


def readfileio(file, chunks=(1, 512, 512), label="raster"):
    """Read one single-band GeoTIFF as a Dask array."""
    with rasterio.open(file) as src:
        if src.count != 1:
            raise ValueError(
                f"{label} must be a single-band raster, but {file} has "
                f"{src.count} bands."
            )
        img = da.from_array(src.read(1), chunks=chunks)
    return img


def readhdf(file, chunks=(1, 512, 512)):
    # Not use anymore, 05012023
    # f = h5py.File(file)
    f = None
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    d = f[a_group_key]
    x = da.from_array(d, chunks=chunks)
    return x


def validate_sar_grids(vv_path, vh_path):
    """Require the two SAR polarizations to use exactly the same grid."""
    with rasterio.open(vv_path) as vv, rasterio.open(vh_path) as vh:
        if vv.count != 1 or vh.count != 1:
            raise ValueError("VV and VH inputs must both be single-band rasters.")

        mismatches = []
        if (vv.width, vv.height) != (vh.width, vh.height):
            mismatches.append("dimensions")
        if vv.crs != vh.crs:
            mismatches.append("CRS")
        if not vv.transform.almost_equals(vh.transform):
            mismatches.append("transform")

        if mismatches:
            raise ValueError(
                "VV and VH rasters are not aligned; mismatched "
                + ", ".join(mismatches)
                + "."
            )


def get_src(filepath, args):
    # Open the GeoTIFF file using rasterio
    with rasterio.open(filepath) as src:
        args.refmeta = src.meta.copy()
        args.transform = src.transform
        args.crs_wkt = src.crs.to_wkt() if src.crs else None
        args.height = src.height
        args.width = src.width
        args.nodata = src.nodata
        args.count = src.count

    args.ref_src = None
    return args


def set_nan(ds, args):
    ds.img_lp = da.where(ds.img_lp == args.nodata, 0, ds.img_lp)
    ds.img_cp = da.where(ds.img_cp == args.nodata, 0, ds.img_cp)
    print('set nan for BRD noise by erosion with 5 pixels')
    valid_mask = da.where(ds.img_lp > 0, 1, 0)
    valid_BRD_mask = ndimage.binary_erosion(valid_mask, iterations=5)
    ds.img_lp = da.where(valid_BRD_mask == 1, ds.img_lp, 0)
    ds.img_cp = da.where(valid_BRD_mask == 1, ds.img_cp, 0)
    del valid_mask, valid_BRD_mask
    return ds


def normalize_inc(ds):
    """
    Normalize linear Gamma0 power using ellipsoid incidence angle in degrees.

    Gamma0_normalized = Gamma0 / cos(theta_ellipsoid)
    Invalid angle pixels remain zero and therefore stay outside classification.
    """
    print("Normalizing Gamma0 with ellipsoid incidence angle (degrees)")
    # Keep scene-sized arrays in float32. Statistical/GMM workers explicitly
    # promote their much smaller windows to float64 before fitting.
    angle = ds.inc.astype(np.float32)
    img_lp = ds.img_lp.astype(np.float32)
    img_cp = ds.img_cp.astype(np.float32)
    cos_angle = da.cos(da.deg2rad(angle)).astype(np.float32)
    # Zero is treated as invalid because aligned incidence-angle rasters use
    # zero outside their valid source footprint.
    valid_angle = da.isfinite(angle) & (angle > 0.0) & (angle < 90.0)
    valid_angle = valid_angle & da.isfinite(cos_angle) & (cos_angle > 0.0)

    ds.img_lp = da.where(
        valid_angle & da.isfinite(img_lp) & (img_lp > 0),
        img_lp / cos_angle,
        np.float32(0.0),
    ).astype(np.float32)
    ds.img_cp = da.where(
        valid_angle & da.isfinite(img_cp) & (img_cp > 0),
        img_cp / cos_angle,
        np.float32(0.0),
    ).astype(np.float32)
    return ds


# Convert the ESA LCC value to GP LCC
def ESALCC2GPLCC(arr):
    '''
    Convert the ESA LCC value to GP LCC, 05/08/2023
    So we don't have to change the predefined LCC value
    '''
    new_arr = np.zeros_like(arr)
    # Define the original and target value ranges
    origin_LCC = [40, 10, 30, 20, 90, 80, 100, 50, 60, 70, 95]
    Valuerange_LCC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 20]
    for i in range(len(origin_LCC)):
        new_arr[(arr == origin_LCC[i])] = Valuerange_LCC[i]
    return new_arr


@tim
def Load_Data(args):
    filepath = args.filepath
    wotif = args.wotif
    LCCtif = None
    oceantif = None

    if len(filepath) != 2:
        raise ValueError(
            "Each scene must contain exactly two local single-band SAR files: "
            "[VV, VH]."
        )
    inc_angle_path = getattr(args, "inc_angle_path", None)
    input_paths = [("VV", filepath[0]), ("VH", filepath[1])]
    if inc_angle_path:
        input_paths.append(("incidence angle", inc_angle_path))
    for label, path in input_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} input does not exist: {path}")
    if inc_angle_path:
        with rasterio.open(inc_angle_path) as inc_src:
            if inc_src.count != 1:
                raise ValueError(
                    "Incidence angle must be a single-band raster, but "
                    f"{inc_angle_path} has {inc_src.count} bands."
                )

    def ancillary(datamode='mem'):
        return AncillaryData(
            ref=infilepath[0],
            chunk_size=args.chunks,
            datamode=datamode,
            scratch_dir=getattr(args, "scratch_dir", None),
        )

    class c:
        pass

    ds = c()
    args.datamode = 'mem'
    infilepath = filepath
    validate_sar_grids(filepath[0], filepath[1])
    args = get_src(filepath[0], args)
    args.tempfiles = []
    ds.img_lp = readfileio(filepath[0], args.chunks, label="VV")
    ds.img_cp = readfileio(filepath[1], args.chunks, label="VH")

    # Output are binary mask so just one band
    args.refmeta.update(count=1)
    args.dual_pol = True
    #
    ds = set_nan(ds, args)
    if inc_angle_path:
        ds.inc = ancillary().getLocalRaster(
            inc_angle_path, resampling="bilinear"
        ).squeeze()

        if ds.inc.shape != ds.img_lp.shape:
            raise ValueError(
                "Aligned incidence-angle raster shape does not match the SAR grid: "
                f"{ds.inc.shape} != {ds.img_lp.shape}."
            )
        if hasattr(ds.inc, "rechunk"):
            ds.inc = ds.inc.rechunk(ds.img_lp.chunks)
        ds = normalize_inc(ds)
    else:
        print(
            "[Load_Data] no inc_angle_path provided; assuming the input "
            "Gamma0 power rasters are already incidence-angle normalized.",
            flush=True,
        )

    # water occurrence file
    if wotif == None:
        if args.datamode == 'mem':
            setattr(ds, 'water_occurrence',
                    ancillary().getWO().squeeze())
            # args.tempfiles = None
        else:
            tempfileWO = ancillary(datamode=args.datamode).getWO()
            setattr(ds, 'water_occurrence', readhdf(tempfileWO.name, args.chunks).squeeze())
            args.tempfiles.append(tempfileWO)
    else:
        setattr(ds, 'water_occurrence', readfileio(wotif, args.chunks))
        # args.tempfiles = None

    # align chunks before combining arrays, to avoid repeated rechunk-merge graphs later
    if hasattr(ds.water_occurrence, "rechunk"):
        ds.water_occurrence = ds.water_occurrence.rechunk(ds.img_lp.chunks)

    if hasattr(ds.img_cp, "rechunk") and ds.img_cp.chunks != ds.img_lp.chunks:
        ds.img_cp = ds.img_cp.rechunk(ds.img_lp.chunks)

    mask_wo = ds.water_occurrence > 100
    zero_lp = da.zeros_like(ds.img_lp)
    zero_cp = da.zeros_like(ds.img_cp)

    ds.img_lp = da.where(mask_wo, zero_lp, ds.img_lp)
    ds.img_cp = da.where(mask_wo, zero_cp, ds.img_cp)

    # land cover file
    if LCCtif == None:
        if args.datamode == 'mem':
            setattr(ds, 'land_cover', ancillary().getLCC().squeeze())
        else:
            tempfileWO = ancillary(datamode=args.datamode).getLCC()
            setattr(ds, 'land_cover', readhdf(tempfileWO.name, args.chunks).squeeze())
            args.tempfiles.append(tempfileWO)
    else:
        setattr(ds, 'land_cover', readfileio(wotif, args.chunks))
        # args.tempfiles = None
    # Convert ESA LCC to GP LCC
    # ds.land_cover = ESALCC2GPLCC(ds.land_cover)

    # ocean mask file
    if args.datamode == 'mem':
        setattr(ds, 'ocean_mask', ancillary().getOceanMask().squeeze())
    else:
        tempfileOM = ancillary(datamode=args.datamode).getOceanMask()
        setattr(ds, 'ocean_mask', readhdf(tempfileOM.name, args.chunks).squeeze())
        args.tempfiles.append(tempfileOM)

    # align chunks
    if hasattr(ds.ocean_mask, "rechunk"):
        ds.ocean_mask = ds.ocean_mask.rechunk(ds.img_lp.chunks)

    # save preprocessed / clipped ocean mask
    #save_ancillary_tif(
    #    ds.ocean_mask,
    #    args,
    #    out_name="ocean_mask.tif",
    #    dtype="uint8",
    #    nodata=255
    #)

    # Local incidence angle exclusion data
    ds.lia = None
    ds.desert_mask = None

    lia_path = getattr(args, "LIA_path", None)
    if lia_path is not None:
        print(f"[Load_Data] reading binary LIA exclusion mask: {lia_path}", flush=True)
        if args.datamode == 'mem':
            ds.lia = ancillary().getLocalRaster(lia_path, resampling="nearest").squeeze()
        else:
            tempfileLIA = ancillary(datamode=args.datamode).getLocalRaster(lia_path, resampling="nearest")
            ds.lia = readhdf(tempfileLIA.name, args.chunks).squeeze()
            args.tempfiles.append(tempfileLIA)

        if hasattr(ds.lia, "rechunk"):
            ds.lia = ds.lia.rechunk(ds.img_lp.chunks)

    else:
        print("[Load_Data] LIA exclusion disabled", flush=True)

    # Snow/ice mask
    ice_mask_path = getattr(args, "ice_mask_path", None)
    if ice_mask_path is not None:
        print(f"[Load_Data] reading ice/snow mask: {ice_mask_path}", flush=True)

        if args.datamode == 'mem':
            setattr(
                ds,
                'ice_mask',
                ancillary().getLocalRaster(
                    ice_mask_path,
                    resampling="nearest"
                ).squeeze()
            )
        else:
            tempfileIce = (
                ancillary(datamode=args.datamode).getLocalRaster(
                    ice_mask_path,
                    resampling="nearest"
                )
            )

            setattr(
                ds,
                'ice_mask',
                readhdf(tempfileIce.name, args.chunks).squeeze()
            )

            args.tempfiles.append(tempfileIce)

        if hasattr(ds.ice_mask, "rechunk"):
            ds.ice_mask = ds.ice_mask.rechunk(ds.img_lp.chunks)

    # Desert mask
    desert_mask_path = getattr(args, "desert_mask_path", None)
    if desert_mask_path is not None:
        print(f"[Load_Data] reading desert mask: {desert_mask_path}", flush=True)

        if args.datamode == 'mem':
            ds.desert_mask = ancillary().getLocalRaster(
                desert_mask_path, resampling="nearest"
            ).squeeze()
        else:
            tempfileDesert = ancillary(datamode=args.datamode).getLocalRaster(
                desert_mask_path, resampling="nearest"
            )
            ds.desert_mask = readhdf(tempfileDesert.name, args.chunks).squeeze()
            args.tempfiles.append(tempfileDesert)

        if hasattr(ds.desert_mask, "rechunk"):
            ds.desert_mask = ds.desert_mask.rechunk(ds.img_lp.chunks)

    if hasattr(ds, "inc"):
        del ds.inc

    # print("img_lp chunks:", getattr(ds.img_lp, "chunks", None))
    # print("img_cp chunks:", getattr(ds.img_cp, "chunks", None))
    # print("water_occurrence chunks:", getattr(ds.water_occurrence, "chunks", None))
    return args, ds


def save_ancillary_tif(arr, args, out_name, dtype="uint8", nodata=255):
    """
    Save a 2D ancillary raster already aligned to SAR grid.

    arr can be numpy array or dask array.
    """

    os.makedirs(args.dirOut, exist_ok=True)
    out_path = os.path.join(args.dirOut, out_name)

    if isinstance(arr, da.Array):
        arr_out = arr.compute()
    else:
        arr_out = arr

    arr_out = np.asarray(arr_out).squeeze()

    # ocean mask / boolean mask: save as 0/1
    if arr_out.dtype == bool:
        arr_out = arr_out.astype(np.uint8)
    else:
        arr_out = arr_out.astype(dtype)

    meta = args.refmeta.copy()
    meta.update(
        driver="GTiff",
        count=1,
        dtype=arr_out.dtype,
        nodata=nodata,
        compress="lzw"
    )

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr_out, 1)

    print(f"[Load_Data] saved ancillary raster: {out_path}", flush=True)
    return out_path

