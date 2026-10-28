"""Relabel RAPID flood pixels with the desert (S1 coherence) and snow (S2) masks.

  python apply_masks.py <Morph_final.tif> <out_prefix> [--desert MASK|wc60 --coh COH.tif]
                        [--snow SNOW.tif] [--snow-possible POSSIBLE.tif]

Desert changes flood pixels (classes 2-5) only; snow changes non-ocean water pixels
(permanent water 1 and flood 2-5). The LIA/shadow exclusion is untouched.

Desert: like RAPID (Load_Data.py), desert raster > 0 (nearest onto the grid)
AND ESA WorldCover class 60 (local tiles from --worldcover-dir or
RAPID_WORLDCOVER_DIR, else the 2020 v100 download); "wc60" uses class 60 alone as a
stand-in,
decided by S1 coherence alone (coh_loss.py output: baseline, event, loss):
    event > TH_LOW                  -> 0
    event <= TH_LOW, loss >= TH_D   -> flood class kept
    otherwise, incl. no coherence   -> 200 undetermined
Snow (snow_latest_clear.py output), decided by S2 alone and applied to permanent
water and to flood pixels the desert step did not set to 0:
    1-5   -> 100        (snow on a clear S2 view up to 4 days before S1)
    6-15  -> 100 + age  (106-115, less credible)
    200   -> 250        (every S2 view in the window was cloud / cloud shadow)
Ocean (Natural Earth 10m, as in RAPID's getOceanMask) is not water here: the snow
step never touches it. With --snow-possible (snow_possible.py output, ERA5 + DEM),
snow labels go only where snow is physically possible; a scene where it is possible
nowhere needs no S2 snow mask at all (--snow may then be left out).
"""
import argparse
import os
import urllib.request
import zipfile

import numpy as np
import rasterio
from osgeo import gdal

gdal.UseExceptions()
TH_LOW, TH_D = 0.5, 0.25
FLOOD = (2, 3, 4, 5)
PERMANENT = 1
UNDETERMINED, ICE, ICE_RECENT_DAYS, SNOW_CLOUD = 200, 100, 5, 250
OCEAN_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_ocean.zip"
OCEAN_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "rapid", "ne_10m_ocean")
WC_URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
          "v100/2020/map/ESA_WorldCover_10m_2020_v100_{}_Map.tif")

parser = argparse.ArgumentParser()
parser.add_argument("morph")
parser.add_argument("prefix")
parser.add_argument("--desert")
parser.add_argument("--coh")
parser.add_argument("--snow")
parser.add_argument("--ocean", help="ocean polygons; default Natural Earth 10m, "
                    "downloaded once to ~/.cache/rapid")
parser.add_argument("--snow-possible")
parser.add_argument("--worldcover-dir", default=os.environ.get("RAPID_WORLDCOVER_DIR"),
                    help="local ESA_WorldCover_10m_*_<tile>_Map.tif (default "
                         "RAPID_WORLDCOVER_DIR, as AncillaryData.py); else downloaded")
args = parser.parse_args()
assert args.desert or args.snow or args.snow_possible, "give --desert and/or --snow"
assert not args.desert or args.coh, "--desert needs --coh"

with rasterio.open(args.morph) as src:
    res = src.read(1)
    profile = src.profile.copy()
    rows, cols = src.height, src.width
    crs_wkt, (w, s, e, n) = src.crs.to_wkt(), tuple(src.bounds)
flood = np.isin(res, FLOOD)
out = res.copy()
change = np.zeros_like(res)  # see codes below
print(f"RAPID flood px: {flood.sum()}")


def desert_raster(path):
    ds = gdal.Warp("", path, format="MEM", dstSRS=crs_wkt, outputBounds=(w, s, e, n),
                   width=cols, height=rows, resampleAlg="near", multithread=True)
    arr = ds.ReadAsArray()
    return (arr > 0) & (arr != 255)


def ocean_shp():
    if args.ocean:
        return args.ocean
    shp = os.path.join(OCEAN_CACHE, "ne_10m_ocean.shp")
    if not os.path.exists(shp):
        os.makedirs(OCEAN_CACHE, exist_ok=True)
        zip_path = os.path.join(OCEAN_CACHE, "ne_10m_ocean.zip")
        urllib.request.urlretrieve(OCEAN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(OCEAN_CACHE)
    return shp


def ocean():
    ds = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Byte)
    ds.SetProjection(crs_wkt)
    ds.SetGeoTransform(profile["transform"].to_gdal())
    gdal.Rasterize(ds, ocean_shp(), burnValues=[1], allTouched=False)
    return ds.ReadAsArray() == 1


def worldcover60():
    tiles = [f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
             for lat in range(int(np.floor(s / 3) * 3), int(np.ceil(n / 3) * 3), 3)
             for lon in range(int(np.floor(w / 3) * 3), int(np.ceil(e / 3) * 3), 3)]
    if args.worldcover_dir:
        import glob
        paths = [sorted(glob.glob(os.path.join(args.worldcover_dir,
                                               f"ESA_WorldCover_10m_*_{t}_Map.tif")))
                 for t in tiles]
        paths = [found[-1] for found in paths if found]
    else:
        paths = [WC_URL.format(t) for t in tiles]
    print("WorldCover tiles:", paths)
    ds = gdal.Warp("", paths, format="MEM", dstSRS=crs_wkt,
                   outputBounds=(w, s, e, n), width=cols, height=rows,
                   resampleAlg="mode", srcNodata=0, dstNodata=0, multithread=True)
    return ds.ReadAsArray() == 60


mask_profile = profile.copy()
mask_profile.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256)

if args.desert:
    desert = worldcover60()
    print(f"WorldCover 60 px: {desert.sum()}")
    if args.desert != "wc60":
        desert &= desert_raster(args.desert)
        print(f"desert raster AND WorldCover 60 px: {desert.sum()}")
    with rasterio.open(f"{args.prefix}_desert.tif", "w", **mask_profile) as dst:
        # 1 desert, 0 not, 255 outside the RAPID footprint
        dst.write(np.where(res == 255, 255, desert).astype(np.uint8), 1)
    with rasterio.open(args.coh) as src:
        event, loss = src.read(2), src.read(3)
    cand = flood & desert
    has = np.isfinite(event) & np.isfinite(loss)
    dry = cand & has & (event > TH_LOW)
    keep = cand & has & (event <= TH_LOW) & (loss >= TH_D)
    undet = cand & ~dry & ~keep
    out[dry], out[undet] = 0, UNDETERMINED
    change[keep], change[dry], change[undet] = 1, 2, 3
    print(f"desert: {cand.sum()} flood px in mask -> 0: {dry.sum()}, kept: {keep.sum()}, "
          f"200: {undet.sum()} (no coherence: {(undet & ~has).sum()})")
    corrected = np.where(res == 255, 255, desert).astype(np.uint8)
    corrected[keep], corrected[undet] = 0, UNDETERMINED
    with rasterio.open(f"{args.prefix}_desert_corrected.tif", "w", **mask_profile) as dst:
        # 1 desert, 0 not desert (incl. flood confirmed by coherence), 200 undetermined, 255 nodata
        dst.write(corrected, 1)
    del corrected
    flood &= ~dry
    del desert, event, loss, cand, has, dry, keep, undet

possible = None
if args.snow_possible:
    with rasterio.open(args.snow_possible) as src:
        possible = src.read(1) == 1
    print(f"snow possible px: {possible.sum()}")
    if not possible.any():
        print("snow impossible everywhere: snow step skipped")
        args.snow = None

if args.snow:
    with rasterio.open(args.snow) as src:
        snow = src.read(1)
    sea = ocean()
    print(f"ocean px (left alone): {sea.sum()}")
    flood &= ~sea
    water = flood | ((res == PERMANENT) & ~sea)
    if possible is not None:
        water &= possible
    recent = water & (snow >= 1) & (snow <= ICE_RECENT_DAYS)
    older = water & (snow > ICE_RECENT_DAYS) & (snow <= 15)
    cloud = water & (snow == 200) & (out != UNDETERMINED)  # desert 200 wins over 250
    out[recent] = ICE
    out[older] = ICE + snow[older]
    out[cloud] = SNOW_CLOUD
    change[recent], change[older], change[cloud] = 4, 5, 6
    print(f"snow: -> 100: {recent.sum()}, -> 106-115: {older.sum()}, -> 250: {cloud.sum()}")
    for d in range(ICE_RECENT_DAYS + 1, 16):
        if (k := (older & (snow == d)).sum()):
            print(f"  -> {ICE + d}: {k}")

change[res == 255] = 255
profile.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256)
with rasterio.open(f"{args.prefix}_Morph_masked.tif", "w", **profile) as dst:
    dst.write(out, 1)
with rasterio.open(f"{args.prefix}_mask_change.tif", "w", **profile) as dst:
    # 0 unchanged; desert: 1 kept flood, 2 -> 0, 3 -> 200; snow: 4 -> 100, 5 -> 106-115, 6 -> 250
    dst.write(change, 1)
u, c = np.unique(out, return_counts=True)
print("output classes:", dict(zip(u.tolist(), c.tolist())))
print(f"-> {args.prefix}_Morph_masked.tif, {args.prefix}_mask_change.tif")
