"""Relabel RAPID flood pixels with the desert (S1 coherence) and snow (S2) masks,
then remove coherent flood patches (S1 coherence).

  python apply_masks.py <Morph_final.tif> <out_prefix> [--desert MASK|wc60 --coh COH.tif]
                        [--snow SNOW.tif] [--snow-possible POSSIBLE.tif]
                        [--no-coh-patch] [--coh-patch-th 0.75] [--coh-patch-min-km2 0.1]
                        [--coh-patch-min-cover 0.5] [--coh-patch-pct 25]
                        [--coh-patch-classes 2,3,4,5]

Desert changes flood pixels (classes 2-5) only; snow changes non-ocean water pixels
(permanent water 1 and flood 2-5). The LIA/shadow exclusion is untouched.

Desert: like RAPID (Load_Data.py), desert raster > 0 (nearest onto the grid)
AND ESA WorldCover class 60 (local tiles from --worldcover-dir or
RAPID_WORLDCOVER_DIR, missing tiles from the 2021 v200 download; else the 2020 v100
download); "wc60" uses class 60 alone as a
stand-in,
decided by S1 coherence alone (coh_loss.py output: baseline, event, loss):
    event > TH_LOW                  -> 0
    event <= TH_LOW, loss >= TH_D   -> flood class kept
    otherwise, incl. no coherence   -> 200 undetermined
Snow (snow_latest_clear.py output: band 1 snow / age, band 2 age of the latest clear
view), decided by S2. Only a clear S2 view 1-5 days back (1 = S1 day) is affirmative:
    snow on it                -> 100   (permanent water and flood)
    no snow on it             -> flood label kept
Older or missing evidence, on flood pixels only (classes 2-5 the desert step did not set
to 0) and only where --snow-possible says snow or ice is possible (GFS / ERA5 + DEM):
    snow, view 6-15 days old  -> 106   (affirmative snow: snow seen, view may be outdated)
    no snow, view 6-15 old    -> 116   (plausible snow: no snow seen, snow may have fallen since)
And on permanent water and flood:
    200   -> 250        (every S2 view in the window was cloud / cloud shadow)
Ocean (Natural Earth 10m, as in RAPID's getOceanMask) is not water here: the snow
step never touches it. With --snow-possible (snow_possible.py output: 2 m
temperature on the DEM, model snow depth and lake / sea ice at the S1 time),
snow labels go only where snow or ice is physically possible; a scene where it is possible
nowhere needs no S2 snow mask at all (--snow may then be left out).
Coherent patches (after snow; whenever --coh is given, unless --no-coh-patch): pixels
of --coh-patch-classes (default flood 2-5; desert-undetermined 200 is not removed) are grouped into
8-connected patches; snow / ice / cloud labels (100, 106, 116, 250), permanent water and
non-water separate patches and are never changed. A patch is set to 0 (dry) only if
    percentile --coh-patch-pct (default 25) of its event coherence > --coh-patch-th (default 0.75)
    AND at least --coh-patch-min-cover of its pixels have coherence
    AND its area >= --coh-patch-min-km2 (default 0.1; coherence is coarser than the RAPID grid).
Patches without enough coherence are kept. Strict by design: it only deletes.
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
EARLY_SNOW, EARLY_CLEAR, WINDOW = 106, 116, 15
OCEAN_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_ocean.zip"
OCEAN_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "rapid", "ne_10m_ocean")
WC_URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
          "v100/2020/map/ESA_WorldCover_10m_2020_v100_{}_Map.tif")
WC_URL_2021 = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
               "v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif")

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
parser.add_argument("--no-coh-patch", action="store_true",
                    help="skip the coherent-patch rule (it runs whenever --coh is given)")
parser.add_argument("--coh-patch-th", type=float, default=0.75,
                    help="remove a patch whose event-coherence percentile exceeds this (default 0.75)")
parser.add_argument("--coh-patch-pct", type=float, default=25,
                    help="percentile of the patch's event coherence (default 25)")
parser.add_argument("--coh-patch-min-km2", type=float, default=0.1,
                    help="only patches at least this large are removed (default 0.1 km2)")
parser.add_argument("--post-compactness", default="0,1.2,1.6,2.0,2.4",
                    help="compactness thresholds of classes 1-5 for the whole-object check after the "
                         "coherent-patch rule (default as RAPID morph); only classes 2-5 are removed")
parser.add_argument("--no-post-compactness", action="store_true",
                    help="skip the compactness check after the coherent-patch rule")
parser.add_argument("--coh-patch-min-cover", type=float, default=0.5,
                    help="only patches with coherence on at least this fraction of their "
                         "pixels are removed (default 0.5)")
parser.add_argument("--coh-patch-classes", default="2,3,4,5",
                    help="classes grouped into patches (default flood 2,3,4,5; 200 is never removed)")
args = parser.parse_args()
assert args.desert or args.snow or args.snow_possible or args.coh, \
    "give --desert and/or --snow and/or --coh"
assert not args.desert or args.coh, "--desert needs --coh"
assert 0 <= args.coh_patch_pct <= 100, "--coh-patch-pct must be 0-100"
assert 0 <= args.coh_patch_min_cover <= 1, "--coh-patch-min-cover must be 0-1"
COH_PATCH_CLASSES = tuple(int(c) for c in args.coh_patch_classes.split(","))
assert not set(COH_PATCH_CLASSES) & {PERMANENT, ICE, SNOW_CLOUD, 255, 0} \
    and not set(COH_PATCH_CLASSES) & {EARLY_SNOW, EARLY_CLEAR}, \
    "--coh-patch-classes may not include non-water, permanent water, snow/ice/cloud or nodata"

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
        paths = []
        for t in tiles:
            found = sorted(glob.glob(os.path.join(args.worldcover_dir,
                                                  f"ESA_WorldCover_10m_*_{t}_Map.tif")))
            # not local: same 2021 v200 tile from ESA
            paths.append(found[-1] if found else WC_URL_2021.format(t))
    else:
        paths = [WC_URL.format(t) for t in tiles]
    # open ocean has no tile at all
    paths = [p for p in paths if gdal.VSIStatL(p) is not None]
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
        age = src.read(2) if src.count >= 2 else None
    if age is None:
        print("WARNING: snow file has no age band (older snow_latest_clear.py): "
              "116 too-early clearance cannot be set; rerun snow_latest_clear.py")
    sea = ocean()
    print(f"ocean px (left alone): {sea.sum()}")
    flood &= ~sea
    water = flood | ((res == PERMANENT) & ~sea)
    if possible is not None:
        water &= possible
    recent = water & (snow >= 1) & (snow <= ICE_RECENT_DAYS)
    cloud = water & (snow == 200) & (out != UNDETERMINED)  # desert 200 wins over 250
    # 106 / 116 only on flood where snow or ice is physically possible
    fl = flood & possible if possible is not None else np.zeros_like(flood)
    if possible is None:
        print("no --snow-possible: 106 / 116 not set")
    early_snow = fl & (snow > ICE_RECENT_DAYS) & (snow <= WINDOW)
    early_clear = (fl & (snow == 0) & (age > ICE_RECENT_DAYS) & (age <= WINDOW)
                   if age is not None else np.zeros_like(flood))
    out[recent] = ICE
    out[early_snow] = EARLY_SNOW
    out[early_clear] = EARLY_CLEAR
    out[cloud] = SNOW_CLOUD
    change[recent], change[early_snow], change[cloud], change[early_clear] = 4, 5, 6, 7
    print(f"snow: -> 100: {recent.sum()}, -> 106: {early_snow.sum()}, "
          f"-> 116: {early_clear.sum()}, -> 250: {cloud.sum()}")


def patch_area_km2(labels, n_patches):
    """Area of every labelled patch, from the pixel size (geographic CRS: per row)."""
    t = profile["transform"]
    if rasterio.crs.CRS.from_wkt(crs_wkt).is_geographic:
        lat = t.f + t.e * (np.arange(rows) + 0.5)
        row_km2 = abs(t.a * 111.32 * np.cos(np.radians(lat))) * abs(t.e * 110.57)
    else:
        row_km2 = np.full(rows, abs(t.a * t.e) / 1e6)
    area = np.zeros(n_patches + 1)
    for r0 in range(0, rows, 2048):
        block = labels[r0:r0 + 2048]
        weights = np.broadcast_to(row_km2[r0:r0 + 2048, None], block.shape)
        area += np.bincount(block.ravel(), weights=weights.ravel(), minlength=n_patches + 1)
    return area


if args.coh and not args.no_coh_patch:
    from scipy import ndimage
    with rasterio.open(args.coh) as src:
        event = src.read(2)
    cand = np.isin(out, COH_PATCH_CLASSES)  # after desert and snow: snow labels split patches
    labels, n_patches = ndimage.label(cand, structure=np.ones((3, 3), bool))
    del cand
    size = np.bincount(labels.ravel(), minlength=n_patches + 1)
    has = (labels > 0) & np.isfinite(event)
    lab_v, val = labels[has], event[has]
    del has, event
    order = np.lexsort((val, lab_v))
    lab_v, val = lab_v[order], val[order]
    del order
    n_coh = np.bincount(lab_v, minlength=n_patches + 1)
    stat = np.full(n_patches + 1, np.nan, np.float32)
    if lab_v.size:
        starts = np.r_[0, np.flatnonzero(np.diff(lab_v)) + 1]
        count = np.diff(np.r_[starts, lab_v.size])
        ids = lab_v[starts]
        if args.coh_patch_pct == 50:
            stat[ids] = 0.5 * (val[starts + (count - 1) // 2] + val[starts + count // 2])
        else:
            stat[ids] = val[starts + np.floor((count - 1) * args.coh_patch_pct / 100).astype(int)]
    del lab_v, val
    area = patch_area_km2(labels, n_patches)
    enough = (n_coh >= args.coh_patch_min_cover * size) & (n_coh > 0)
    big = area >= args.coh_patch_min_km2
    remove = np.isfinite(stat) & (stat > args.coh_patch_th) & enough & big
    remove[0] = False
    removed = remove[labels]
    del labels
    print(f"coherent patches: {n_patches} patches of classes {COH_PATCH_CLASSES}, "
          f"{int((enough & big)[1:].sum())} testable (>= {args.coh_patch_min_km2} km2, "
          f"coherence on >= {args.coh_patch_min_cover:.0%}), "
          f"{int(remove.sum())} removed (p{args.coh_patch_pct:g} event coherence > "
          f"{args.coh_patch_th}): {removed.sum()} px, {area[remove].sum():.2f} km2 "
          f"of {area[1:].sum():.2f} km2")
    for c in COH_PATCH_CLASSES:
        if (k := (removed & (out == c)).sum()):
            print(f"  class {c} -> 0: {k} px")
    out[removed] = 0
    change[removed] = 8
    del removed, remove, stat, n_coh, size, area

    if not args.no_post_compactness:
        # Whole-object compactness after the deletion (objects = 8-connected water of classes 1-5,
        # judged with the threshold of their most reliable class; class-1 objects are kept). Same
        # score as RAPID morph: area / 4-connected boundary pixels. Only classes 2-5 are removed.
        comp_th = np.asarray([float(x) for x in args.post_compactness.split(",")])
        water = (out >= 1) & (out <= 5)
        labels, n_obj = ndimage.label(water, structure=np.ones((3, 3), bool))
        if n_obj:
            area_px = np.bincount(labels.ravel(), minlength=n_obj + 1).astype(float)
            lab_w, cls_w = labels[water], out[water].astype(np.int32)
            min_class = np.full(n_obj + 1, 6, np.int32)
            for c in range(5, 0, -1):
                min_class[np.bincount(lab_w[cls_w == c], minlength=n_obj + 1) > 0] = c
            del lab_w, cls_w
            obj = labels > 0
            boundary = np.zeros(labels.shape, bool)
            boundary[0, :] |= obj[0, :]; boundary[-1, :] |= obj[-1, :]
            boundary[:, 0] |= obj[:, 0]; boundary[:, -1] |= obj[:, -1]
            d = labels[1:, :] != labels[:-1, :]
            boundary[1:, :] |= obj[1:, :] & d; boundary[:-1, :] |= obj[:-1, :] & d
            d = labels[:, 1:] != labels[:, :-1]
            boundary[:, 1:] |= obj[:, 1:] & d; boundary[:, :-1] |= obj[:, :-1] & d
            del d, obj
            perim = np.bincount(labels[boundary], minlength=n_obj + 1).astype(float)
            del boundary
            score = np.where(perim > 0, area_px / np.maximum(perim, 1), np.inf)
            th = np.where(min_class <= 5, comp_th[np.clip(min_class, 1, 5) - 1], 0)
            drop = (min_class >= 2) & (min_class <= 5) & (score < th)
            drop[0] = False
            dropped = drop[labels]
            del labels
            print(f"post compactness: {n_obj} water objects, {int(drop.sum())} removed "
                  f"({int(dropped.sum())} px, classes 2-5 only)")
            out[dropped] = 0
            change[dropped] = 9
            del dropped, drop, score, th, perim, min_class, area_px

change[res == 255] = 255
profile.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256)
with rasterio.open(f"{args.prefix}_Morph_masked.tif", "w", **profile) as dst:
    dst.write(out, 1)
with rasterio.open(f"{args.prefix}_mask_change.tif", "w", **profile) as dst:
    # 0 unchanged; desert: 1 kept flood, 2 -> 0, 3 -> 200; snow: 4 -> 100, 5 -> 106, 6 -> 250, 7 -> 116;
    # 8 coherent patch -> 0; 9 removed by compactness after the coherent-patch rule
    dst.write(change, 1)
u, c = np.unique(out, return_counts=True)
print("output classes:", dict(zip(u.tolist(), c.tolist())))
print(f"-> {args.prefix}_Morph_masked.tif, {args.prefix}_mask_change.tif")
