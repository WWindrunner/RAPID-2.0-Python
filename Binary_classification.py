import os
import dask
import dask.array as da
import numpy as np
from utils import *
from scipy.stats.distributions import chi2
from scipy.special import gamma, iv
import rasterio
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from EM_threshold import *
import scipy.ndimage.filters as filters
from scipy.ndimage import median_filter
from scipy.io import savemat, loadmat
from Initial_sample import *


def _first_col(x):
    x = np.asarray(x)
    return x[:, 0] if x.ndim == 2 else x


def _mode_0_100_int(values: np.ndarray) -> float:
    v = np.asarray(values).ravel()
    if v.size == 0:
        return float("nan")
    v = np.clip(v.astype(np.int32, copy=False), 0, 100)
    return float(np.argmax(np.bincount(v, minlength=101)))


def _hist_peak(sample: np.ndarray, bin_width: float) -> float:
    s = np.asarray(sample, dtype=np.float64).ravel()
    if s.size == 0:
        return float("nan")
    mn = np.nanmin(s)
    mx = np.nanmax(s)
    if not np.isfinite(mn) or not np.isfinite(mx):
        return float("nan")
    if mx == mn:
        return float(mn)

    start = np.floor(mn / bin_width) * bin_width
    stop = np.ceil(mx / bin_width) * bin_width
    edges = np.arange(start, stop + bin_width * 1.0000001, bin_width, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([start, start + bin_width], dtype=np.float64)

    counts, edges = np.histogram(s, bins=edges)
    pdf = counts.astype(np.float64) / (s.size * bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return float(centers[int(np.argmax(pdf))])


def select_water_tiles(args, ds):
    """
    Dask-friendly version:
      - Reads tiles from ds.* (dask arrays) by slicing then .compute() to numpy
      - Runs the same MATLAB logic (percentile/while/hist)
      - Stores outputs on ds: r0Sel,c0Sel,nRowsSel,nColsSel,rWaterSeeds,cWaterSeeds,thOcc,modeOcc,medianWater,stdWater,peakWater

    Required in ds:
      ds.water_occurrence  (0..100)
      ds.land_cover
      ds.ocean_mask        (ocean==1)
      ds.img_lp            (linear sigma0, must be >0; will be converted to dB)

    Required in args:
      args.thOccLim (low, high) e.g. (5,99)
      args.thMinSampleSize
      args.thIPrc
      args.thFrac
      args.binWidthInit
      args.codePW
      args.r0, args.c0, args.nRows, args.nCols  (0-based tiles)
      args.showProg (optional)
    """

    # tiles (0-based)
    r0 = _first_col(args.r0)
    c0 = _first_col(args.c0)
    nRows = _first_col(args.nRows)
    nCols = _first_col(args.nCols)

    nR = int(r0.shape[0])
    nC = int(c0.shape[0])

    th_low, th_high = float(args.thOccLim[0]), float(args.thOccLim[1])

    medianWater = np.full((nR, nC), np.nan, dtype=np.float64)
    stdWater = np.full((nR, nC), np.nan, dtype=np.float64)
    peakWater = np.full((nR, nC), np.nan, dtype=np.float64)
    thOcc = np.full((nR, nC), np.nan, dtype=np.float64)
    modeOcc = np.full((nR, nC), np.nan, dtype=np.float64)

    total = nR * nC

    # NOTE: we only compute per-window slices, not full arrays
    for iR in range(nR):
        for iC in range(nC):
            if getattr(args, "showProg", False):
                k = iR * nC + (iC + 1)
                print(f"\r{int(k / total * 100)}%...", end="", flush=True)

            row0 = int(r0[iR])
            col0 = int(c0[iC])
            h = int(nRows[iR])
            w = int(nCols[iC])

            # pull window into memory (numpy)
            water_prob = ds.water_occurrence[row0:row0 + h, col0:col0 + w].compute()
            LCC = ds.land_cover[row0:row0 + h, col0:col0 + w].compute()
            I = ds.img_lp[row0:row0 + h, col0:col0 + w].compute()

            # ocean mask is optional: if ds.ocean_mask missing, treat as all non-ocean
            if hasattr(ds, "ocean_mask") and ds.ocean_mask is not None:
                isOcean = ds.ocean_mask[row0:row0 + h, col0:col0 + w].compute()
                basinMask = (isOcean != 1)
            else:
                basinMask = np.ones_like(I, dtype=bool)

            basinMask = basinMask & np.isfinite(I) & (I > 0) & (LCC == int(args.codePW))
            if np.count_nonzero(basinMask) == 0:
                continue

            I_dB = 10.0 * np.log10(I.astype(np.float64, copy=False))

            thOcc[iR, iC] = th_high
            sample0 = None
            isSample0 = None

            while thOcc[iR, iC] > th_low:
                isSample = basinMask & (water_prob >= thOcc[iR, iC])

                if sample0 is None or sample0.size == 0:
                    if np.count_nonzero(isSample) > int(args.thMinSampleSize):
                        sample0 = I_dB[isSample]
                        isSample0 = isSample
                        medianWater[iR, iC] = float(np.median(sample0))
                    thOcc[iR, iC] -= 1
                    continue

                I_prc = float(np.percentile(sample0, float(args.thIPrc)))
                newMask = isSample & (~isSample0)
                Inew = I_dB[newMask]

                if Inew.size > 0:
                    fracHigh = float(np.count_nonzero(Inew > I_prc)) / float(Inew.size)
                    if fracHigh > float(args.thFrac):
                        medianWater[iR, iC] = float(np.median(sample0))
                        stdWater[iR, iC] = float(np.std(sample0, ddof=1))
                        thOcc[iR, iC] += 1
                        break
                    else:
                        sample0 = I_dB[isSample]
                        isSample0 = isSample

                thOcc[iR, iC] -= 1

            if sample0 is not None and sample0.size > 0:
                modeOcc[iR, iC] = _mode_0_100_int(water_prob[isSample0])
                peakWater[iR, iC] = _hist_peak(sample0, float(args.binWidthInit))
                stdWater[iR, iC] = float(np.std(sample0, ddof=1))
            else:
                thOcc[iR, iC] = np.nan

    if getattr(args, "showProg", False):
        print()

    maskTiles = np.isfinite(thOcc)
    r_idx, c_idx = np.where(maskTiles)

    # Selected tiles (0-based window indices)
    rWaterSeeds = r_idx.astype(np.int32)
    cWaterSeeds = c_idx.astype(np.int32)

    # Selected windows' pixel origins/sizes
    r0Sel = r0[r_idx]
    c0Sel = c0[c_idx]
    nRowsSel = nRows[r_idx]
    nColsSel = nCols[r_idx] if nCols.shape[0] == nR else nCols[c_idx]  # safe if your shapes differ

    # Flatten selected statistics (order is numpy row-major)
    setattr(ds, "r0Sel", r0Sel)
    setattr(ds, "c0Sel", c0Sel)
    setattr(ds, "nRowsSel", nRowsSel)
    setattr(ds, "nColsSel", nColsSel)
    setattr(ds, "rWaterSeeds", rWaterSeeds)
    setattr(ds, "cWaterSeeds", cWaterSeeds)
    setattr(ds, "thOcc", thOcc[maskTiles])
    setattr(ds, "modeOcc", modeOcc[maskTiles])
    setattr(ds, "medianWater", medianWater[maskTiles])
    setattr(ds, "stdWater", stdWater[maskTiles])
    setattr(ds, "peakWater", peakWater[maskTiles])

    return args, ds


def findRange_bimodality(args, ds):
    """
      if exist(SBAInit.mat) load else select_water_tiles + save
    Uses ds.* already loaded by Load_Data.
    """

    # defaults (same as your snippet)
    args.thOccLim = getattr(args, "thOccLim", (5, 99))
    args.codePW = getattr(args, "codePW", 80)
    args.thMinSampleSize = getattr(args, "thMinSampleSize", 500)
    args.thIPrc = getattr(args, "thIPrc", 50)
    args.thFrac = getattr(args, "thFrac", 1 - args.thIPrc / 100.0 + 0.1)
    args.binWidthInit = getattr(args, "binWidthInit", 0.2)

    if not getattr(args, "fileSBAInit", None):
        if not getattr(args, "OUT_PATH", None):
            raise ValueError("Need args.fileSBAInit or args.OUT_PATH")
        args.fileSBAInit = os.path.join(args.OUT_PATH, "SBAInit.mat")

    if os.path.exists(args.fileSBAInit):
        d = loadmat(args.fileSBAInit)
        # load into ds so downstream can use it
        for k in ["r0Sel","c0Sel","nRowsSel","nColsSel",
                  "rWaterSeeds","cWaterSeeds",
                  "thOcc","modeOcc","medianWater","stdWater","peakWater"]:
            if k in d:
                setattr(ds, k, np.asarray(d[k]).squeeze())
    else:
        args, ds = select_water_tiles(args, ds)
        os.makedirs(os.path.dirname(args.fileSBAInit) or ".", exist_ok=True)
        savemat(args.fileSBAInit, {
            "r0Sel": ds.r0Sel,
            "c0Sel": ds.c0Sel,
            "nRowsSel": ds.nRowsSel,
            "nColsSel": ds.nColsSel,
            "rWaterSeeds": ds.rWaterSeeds,
            "cWaterSeeds": ds.cWaterSeeds,
            "thOcc": ds.thOcc,
            "modeOcc": ds.modeOcc,
            "medianWater": ds.medianWater,
            "stdWater": ds.stdWater,
            "peakWater": ds.peakWater,
        })

    nWindows = int(np.size(getattr(ds, "r0Sel", np.array([]))))
    print(f"detected {nWindows} initial windows for sampling")
    return args, ds


@tim
def optimizeDist6(args, ds):
    # SBA init selection (only step a)
    args, ds = findRange_bimodality(args, ds)

    return args, ds


@tim
def dist_th(args, ds):

    return args, ds


def Binary_classification(args, ds):
    # try:
    with Timer('BC TIME ' + args.filepath[0]):
        args, ds = optimizeDist6(args, ds)
        if not args.failed:
            print('finished initialization ...')
        else:
            print('initialization failed ...')
        args, ds = dist_th(args, ds)
        # except Exception as e:
        #    print(e)
        if bool(args.tempfiles):
            for itemp in args.tempfiles:
                print('closing ' + itemp.name)
                itemp.close()
    return args, ds
