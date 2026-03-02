import os
import dask
import dask.array as da
import numpy as np
from utils import *
from scipy.stats.distributions import chi2
from scipy.special import gamma, iv
import rasterio
import csv
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from EM_threshold import *
import scipy.ndimage.filters as filters
from scipy.ndimage import median_filter
from scipy.io import savemat, loadmat
from Initial_sample import *
from typing import Tuple
from rasterio.windows import Window


def get_block(
        rowsImage: int,
        colsImage: int,
        block_size: int,
        overlap: int = 0,
        offset: bool = False,
        keep_size: bool = False
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Same logic as your file-based get_block, but uses provided image shape.
    Conventions:
      - Output r0, c0 are 0-based (top-left pixel offsets).
      - If overlap==0: r0 shape (nR,), c0 shape (nC,)
      - If overlap>0: r0 shape (nR,2), c0 shape (nC,2) (read/process columns)
    """
    rowsImage = int(rowsImage)
    colsImage = int(colsImage)

    if overlap > 0 and offset:
        raise ValueError("cannot set non-zero overlap with offset")

    hasReminder = False
    offsetGrids = int(round(block_size / 2))

    if offset:
        rowsImage1 = rowsImage - offsetGrids
        colsImage1 = colsImage - offsetGrids
        nR = int(np.floor(rowsImage1 / block_size) + (rowsImage1 % block_size > 0) + 1)
        nC = int(np.floor(colsImage1 / block_size) + (colsImage1 % block_size > 0) + 1)
        if rowsImage1 % block_size > 0:
            hasReminder = True
    else:
        nR = int(np.floor(rowsImage / block_size) + (rowsImage % block_size > 0))
        nC = int(np.floor(colsImage / block_size) + (colsImage % block_size > 0))
        if rowsImage % block_size > 0:
            hasReminder = True

    if overlap == 0:
        if offset:
            r0 = np.array([0] + [k * block_size + offsetGrids for k in range(nR - 1)], dtype=np.int64)
            c0 = np.array([0] + [k * block_size + offsetGrids for k in range(nC - 1)], dtype=np.int64)

            nRows = np.minimum(rowsImage - r0, block_size).astype(np.int64)
            nCols = np.minimum(colsImage - c0, block_size).astype(np.int64)
            nRows[0] = offsetGrids
            nCols[0] = offsetGrids
        else:
            r0 = (np.arange(nR, dtype=np.int64) * block_size)
            c0 = (np.arange(nC, dtype=np.int64) * block_size)
            nRows = np.minimum(rowsImage - r0, block_size).astype(np.int64)
            nCols = np.minimum(colsImage - c0, block_size).astype(np.int64)
    else:
        r0 = np.zeros((nR, 2), dtype=np.int64)
        r0[:, 0] = np.arange(nR, dtype=np.int64) * block_size - overlap
        r0[0, 0] = 0
        r0[:, 1] = np.arange(nR, dtype=np.int64) * block_size

        nRows = np.zeros((nR, 2), dtype=np.int64)
        nRows[:, 0] = np.minimum(block_size + overlap * 2, rowsImage - r0[:, 0])
        nRows[0, 0] = min(block_size + overlap, rowsImage)
        nRows[-1, 0] = min(rowsImage - r0[-1, 0], block_size + overlap)
        nRows[:, 1] = np.minimum(block_size, rowsImage - r0[:, 1])
        nRows[-1, 1] = min(rowsImage - r0[-1, 1], block_size)

        c0 = np.zeros((nC, 2), dtype=np.int64)
        c0[:, 0] = np.arange(nC, dtype=np.int64) * block_size - overlap
        c0[0, 0] = 0
        c0[:, 1] = np.arange(nC, dtype=np.int64) * block_size

        nCols = np.zeros((nC, 2), dtype=np.int64)
        nCols[:, 0] = np.minimum(block_size + overlap * 2, colsImage - c0[:, 0])
        nCols[0, 0] = min(block_size + overlap, colsImage)
        nCols[-1, 0] = min(colsImage - c0[-1, 0], block_size + overlap)
        nCols[:, 1] = np.minimum(block_size, colsImage - c0[:, 1])
        nCols[-1, 1] = min(colsImage - c0[-1, 1], block_size)

    if keep_size and hasReminder:
        if overlap == 0:
            r0[-1] = max(0, rowsImage - block_size)
            c0[-1] = max(0, colsImage - block_size)
            nRows[-1] = min(block_size, rowsImage)
            nCols[-1] = min(block_size, colsImage)
        else:
            r0[-1, 0] = max(0, rowsImage - block_size - 2 * overlap)
            r0[-1, 1] = r0[-2, 1] + block_size
            c0[-1, 0] = max(0, colsImage - block_size - 2 * overlap)
            c0[-1, 1] = c0[-2, 1] + block_size
            nRows[-1, 0] = block_size + 2 * overlap
            nRows[-1, 1] = max(1, rowsImage - r0[-1, 1])
            nCols[-1, 0] = block_size + 2 * overlap
            nCols[-1, 1] = max(1, colsImage - c0[-1, 1])

    return nR, nC, r0, c0, nRows, nCols, rowsImage, colsImage


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
            water_prob = water_prob.astype(np.float64, copy=False)
            water_prob[water_prob >= 255] = np.nan

            # ocean mask is optional: if ds.ocean_mask missing, treat as all non-ocean
            if hasattr(ds, "ocean_mask") and ds.ocean_mask is not None:
                isOcean = ds.ocean_mask[row0:row0 + h, col0:col0 + w].compute()
                basinMask = (isOcean != 1)
            else:
                basinMask = np.ones_like(I, dtype=bool)

            basinMask = basinMask & np.isfinite(I) & (I > 0) #& (LCC == int(args.codePW))
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

def filter_windows_by_modeOcc(args, ds):
    """
    After ds has initial windows from select_water_tiles:
      ds.r0Sel, ds.c0Sel, ds.nRowsSel, ds.nColsSel
      ds.rWaterSeeds, ds.cWaterSeeds
      ds.thOcc, ds.modeOcc, ds.medianWater, ds.stdWater, ds.peakWater

    Implements:
      if nWindows < min_windowsWater: warning
      if nWindows > rec_windowsWater: find thOCCFilter then remove modeOcc <= thOCCFilter
    """
    r0Sel = np.asarray(ds.r0Sel).reshape(-1)
    nWindows = r0Sel.size
    print(f"detected {nWindows} initial windows for sampling")

    if nWindows < int(args.min_windowsWater):
        print("WARNING: insufficient initialization windows")

    if nWindows > int(args.rec_windowsWater):
        th_low, th_high = float(args.thOccLim[0]), float(args.thOccLim[1])
        modeOcc = np.asarray(ds.modeOcc).reshape(-1)

        thOCCFilter = th_low
        for t in range(int(th_high), int(th_low) - 1, -1):
            if np.sum(modeOcc > t) > int(args.rec_windowsWater):
                thOCCFilter = t
                break

        mask_remove = (modeOcc <= thOCCFilter)

        # apply remove for all associated arrays
        for name in [
            "r0Sel","c0Sel","nRowsSel","nColsSel",
            "rWaterSeeds","cWaterSeeds",
            "thOcc","modeOcc","medianWater","stdWater","peakWater"
        ]:
            v = np.asarray(getattr(ds, name)).reshape(-1)
            setattr(ds, name, v[~mask_remove])

        nWindows = np.asarray(ds.r0Sel).size

    print(f"kept {nWindows} final windows for sampling")
    return args, ds


# -----------------------------
# thresholdGMM (Python port)
# -----------------------------

def _weighted_pdf(x, w, mu, sig):
    return (w / (np.sqrt(2.0*np.pi) * sig)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def _solve_gaussian_intersection(w1, mu1, s1, w2, mu2, s2):
    """
    Solve w1*N(mu1,s1) = w2*N(mu2,s2) in 1D.
    Returns a threshold x (standardized space). Prefer solution between means.
    """
    # log form:
    # log(w1/s1) - (x-mu1)^2/(2s1^2) = log(w2/s2) - (x-mu2)^2/(2s2^2)
    # Rearranged to quadratic ax^2 + bx + c = 0
    a = (1.0/(2*s2*s2)) - (1.0/(2*s1*s1))
    b = (mu1/(s1*s1)) - (mu2/(s2*s2))
    c = (mu2*mu2/(2*s2*s2)) - (mu1*mu1/(2*s1*s1)) + np.log((w2*s1)/(w1*s2))

    if np.isclose(a, 0.0):
        # linear
        if np.isclose(b, 0.0):
            return 0.5*(mu1+mu2)
        return -c / b

    disc = b*b - 4*a*c
    if disc < 0:
        return 0.5*(mu1+mu2)

    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)

    lo, hi = (min(mu1, mu2), max(mu1, mu2))
    cand = []
    if lo <= r1 <= hi: cand.append(r1)
    if lo <= r2 <= hi: cand.append(r2)

    if len(cand) == 1:
        return cand[0]
    if len(cand) == 2:
        # choose the one with higher average pdf (same spirit as MATLAB picks higher density)
        p1 = (_weighted_pdf(cand[0], w1, mu1, s1) + _weighted_pdf(cand[0], w2, mu2, s2)) / 2.0
        p2 = (_weighted_pdf(cand[1], w1, mu1, s1) + _weighted_pdf(cand[1], w2, mu2, s2)) / 2.0
        return cand[int(p2 > p1)]
    # if none between means, take the root closer to midpoint
    mid = 0.5*(mu1+mu2)
    return r1 if abs(r1-mid) < abs(r2-mid) else r2


def threshold_gmm_window(I_lin, probW, thProb, waterOnly=False):
    """
    Equivalent to thresholdGMM(filePol,fileProb,thProb,r0,c0,nRows,nCols,waterOnly,false)
    but takes arrays already cropped to one window.

    Inputs:
      I_lin: numpy 2D, linear backscatter (>0)
      probW: numpy 2D or None, water occurrence 0..100
      thProb: float threshold (e.g., thOcc[w])
      waterOnly: bool (MATLAB worker passes false)

    Outputs:
      th (Iu), pc (2,), mu (2,), sigma (2,), RH (float)
      If insufficient samples: return NaNs.
    """
    I = np.asarray(I_lin, dtype=np.float64)
    basinMask0 = np.isfinite(I) & (I > 0) & (~np.isinf(I))

    if probW is not None:
        p = np.asarray(probW, dtype=np.float64)
        basinMask = basinMask0 & (p >= float(thProb))
    else:
        basinMask = basinMask0

    if np.count_nonzero(basinMask) == 0:
        return np.nan, np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.nan

    if not waterOnly:
        rr, cc = np.where(basinMask)
        minRow, maxRow = rr.min(), rr.max()
        minCol, maxCol = cc.min(), cc.max()
        hH = int(round((maxRow - minRow) / 2.0))
        hW = int(round((maxCol - minCol) / 2.0))
        rowTop = max(0, minRow - hH)
        rowBtm = min(I.shape[0] - 1, maxRow + hH)
        colLeft = max(0, minCol - hW)
        colRight = min(I.shape[1] - 1, maxCol + hW)
        I0 = I[rowTop:rowBtm+1, colLeft:colRight+1]
        vec = I0[np.isfinite(I0) & (~np.isinf(I0)) & (I0 > 0)]
    else:
        vec = I[basinMask]

    if vec.size < 100:
        return np.nan, np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.nan

    I_db = 10.0 * np.log10(vec)
    I_mean = float(np.mean(I_db))
    I_std = float(np.std(I_db, ddof=1))
    if I_std == 0:
        return np.nan, np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.nan

    Z = (I_db - I_mean) / I_std
    Z = Z.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2,
        max_iter=1000,
        reg_covar=1e-6,
        n_init=20,
        covariance_type="full",
        random_state=0
    )
    gmm.fit(Z)

    mu_std = gmm.means_.flatten()
    sig_std = np.sqrt(gmm.covariances_.flatten())
    pc = gmm.weights_.flatten()

    # sort by mean
    order = np.argsort(mu_std)
    mu_std = mu_std[order]
    sig_std = sig_std[order]
    pc = pc[order]

    # intersection in standardized space
    x_th = _solve_gaussian_intersection(pc[0], mu_std[0], sig_std[0], pc[1], mu_std[1], sig_std[1])

    mu = mu_std * I_std + I_mean
    sigma = sig_std * I_std
    th = x_th * I_std + I_mean  # Iu in dB

    # RH computation: thDensity / peakDensity
    # MATLAB: histogram(I,50,'Normalization','pdf'), peakDensity = max(hist.Values)
    counts, edges = np.histogram(I_db, bins=50, density=True)
    peakDensity = float(np.max(counts)) if counts.size else np.nan

    f1 = _weighted_pdf(x_th, pc[0], mu_std[0], sig_std[0])
    f2 = _weighted_pdf(x_th, pc[1], mu_std[1], sig_std[1])
    thDensity = (f1 + f2) / I_std  # match MATLAB /I_std scaling
    RH = float(thDensity / peakDensity) if (peakDensity and np.isfinite(peakDensity)) else np.nan

    return float(th), pc.astype(np.float64), mu.astype(np.float64), sigma.astype(np.float64), RH


# -----------------------------
# validateGMM (Python port)
# -----------------------------

def validate_gmm(RH, pc, mu, sigma, Iu, thPC=(0.001, 0.999), thAD=0.0, thRH=0.5, thDist2Mean=0.1, separately=True):
    """
    Port of validateGMM.m for the 2-mode case.
    Returns (maskValid, AD)
    """
    # maskIu: real (MATLAB imag(Iu)==0)
    maskIu = np.isfinite(Iu) and (not np.iscomplexobj(Iu))

    pc = np.asarray(pc).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)

    if pc.size < 2 or mu.size < 2 or sigma.size < 2 or not np.isfinite(RH):
        return False, np.nan

    if separately:
        maskPC = (pc[0] > thPC[0]) and (pc[0] < thPC[1])
        AD = float(abs(mu[0] - mu[1]) / min(sigma[0], sigma[1]))
        maskAD = AD > thAD
        maskRH = float(RH) < thRH

        dist2Mean = min(abs(Iu - mu[0]), abs(Iu - mu[1]))
        mStd = np.sqrt(sigma[0] * sigma[1])
        maskDist2Mean = (dist2Mean / mStd) > thDist2Mean
    else:
        # not needed for 2-mode windows here
        maskPC = True
        AD = float(abs(mu[0] - mu[1]) / min(sigma[0], sigma[1]))
        maskAD = AD > thAD
        maskRH = float(RH) < thRH
        dist2Mean = min(abs(Iu - mu[0]), abs(Iu - mu[1]))
        mStd = np.sqrt(sigma[0] * sigma[1])
        maskDist2Mean = (dist2Mean / mStd) > thDist2Mean

    maskValid = bool(maskPC and maskRH and maskAD and maskIu and maskDist2Mean)
    return maskValid, AD


# -----------------------------
# Worker task (init_find_bimod_th_worker equivalent)
# -----------------------------

def _init_bimod_one_window(args, ds, w):
    """
    One dask task for one window.
    Uses ds.img_lp + ds.water_occurrence, plus ds.ocean_mask already applied earlier in your seed selection.
    """
    thOcc = float(np.asarray(ds.thOcc)[w])
    r0 = int(np.asarray(ds.r0Sel)[w])
    c0 = int(np.asarray(ds.c0Sel)[w])
    nRows = int(np.asarray(ds.nRowsSel)[w])
    nCols = int(np.asarray(ds.nColsSel)[w])

    I_lin = ds.img_lp[r0:r0+nRows, c0:c0+nCols].compute()
    probW = ds.water_occurrence[r0:r0+nRows, c0:c0+nCols].compute()

    Iu, pc, mu, sigma, RH = threshold_gmm_window(I_lin, probW, thOcc, waterOnly=False)
    maskValid, AD = validate_gmm(
        RH, pc, mu, sigma, Iu,
        thPC=tuple(args.thPC),
        thAD=float(args.thAD),
        thRH=float(args.thRH),
        thDist2Mean=float(args.thDist2Mean),
        separately=True
    )

    return {
        "w": w,
        "Iu": Iu,
        "pc": pc,
        "mu": mu,
        "sigma": sigma,
        "RH": RH,
        "AD": AD,
        "valid": maskValid,
    }


def run_init_bimodality_dask(args, ds):
    r0s = np.asarray(ds.r0Sel).astype(np.int64).ravel()
    c0s = np.asarray(ds.c0Sel).astype(np.int64).ravel()
    nrs = np.asarray(ds.nRowsSel).astype(np.int64).ravel()
    ncs = np.asarray(ds.nColsSel).astype(np.int64).ravel()
    ths = np.asarray(ds.thOcc).astype(np.float64).ravel()

    nWindows = r0s.size
    if nWindows == 0:
        return args, ds

    # 1) 联合 bbox（最小覆盖）
    r_min = int(r0s.min())
    c_min = int(c0s.min())
    r_max = int((r0s + nrs).max())
    c_max = int((c0s + ncs).max())

    # 2) 一次性把 bbox 子区域 compute 成 numpy
    # 注意：这里不会创建任何新的 dask task；只是把必要数据拉到本 worker
    img_bbox = ds.img_lp[r_min:r_max, c_min:c_max].compute()
    wo_bbox = ds.water_occurrence[r_min:r_max, c_min:c_max].compute()

    # 3) 分配输出
    Iu = np.full((nWindows,), np.nan, dtype=np.float64)
    RH = np.full((nWindows,), np.nan, dtype=np.float64)
    AD = np.full((nWindows,), np.nan, dtype=np.float64)
    pc = np.full((nWindows, 2), np.nan, dtype=np.float64)
    mu = np.full((nWindows, 2), np.nan, dtype=np.float64)
    sigma = np.full((nWindows, 2), np.nan, dtype=np.float64)
    valid = np.zeros((nWindows,), dtype=bool)

    # 4) 纯 numpy window 循环（没有 dask / 没有 compute）
    for w in range(nWindows):
        r0 = int(r0s[w]);
        c0 = int(c0s[w])
        nRows = int(nrs[w]);
        nCols = int(ncs[w])
        thOcc = float(ths[w])

        rr0 = r0 - r_min
        cc0 = c0 - c_min

        I_lin = img_bbox[rr0:rr0 + nRows, cc0:cc0 + nCols]
        probW = wo_bbox[rr0:rr0 + nRows, cc0:cc0 + nCols]

        Iu_w, pc_w, mu_w, sigma_w, RH_w = threshold_gmm_window(
            I_lin, probW, thOcc, waterOnly=False
        )
        maskValid_w, AD_w = validate_gmm(
            RH_w, pc_w, mu_w, sigma_w, Iu_w,
            thPC=tuple(args.thPC),
            thAD=float(args.thAD),
            thRH=float(args.thRH),
            thDist2Mean=float(args.thDist2Mean),
            separately=True
        )

        if np.isfinite(Iu_w):
            Iu[w] = float(Iu_w)
        RH[w] = float(RH_w) if np.isfinite(RH_w) else np.nan
        AD[w] = float(AD_w) if np.isfinite(AD_w) else np.nan
        pc[w, :] = np.asarray(pc_w, dtype=np.float64).reshape(2)
        mu[w, :] = np.asarray(mu_w, dtype=np.float64).reshape(2)
        sigma[w, :] = np.asarray(sigma_w, dtype=np.float64).reshape(2)
        valid[w] = bool(maskValid_w)

    # 5) 过滤逻辑保持不变
    maskIuReal = np.isfinite(Iu)
    maskKeep = maskIuReal

    mu1 = mu[:, 0]

    for name, arr in [
        ("thOcc", np.asarray(ds.thOcc).reshape(-1)),
        ("r0Sel", np.asarray(ds.r0Sel).reshape(-1)),
        ("c0Sel", np.asarray(ds.c0Sel).reshape(-1)),
        ("nRowsSel", np.asarray(ds.nRowsSel).reshape(-1)),
        ("nColsSel", np.asarray(ds.nColsSel).reshape(-1)),
        ("peakWater", np.asarray(ds.peakWater).reshape(-1)),
        ("medianWater", np.asarray(ds.medianWater).reshape(-1)),
        ("stdWater", np.asarray(ds.stdWater).reshape(-1)),
        ("rWaterSeeds", np.asarray(ds.rWaterSeeds).reshape(-1)),
        ("cWaterSeeds", np.asarray(ds.cWaterSeeds).reshape(-1)),
    ]:
        setattr(ds, name, arr[maskKeep])

    ds.mu1 = mu1[maskKeep]
    ds.Iu = Iu[maskKeep]
    ds.pc = pc[maskKeep, :]
    ds.mu = mu[maskKeep, :]
    ds.sigma = sigma[maskKeep, :]
    ds.RH = RH[maskKeep]
    ds.AD = AD[maskKeep]
    ds.valid_bimod = valid[maskKeep]

    if getattr(args, "fileSBAInitUnCat", None):
        os.makedirs(os.path.dirname(args.fileSBAInitUnCat) or ".", exist_ok=True)
        savemat(args.fileSBAInitUnCat, {
            "r0Sel": ds.r0Sel,
            "c0Sel": ds.c0Sel,
            "nRowsSel": ds.nRowsSel,
            "nColsSel": ds.nColsSel,
            "thOcc": ds.thOcc,
            "mu1": ds.mu1,
            "Iu": ds.Iu,
            "peakWater": ds.peakWater,
            "medianWater": ds.medianWater,
            "stdWater": ds.stdWater,
        })

    return args, ds


# -----------------------------
# kmeans categorization (MATLAB: kmeans(peakWater,nWaterClasses))
# -----------------------------

def categorize_water_classes(args, ds):
    """
    MATLAB:
      [idx,C]=kmeans(peakWater,nWaterClasses);
      remove classes with ratio < 1/nWaterClasses/2
      save SBAInitCat.mat
    """
    peak = np.asarray(ds.peakWater).reshape(-1, 1)
    nWaterClasses = int(getattr(args, "nWaterClasses", 5))

    if peak.shape[0] == 0:
        raise RuntimeError("No windows left before kmeans categorization.")

    km = KMeans(n_clusters=nWaterClasses, n_init=20, random_state=0)
    idx0 = km.fit_predict(peak)  # 0..k-1
    C0 = km.cluster_centers_.flatten()

    # class ratios
    counts = np.bincount(idx0, minlength=nWaterClasses).astype(np.float64)
    ratio = counts / float(idx0.size)

    # remove classes with too few windows
    class2remove = ratio < (1.0 / nWaterClasses / 2.0)
    keep_mask = ~np.isin(idx0, np.where(class2remove)[0])

    # apply keep
    for name in ["mu1","r0Sel","c0Sel","nRowsSel","nColsSel","thOcc","Iu","peakWater","medianWater","stdWater"]:
        v = np.asarray(getattr(ds, name)).reshape(-1)
        setattr(ds, name, v[keep_mask])

    # recompute idx/C on remaining windows (MATLAB shrinks C and idx)
    peak2 = np.asarray(ds.peakWater).reshape(-1, 1)
    nWaterClasses2 = int(np.unique(idx0[keep_mask]).size)
    # We keep original cluster IDs but compress to 1..K like MATLAB idx
    uniq = np.unique(idx0[keep_mask])
    remap = {old: i+1 for i, old in enumerate(uniq)}  # MATLAB style 1..K
    idx = np.array([remap[i] for i in idx0[keep_mask]], dtype=np.int32)
    C = np.array([C0[i] for i in uniq], dtype=np.float64)

    args.nWaterClasses = int(nWaterClasses2)
    nWindows = int(np.asarray(ds.thOcc).size)
    print(f"kept {nWindows} final windows of the major classes for sampling")

    ds.idx = idx
    ds.C = C

    if getattr(args, "fileSBAInitCat", None):
        os.makedirs(os.path.dirname(args.fileSBAInitCat) or ".", exist_ok=True)
        savemat(args.fileSBAInitCat, {
            "thOcc": ds.thOcc,
            "r0Sel": ds.r0Sel,
            "c0Sel": ds.c0Sel,
            "nRowsSel": ds.nRowsSel,
            "nColsSel": ds.nColsSel,
            "mu1": ds.mu1,
            "Iu": ds.Iu,
            "C": ds.C,
            "idx": ds.idx,
        })

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
    
    # pick a small patch from the center
    #rows, cols = ds.img_lp.shape
    #r0 = rows//2
    #c0 = cols//2
    #H = 512
    #W = 512

    #I_patch = ds.img_lp[r0:r0+H, c0:c0+W].compute()
    #P_patch = ds.water_occurrence[r0:r0+H, c0:c0+W].compute()
    #L_patch = ds.land_cover[r0:r0+H, c0:c0+W].compute()
    #O_patch = ds.ocean_mask[r0:r0+H, c0:c0+W].compute() if hasattr(ds, "ocean_mask") else None

    #print("img_lp: min/max/finite_ratio/>0_ratio",
    #      np.nanmin(I_patch), np.nanmax(I_patch),
    #      np.isfinite(I_patch).mean(),
    #      (I_patch>0).mean())

    #print("water_occurrence: min/max/unique_sample",
    #      np.nanmin(P_patch), np.nanmax(P_patch),
    #      np.unique(P_patch[~np.isnan(P_patch)])[:20])

    #print("land_cover: unique_sample",
    #      np.unique(L_patch[~np.isnan(L_patch)])[:30])

    #if O_patch is not None:
    #    vals, cnts = np.unique(O_patch[~np.isnan(O_patch)], return_counts=True)
    #    print("ocean_mask unique counts (sample patch):", dict(zip(vals.tolist(), cnts.tolist())))

    if not getattr(args, "fileSBAInit", None):
        if not getattr(args, "OUT_PATH", None):
            raise ValueError("Need args.fileSBAInit or args.OUT_PATH")
        args.fileSBAInit = os.path.join(args.OUT_PATH, "SBAInit.mat")

    if os.path.exists(args.fileSBAInit):
        d = loadmat(args.fileSBAInit)
        # load into ds so downstream can use it
        for k in ["r0Sel", "c0Sel", "nRowsSel", "nColsSel",
                  "rWaterSeeds", "cWaterSeeds",
                  "thOcc", "modeOcc", "medianWater", "stdWater", "peakWater"]:
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


def createSampleMapSBA(
    fileSampleMap: str,
    ds,
    thProb,      # per-window array, same length as r0/c0
    r0, c0,      # per-window pixel offsets (0-based)
    nRows, nCols,# per-window sizes
    Iu,          # per-window Iu threshold in dB (NaN -> inf)
    showProg: bool = False,
    ref_raster_path: str | None = None,  # to fetch transform/crs/shape
):
    """
    Returns:
      pol_water: 1D numpy array of linear I values sampled from all windows.

    Notes:
      - Writes a uint8 GeoTIFF sample map:
          1 = sample pixel
          0 = evaluated but not sample (inside written windows)
          255 = nodata elsewhere (not touched)
      - Uses ds.img_lp and ds.water_occurrence (dask arrays). Each window slice is .compute().
    """

    r0 = np.asarray(r0).reshape(-1)
    c0 = np.asarray(c0).reshape(-1)
    nRows = np.asarray(nRows).reshape(-1)
    nCols = np.asarray(nCols).reshape(-1)
    thProb = np.asarray(thProb).reshape(-1)
    Iu = np.asarray(Iu).reshape(-1)

    nWindows = int(r0.size)
    if not (c0.size == nWindows == nRows.size == nCols.size == thProb.size == Iu.size):
        raise ValueError("Window arrays must have the same length.")

    # MATLAB: Iu(isnan(Iu)) = inf
    Iu = Iu.copy()
    Iu[np.isnan(Iu)] = np.inf

    # Determine output raster geometry from ref raster (recommended)
    if ref_raster_path is None:
        # Common in your framework: args.filepath[0]
        # If you don't pass ref_raster_path, you'll get a clear error here.
        raise ValueError("ref_raster_path is required to write a georeferenced GeoTIFF sample map.")

    with rasterio.open(ref_raster_path) as ref:
        rowsImage, colsImage = ref.height, ref.width
        transform = ref.transform
        crs = ref.crs

    # Recreate output file
    if os.path.exists(fileSampleMap):
        os.remove(fileSampleMap)

    profile = {
        "driver": "GTiff",
        "height": rowsImage,
        "width": colsImage,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 255,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Create and fill with nodata=255
    with rasterio.open(fileSampleMap, "w", **profile) as dst:
        # fill whole raster with 255 once (like MATLAB leaving nodata elsewhere)
        dst.write(np.full((rowsImage, colsImage), 255, dtype=np.uint8), 1)

        pol_list = []

        for w in range(nWindows):
            if showProg:
                prog = (w + 1) / nWindows
                print(f"\r{int(prog*100)}%...", end="", flush=True)

            rr = int(r0[w]); cc = int(c0[w])
            hh = int(nRows[w]); ww = int(nCols[w])

            # slice from dask arrays -> numpy
            I = ds.img_lp[rr:rr+hh, cc:cc+ww].compute()
            water_prob = ds.water_occurrence[rr:rr+hh, cc:cc+ww].compute()

            # MATLAB condition: water_prob > thProb(w) & I>0 & I_dB < Iu(w)
            I = np.asarray(I, dtype=np.float64)
            water_prob = np.asarray(water_prob, dtype=np.float64)
            water_prob[water_prob >= 255] = np.nan

            # avoid log10 warnings
            valid_I = np.isfinite(I) & (I > 0)
            I_dB = np.full_like(I, np.nan, dtype=np.float64)
            I_dB[valid_I] = 10.0 * np.log10(I[valid_I])

            isSample = (water_prob > float(thProb[w])) & valid_I & (I_dB < float(Iu[w]))

            if np.any(isSample):
                pol_list.append(I[isSample].astype(np.float64, copy=False))

            # write window: 0/1 inside this window, leave others as nodata
            win = Window(col_off=cc, row_off=rr, width=ww, height=hh)
            dst.write(isSample.astype(np.uint8), 1, window=win)

        if showProg:
            print()

    if len(pol_list) == 0:
        return np.array([], dtype=np.float64)
    return np.concatenate(pol_list, axis=0)


def generate_initial_sample_maps(args, ds, fileSampleMap_root: str, showProg: bool = False):
    """
    Implements the loop:
      for category in 1..nCats:
        maskCat = cat == ucat(category)
        createSampleMapSBA(...)
    """

    print("generating the intial sample mask")

    cat = np.asarray(ds.idx).reshape(-1)  # MATLAB cat == idx (after kmeans)
    ucat = np.unique(cat)
    nCats = int(ucat.size)

    pol_water_by_cat = [None] * nCats

    for i, cat_id in enumerate(ucat, start=1):
        maskCat = (cat == cat_id)

        r0Cat = np.asarray(ds.r0Sel).reshape(-1)[maskCat]
        c0Cat = np.asarray(ds.c0Sel).reshape(-1)[maskCat]
        nRowsCat = np.asarray(ds.nRowsSel).reshape(-1)[maskCat]
        nColsCat = np.asarray(ds.nColsSel).reshape(-1)[maskCat]
        thOccCat = np.asarray(ds.thOcc).reshape(-1)[maskCat]
        IuCat = np.asarray(ds.Iu).reshape(-1)[maskCat]

        fileSampleMap = f"{fileSampleMap_root}{i}.tif"

        pol_water = createSampleMapSBA(
            fileSampleMap=fileSampleMap,
            ds=ds,
            thProb=thOccCat,
            r0=r0Cat, c0=c0Cat,
            nRows=nRowsCat, nCols=nColsCat,
            Iu=IuCat,
            showProg=showProg,
            ref_raster_path=args.filepath[0],  # 关键：用输入影像拿 transform/crs/shape
        )
        pol_water_by_cat[i-1] = pol_water

        # --- placeholders you said to leave for later ---
        # pol_water_dB = 10*np.log10(pol_water)
        # muGau1 = np.mean(pol_water_dB)
        # sigmaGau1 = np.std(pol_water_dB, ddof=1)
        # sigma2, n_est, mKLD, mNSE, mR2, thProbOut, binWidth = findInitDist(...)
        # KLD_chi2_frm_sample(...)

    return pol_water_by_cat


@tim
def optimizeDist6(args, ds):
    # SBA init selection (only step a)
    args, ds = findRange_bimodality(args, ds)

    # 1) filter windows by modeOcc + count rules
    args, ds = filter_windows_by_modeOcc(args, ds)

    # 2) run init bimodality GMM per window (dask parallel)
    args, ds = run_init_bimodality_dask(args, ds)

    # 3) categorize by peakWater (kmeans) + remove tiny classes + save Cat
    args, ds = categorize_water_classes(args, ds)

    if args.writeSampleMap:
        fileSampleMap_root = os.path.join(args.dirTempPol, "SampleMap_")
        generate_initial_sample_maps(
            args, ds,
            fileSampleMap_root,
            showProg=args.showProg
        )

    return args, ds


@tim
def dist_th(args, ds):
    return args, ds


def Binary_classification(args, ds):
    # try:
    with Timer('BC TIME ' + args.filepath[0]):
        rowsImage, colsImage = ds.img_lp.shape
        args.blockSizeSBA = max(int(round(rowsImage / 50)), int(round(colsImage / 50)))
        nR, nC, r0, c0, nRows, nCols, rowsImage, colsImage = get_block(
            rowsImage, colsImage,
            block_size=args.blockSizeSBA,
            overlap=int(round(args.blockSizeSBA * args.overlapRatio / 2)),
            offset=False,
            keep_size=True
        )

        args.nR, args.nC, args.r0, args.c0, args.nRows, args.nCols = nR, nC, r0, c0, nRows, nCols

        args, ds = optimizeDist6(args, ds)
        # TESTING
        sys.exit()

        args, ds = dist_th(args, ds)

        if bool(args.tempfiles):
            for itemp in args.tempfiles:
                print('closing ' + itemp.name)
                itemp.close()
    return args, ds
