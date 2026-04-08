import math
import os
import dask
import dask.array as da
from dask import delayed
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
from rasterio.windows import Window
from Dist_th import dist_th_window_worker, dist_th_postprocess
from Statistics import KLD_chi2_frm_sample, validate_gmm, get_block, build_Iu_and_meanTransact


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
# Worker task (init_find_bimod_th_worker equivalent)
# -----------------------------

def _init_bimod_one_window_from_bbox(args,
                                    img_bbox, wo_bbox,
                                    r_min, c_min,
                                    thOcc, r0, c0, nRows, nCols, w):
    rr0 = int(r0) - int(r_min)
    cc0 = int(c0) - int(c_min)

    I_lin = img_bbox[rr0:rr0 + int(nRows), cc0:cc0 + int(nCols)]
    probW = wo_bbox [rr0:rr0 + int(nRows), cc0:cc0 + int(nCols)]

    Iu, pc, mu, sigma, RH = threshold_gmm_window(I_lin, probW, float(thOcc), waterOnly=False)

    maskValid, AD = validate_gmm(
        RH, pc, mu, sigma, Iu,
        thPC=tuple(args.thPC),
        thAD=float(args.thAD),
        thRH=float(args.thRH),
        thDist2Mean=float(args.thDist2Mean),
        separately=True
    )

    return {
        "w": int(w),
        "Iu": Iu,
        "pc": pc,
        "mu": mu,
        "sigma": sigma,
        "RH": RH,
        "AD": AD,
        "valid": bool(maskValid),
    }


def run_init_bimodality_dask(args, ds):
    # ---- 准备 window 参数 ----
    r0s = np.asarray(ds.r0Sel).astype(np.int64).ravel()
    c0s = np.asarray(ds.c0Sel).astype(np.int64).ravel()
    nrs = np.asarray(ds.nRowsSel).astype(np.int64).ravel()
    ncs = np.asarray(ds.nColsSel).astype(np.int64).ravel()
    ths = np.asarray(ds.thOcc).astype(np.float64).ravel()

    nWindows = int(r0s.size)
    if nWindows == 0:
        return args, ds

    # ---- 联合 bbox ----
    r_min = int(r0s.min())
    c_min = int(c0s.min())
    r_max = int((r0s + nrs).max())
    c_max = int((c0s + ncs).max())

    # ---- 只 compute 两次：把 bbox 子区域拉成 numpy ----
    # 这一步发生在当前 worker/进程内（线程模式下就是同一进程）
    img_bbox = ds.img_lp[r_min:r_max, c_min:c_max].compute()
    wo_bbox = ds.water_occurrence[r_min:r_max, c_min:c_max].compute()

    print("[run_init_bimodality] start serial windows (NO DASK)", flush=True)
    import time

    results = []
    t0 = time.perf_counter()

    # 控制日志频率
    log_every = 10          # 每完成多少个 window 打一次进度
    slow_warn_s = 30      # 单个 window 超过多少秒报警
    topk_track = 5         # 记录最慢的 topk windows

    # 记录最慢 windows（(elapsed, w)）
    slowest = []

    for w in range(nWindows):
        t_w0 = time.perf_counter()

        # 取参数（保持你原来的方式）
        thOcc = float(ths[w])
        r0 = int(r0s[w]); c0 = int(c0s[w])
        nRows = int(nrs[w]); nCols = int(ncs[w])

        # --- 切片（numpy view，理论上很快） ---
        rr0 = r0 - r_min
        cc0 = c0 - c_min
        I_lin = img_bbox[rr0:rr0+nRows, cc0:cc0+nCols]
        probW = wo_bbox [rr0:rr0+nRows, cc0:cc0+nCols]

        # --- 计时：GMM ---
        t_g0 = time.perf_counter()
        Iu, pc, mu, sigma, RH = threshold_gmm_window(I_lin, probW, thOcc, waterOnly=False)
        t_g1 = time.perf_counter()

        # --- 计时：validate ---
        t_v0 = time.perf_counter()
        maskValid, AD = validate_gmm(
            RH, pc, mu, sigma, Iu,
            thPC=tuple(args.thPC),
            thAD=float(args.thAD),
            thRH=float(args.thRH),
            thDist2Mean=float(args.thDist2Mean),
            separately=True
        )
        t_v1 = time.perf_counter()

        t_w1 = time.perf_counter()
        dt = t_w1 - t_w0
        dt_gmm = t_g1 - t_g0
        dt_val = t_v1 - t_v0

        results.append({
            "w": w,
            "Iu": Iu,
            "pc": pc,
            "mu": mu,
            "sigma": sigma,
            "RH": RH,
            "AD": AD,
            "valid": bool(maskValid),

            # --- optional profiling fields ---
            "dt_total": dt,
            "dt_gmm": dt_gmm,
            "dt_val": dt_val,
            "r0": r0, "c0": c0, "nRows": nRows, "nCols": nCols, "thOcc": thOcc
        })

        # 维护 slowest topk
        slowest.append((dt, w))
        slowest.sort(reverse=True)
        if len(slowest) > topk_track:
            slowest = slowest[:topk_track]

        # 进度日志
        if (w + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"[run_init_bimodality] {w+1}/{nWindows} done | elapsed={elapsed:.1f}s | last={dt:.3f}s (gmm={dt_gmm:.3f}s, val={dt_val:.3f}s)",
              flush=True)

        # 慢 window 报警（立即告诉你哪个窗口慢）
        if dt > slow_warn_s:
            print(f"[run_init_bimodality][SLOW] w={w} dt={dt:.3f}s (gmm={dt_gmm:.3f}s, val={dt_val:.3f}s) "
                  f"size={nRows}x{nCols} r0,c0=({r0},{c0}) thOcc={thOcc:.3f}",
                  flush=True)

    # 汇总最慢窗口
    elapsed_all = time.perf_counter() - t0
    print(f"[run_init_bimodality] serial windows finished. total_elapsed={elapsed_all:.1f}s", flush=True)
    print("[run_init_bimodality] top slowest windows:", flush=True)
    for dt, w in slowest:
        r = results[w]
        print(f"  w={w:4d} dt={dt:.3f}s (gmm={r['dt_gmm']:.3f}s, val={r['dt_val']:.3f}s) "
              f"size={r['nRows']}x{r['nCols']} r0,c0=({r['r0']},{r['c0']}) thOcc={r['thOcc']:.3f}",
              flush=True)

    # ---- 分配输出数组 ----
    Iu = np.full((nWindows,), np.nan, dtype=np.float64)
    RH = np.full((nWindows,), np.nan, dtype=np.float64)
    AD = np.full((nWindows,), np.nan, dtype=np.float64)
    pc = np.full((nWindows, 2), np.nan, dtype=np.float64)
    mu = np.full((nWindows, 2), np.nan, dtype=np.float64)
    sigma = np.full((nWindows, 2), np.nan, dtype=np.float64)
    valid = np.zeros((nWindows,), dtype=bool)

    for r in results:
        w = int(r["w"])
        Iu[w] = float(r["Iu"]) if np.isfinite(r["Iu"]) else np.nan
        RH[w] = float(r["RH"]) if np.isfinite(r["RH"]) else np.nan
        AD[w] = float(r["AD"]) if np.isfinite(r["AD"]) else np.nan
        pc[w, :] = np.asarray(r["pc"], dtype=np.float64).reshape(2)
        mu[w, :] = np.asarray(r["mu"], dtype=np.float64).reshape(2)
        sigma[w, :] = np.asarray(r["sigma"], dtype=np.float64).reshape(2)
        valid[w] = bool(r["valid"])

    # ---- MATLAB: maskValid = imag(Iu)==0; 这里用 finite 代替 ----
    maskIuReal = np.isfinite(Iu)
    maskKeep = maskIuReal  # 或：maskKeep = maskIuReal & valid

    mu1 = mu[:, 0]

    # ---- 同步过滤 ds 的相关字段（保持你原逻辑）----
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

    # ---- 保存 UnCat ----
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
        if getattr(args, "dirTempPol", None):
            args.fileSBAInit = os.path.join(args.dirTempPol, "SBAInit.mat")
        elif getattr(args, "dirOut", None):
            args.fileSBAInit = os.path.join(args.dirOut, "SBAInit.mat")
        elif getattr(args, "OUT_PATH", None):
            args.fileSBAInit = os.path.join(args.OUT_PATH, "SBAInit.mat")
        else:
            raise ValueError("Need args.fileSBAInit or one of args.dirTempPol / args.dirOut / args.OUT_PATH")

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
            I_view = ds.img_lp[rr:rr + hh, cc:cc + ww]
            water_view = ds.water_occurrence[rr:rr + hh, cc:cc + ww]
            I, water_prob = dask.compute(I_view, water_view)

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


def findInitDist(
    I: np.ndarray,
    muGaussian,
    sigmaGaussian,
    Iu,
    maxENL: float,
):
    """
    MATLAB:
      [sigma2, ENLa, mKLD, mNSE, mR2, thProb, binWidth] = findInitDist(I, muGaussian, sigmaGaussian, Iu, maxENL)

    Notes
    -----
    - I is linear-domain intensity samples (NOT dB).
    - muGaussian and sigmaGaussian are in dB domain in the original workflow.
    - Iu is provided in dB in your call site (you pass `inf` sometimes).
      In MATLAB they do: Iu = 10^(Iu/10) at the beginning.
    - binWidth is percentile(I, 5) in MATLAB.
    - Two modes:
        mode=1 if accLeft2Peak < 0.1: ENLs=0.2:0.1:1.5, anchor sigma2 from mu/sigma
        mode=2 else: ENLs=1.1:0.1:maxENL, anchor sigma2 from peak position ratio
    - scoring: maximize (1 - KLD + NSE + r^2)
    """
    I = np.asarray(I, dtype=float)
    I = I[np.isfinite(I)]
    if I.size == 0:
        return float("nan"), float("nan"), float("inf"), float("-inf"), 0.0, float("nan"), float("nan")

    # MATLAB: binWidth = prctile(I, 5)
    binWidth = float(np.percentile(I, 5))
    if not np.isfinite(binWidth) or binWidth <= 0:
        # fallback: small positive bin width
        binWidth = float(np.maximum(np.std(I) / 50.0, 1e-6))

    # build histogram to find peak location
    # use bins anchored at min(I) for stability, with fixed binWidth
    i_min = float(np.min(I))
    i_max = float(np.max(I))
    n_bins = int(np.ceil((i_max - i_min) / binWidth)) + 1
    if n_bins < 3:
        n_bins = 3
    edges = i_min + binWidth * np.arange(n_bins + 1, dtype=float)
    hist_prob, bin_edges = np.histogram(I, bins=edges, density=False)
    hist_prob = hist_prob.astype(float)
    hist_prob = hist_prob / np.sum(hist_prob) if np.sum(hist_prob) > 0 else hist_prob

    if hist_prob.size == 0:
        return float("nan"), float("nan"), float("inf"), float("-inf"), 0.0, float("nan"), float("nan")

    locPeaks = int(np.argmax(hist_prob))  # peak bin index
    accLeft2Peak = float(np.sum(hist_prob[: locPeaks + 1]))

    if accLeft2Peak < 0.1:
        ENLs = np.arange(0.2, 1.5 + 1e-12, 0.1)
        mode = 1
    else:
        ENLs = np.arange(1.1, float(maxENL) + 1e-12, 0.1)
        mode = 2

    nLooks = ENLs.size
    KLDs = np.full(nLooks, np.inf, dtype=float)
    NSEs = np.full(nLooks, -np.inf, dtype=float)
    Rs = np.zeros(nLooks, dtype=float)
    thProbs = np.zeros(nLooks, dtype=float)

    # MATLAB: Iu=10^(Iu/10); then mean(muGaussian), mean(sigmaGaussian)
    if np.isfinite(Iu):
        Iu_lin = 10.0 ** (float(Iu) / 10.0)
    else:
        Iu_lin = float("inf")

    muG = float(np.mean(muGaussian))
    sigG = float(np.mean(sigmaGaussian))

    C = np.full(nLooks, np.nan, dtype=float)

    if mode == 1:
        # MATLAB:
        # sigma2 = 10^(mu/10) * exp(sigma^2*(log(10))^2/200)
        sigma2_anchor = (10.0 ** (muG / 10.0)) * math.exp((sigG**2) * (math.log(10.0) ** 2) / 200.0)
        C[:] = sigma2_anchor

        for i, ENL in enumerate(ENLs):
            if np.isfinite(Iu_lin):
                Xu = 2.0 * ENL * Iu_lin / sigma2_anchor
                thProbs[i] = float(chi2.cdf(Xu, df=2.0 * ENL))
            else:
                thProbs[i] = 0.99

            KLDs[i], _, _, Rs[i], NSEs[i] = KLD_chi2_frm_sample(
                I=I,
                sigma2=sigma2_anchor,
                interval=binWidth,
                n=float(ENL),
                p=float(thProbs[i]),
                trailing=3,
                real2theo=False,
                out_figure=None,
            )

    else:
        # MATLAB:
        # Ip = (edges(locPeaks)+edges(locPeaks+1))/2
        # XPeak = 2*ENL-2
        # C = 2*Ip*ENL/XPeak
        Ip = 0.5 * (bin_edges[locPeaks] + bin_edges[locPeaks + 1])

        for i, ENL in enumerate(ENLs):
            XPeak = 2.0 * ENL - 2.0
            if XPeak <= 0:
                # avoid invalid
                C[i] = np.nan
                KLDs[i] = np.inf
                NSEs[i] = -np.inf
                Rs[i] = 0.0
                thProbs[i] = 0.99 if not np.isfinite(Iu_lin) else np.nan
                continue

            C[i] = 2.0 * Ip * ENL / XPeak

            if np.isfinite(Iu_lin):
                Xu = 2.0 * ENL * Iu_lin / C[i]
                thProbs[i] = float(min(0.99, chi2.cdf(Xu, df=2.0 * ENL)))
            else:
                thProbs[i] = 0.99

            KLDs[i], _, _, Rs[i], NSEs[i] = KLD_chi2_frm_sample(
                I=I,
                sigma2=float(C[i]),
                interval=binWidth,
                n=float(ENL),
                p=float(thProbs[i]),
                trailing=3,
                real2theo=False,
                out_figure=None,
            )

    # MATLAB:
    # [maxScore,maxInd]=max(1-KLDs+NSEs+Rs.^2);
    scores = (1.0 - KLDs) + NSEs + (Rs ** 2)
    maxInd = int(np.nanargmax(scores)) if np.any(np.isfinite(scores)) else 0

    mKLD = float(KLDs[maxInd])
    mNSE = float(NSEs[maxInd])
    mR2 = float(Rs[maxInd] ** 2)
    ENLa = float(ENLs[maxInd])
    sigma2_best = float(C[maxInd])
    thProb_best = float(thProbs[maxInd])

    return sigma2_best, ENLa, mKLD, mNSE, mR2, thProb_best, binWidth


def generate_initial_sample_maps(
    args,
    ds,
    fileSampleMap_root: str,
    dirTempPol: str,
    maxENL: float,
    thKLD: float,
    thNSE: float,
    thR2: float,
    save_sample_map: bool = False,
    save_sample_dist: bool = False,
    showProg: bool = False,
):
    """
    Key steps per category:
      1) createSampleMapSBA -> pol_water (linear)
      2) estimate mu/sigma in dB
      3) findInitDist -> sigma2, n_est, scores, thProb, binWidth
      4) KLD_chi2_frm_sample (optional figure)
      5) remove invalid categories by thresholds; delete files
    """
    print("generating the initial sample mask")

    cat = np.asarray(ds.idx).reshape(-1)
    ucat = np.unique(cat)
    nCats = int(ucat.size)

    pol_water_by_cat = [None] * nCats

    sigma2 = np.full(nCats, np.nan, dtype=float)
    n_est = np.full(nCats, np.nan, dtype=float)
    binWidth = np.full(nCats, np.nan, dtype=float)
    mKLD = np.full(nCats, np.nan, dtype=float)
    mNSE = np.full(nCats, np.nan, dtype=float)
    mR2 = np.full(nCats, np.nan, dtype=float)
    thProb = np.full(nCats, np.nan, dtype=float)

    for i, cat_id in enumerate(ucat, start=1):
        maskCat = (cat == cat_id)

        r0Cat = np.asarray(ds.r0Sel).reshape(-1)[maskCat]
        c0Cat = np.asarray(ds.c0Sel).reshape(-1)[maskCat]
        nRowsCat = np.asarray(ds.nRowsSel).reshape(-1)[maskCat]
        nColsCat = np.asarray(ds.nColsSel).reshape(-1)[maskCat]
        thOccCat = np.asarray(ds.thOcc).reshape(-1)[maskCat]
        IuCat = np.asarray(ds.Iu).reshape(-1)[maskCat]

        fileSampleMap = f"{fileSampleMap_root}{i}.tif"
        fileSampleDist = os.path.join(dirTempPol, f"SampleDist_{i}.png")

        # 1) sample map + pol_water
        pol_water = createSampleMapSBA(
            fileSampleMap=fileSampleMap,
            ds=ds,
            thProb=thOccCat,
            r0=r0Cat, c0=c0Cat,
            nRows=nRowsCat, nCols=nColsCat,
            Iu=IuCat,
            showProg=showProg,
            ref_raster_path=args.filepath[0],
        )

        pol_water_by_cat[i - 1] = pol_water

        # 2) dB
        pol_water_dB = 10.0 * np.log10(pol_water)
        muGau1 = float(np.mean(pol_water_dB))
        sigmaGau1 = float(np.std(pol_water_dB, ddof=1))  # MATLAB std 默认 ddof=1

        # 3) initialize distribution
        sigma2_i, n_est_i, mKLD_i, mNSE_i, mR2_i, thProb_i, binWidth_i = findInitDist(
            I=pol_water,
            muGaussian=muGau1,
            sigmaGaussian=sigmaGau1,
            Iu=np.inf,
            maxENL=maxENL,
        )

        sigma2[i - 1] = sigma2_i
        n_est[i - 1] = n_est_i
        mKLD[i - 1] = mKLD_i
        mNSE[i - 1] = mNSE_i
        mR2[i - 1] = mR2_i
        thProb[i - 1] = thProb_i
        binWidth[i - 1] = binWidth_i

        # 4) save (optional)
        if save_sample_dist:
            KLD_chi2_frm_sample(
                I=pol_water,
                sigma2=sigma2_i,
                interval=binWidth_i,
                n=n_est_i,
                p=thProb_i,
                trailing=3,
                real2theo=False,
                out_figure=fileSampleDist,
            )

    # === remove categories with low statistics ===
    maskInvalidCat = (mKLD > thKLD) | (mNSE < thNSE) | (mR2 < thR2)

    indRemove = np.where(maskInvalidCat)[0]  # 0-based indices
    for idx0 in indRemove:
        i = idx0 + 1
        fileSampleMap = f"{fileSampleMap_root}{i}.tif"
        fileSampleDist = os.path.join(dirTempPol, f"SampleDist_{i}.png")
        if os.path.exists(fileSampleMap):
            os.remove(fileSampleMap)
        if os.path.exists(fileSampleDist):
            os.remove(fileSampleDist)

    if not save_sample_map:
        for idx0 in np.where(~maskInvalidCat)[0]:
            i = idx0 + 1
            fileSampleMap = f"{fileSampleMap_root}{i}.tif"
            if os.path.exists(fileSampleMap):
                os.remove(fileSampleMap)

    if not save_sample_dist:
        for idx0 in np.where(~maskInvalidCat)[0]:
            i = idx0 + 1
            fileSampleDist = os.path.join(dirTempPol, f"SampleDist_{i}.png")
            if os.path.exists(fileSampleDist):
                os.remove(fileSampleDist)

    return {
        "pol_water_by_cat": pol_water_by_cat,
        "sigma2": sigma2,
        "n_est": n_est,
        "binWidth": binWidth,
        "mKLD": mKLD,
        "mNSE": mNSE,
        "mR2": mR2,
        "thProb": thProb,
        "maskInvalidCat": maskInvalidCat,
        "ucat": ucat,
    }


@tim
def optimizeDist6(args, ds):
    # SBA init selection (only step a)
    args, ds = findRange_bimodality(args, ds)

    # 1) filter windows by modeOcc + count rules
    args, ds = filter_windows_by_modeOcc(args, ds)

    use_sba_cache = bool(getattr(args, "useSBACache", False))
    print(f"[optimizeDist6] useSBACache={use_sba_cache}", flush=True)

    # 2) run init bimodality GMM per window
    if (
        use_sba_cache
        and getattr(args, "fileSBAInitUnCat", None)
        and os.path.exists(args.fileSBAInitUnCat)
    ):
        print("loading SBAInitUnCat cache ...", flush=True)
        d = loadmat(args.fileSBAInitUnCat)

        # restore arrays with explicit dtype / contiguous layout
        for k in ["r0Sel", "c0Sel", "nRowsSel", "nColsSel"]:
            if k in d:
                setattr(ds, k, np.ascontiguousarray(np.asarray(d[k]).squeeze(), dtype=np.int64))

        for k in ["thOcc", "mu1", "Iu", "peakWater", "medianWater", "stdWater"]:
            if k in d:
                setattr(ds, k, np.ascontiguousarray(np.asarray(d[k]).squeeze(), dtype=np.float64))

        # optional warm-up
        try:
            r0s = np.asarray(ds.r0Sel, dtype=np.int64).ravel()
            c0s = np.asarray(ds.c0Sel, dtype=np.int64).ravel()
            nrs = np.asarray(ds.nRowsSel, dtype=np.int64).ravel()
            ncs = np.asarray(ds.nColsSel, dtype=np.int64).ravel()

            if r0s.size > 0:
                r_min = int(r0s.min())
                c_min = int(c0s.min())
                r_max = int((r0s + nrs).max())
                c_max = int((c0s + ncs).max())

                print(
                    f"[optimizeDist6] warming SBA bbox from cache: "
                    f"r=({r_min},{r_max}), c=({c_min},{c_max})",
                    flush=True
                )

                _ = ds.img_lp[r_min:r_max, c_min:c_max].compute()
                _ = ds.water_occurrence[r_min:r_max, c_min:c_max].compute()

                print("[optimizeDist6] SBA cache warm-up done", flush=True)
            else:
                print("[optimizeDist6] SBA cache warm-up skipped: empty r0Sel/c0Sel", flush=True)

        except Exception as e:
            print(f"[optimizeDist6] SBA cache warm-up skipped due to error: {e}", flush=True)

    else:
        if use_sba_cache:
            print("[optimizeDist6] SBAInitUnCat cache not found, recomputing ...", flush=True)
        args, ds = run_init_bimodality_dask(args, ds)

    # 3) categorize by peakWater (kmeans) + remove tiny classes + save Cat
    if (
        use_sba_cache
        and getattr(args, "fileSBAInitCat", None)
        and os.path.exists(args.fileSBAInitCat)
    ):
        print("loading SBAInitCat cache ...", flush=True)
        d = loadmat(args.fileSBAInitCat)

        for k in ["r0Sel", "c0Sel", "nRowsSel", "nColsSel", "idx"]:
            if k in d:
                setattr(ds, k, np.ascontiguousarray(np.asarray(d[k]).squeeze(), dtype=np.int64))

        for k in ["thOcc", "mu1", "Iu", "C"]:
            if k in d:
                setattr(ds, k, np.ascontiguousarray(np.asarray(d[k]).squeeze(), dtype=np.float64))

    else:
        if use_sba_cache:
            print("[optimizeDist6] SBAInitCat cache not found, recategorizing ...", flush=True)
        args, ds = categorize_water_classes(args, ds)

    # 4) per-category sample map + init dist fitting + category screening
    fileSampleMap_root = os.path.join(args.dirTempPol, "SampleMap_")

    out = generate_initial_sample_maps(
        args=args,
        ds=ds,
        fileSampleMap_root=fileSampleMap_root,
        dirTempPol=args.dirTempPol,
        maxENL=args.maxENL,
        thKLD=args.thKLD,
        thNSE=args.thNSE,
        thR2=args.thR2,
        save_sample_map=args.writeSampleMap,
        save_sample_dist=getattr(args, "writeSampleDist", False),
        showProg=args.showProg,
    )

    ds.sigma2 = out["sigma2"]
    ds.n_est = out["n_est"]
    ds.binWidth = out["binWidth"]
    ds.mKLD = out["mKLD"]
    ds.mNSE = out["mNSE"]
    ds.mR2 = out["mR2"]
    ds.thProb = out["thProb"]
    ds.maskInvalidCat = out["maskInvalidCat"]
    ds.ucat = out["ucat"]
    ds.pol_water_by_cat = out["pol_water_by_cat"]

    return args, ds


def _block_col(x, which: int):
    """
    which=0 -> read window (with overlap)
    which=1 -> process window (core block)
    if x is 1D already, return as-is.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x[:, which]
    return x


@tim
def dist_th(args, ds):
    """
    Optimized Python replacement of MATLAB dist_th:
      - normalize block outputs to process windows
      - precompute thProbs / Iu / meanTransact once
      - read row stripes from dask arrays once per stripe
      - slice windows from local numpy stripe instead of per-window .compute()

    Outputs stored into ds:
      ds.matThProb: (nR, nC, nCats)
      ds.matCat:    (nR, nC, nCats) bool
      ds.matThI_low:(nR, nC)
    """
    print("[dist_th] packing SBA windows and dispatching tasks...")

    # --- normalize get_block outputs to 1D (process col=1) ---
    args.r0 = _block_col(args.r0, 1).astype(np.int64, copy=False)
    args.nRows = _block_col(args.nRows, 1).astype(np.int64, copy=False)
    args.c0 = _block_col(args.c0, 1).astype(np.int64, copy=False)
    args.nCols = _block_col(args.nCols, 1).astype(np.int64, copy=False)

    nR = int(args.nR)
    nC = int(args.nC)
    total = nR * nC

    print(f"[dist_th] grid: nR={nR}, nC={nC}, total_windows={total}", flush=True)

    # --- class stats from optimizeDist6 ---
    sigma2 = np.asarray(ds.sigma2, dtype=float).reshape(-1)
    ENL = np.asarray(ds.n_est, dtype=float).reshape(-1)
    binW = np.asarray(ds.binWidth, dtype=float).reshape(-1)
    nCats = int(sigma2.size)

    print(
        f"[dist_th] class stats: nCats={nCats}, "
        f"sigma2.shape={sigma2.shape}, ENL.shape={ENL.shape}, binW.shape={binW.shape}",
        flush=True
    )

    # --- precompute thProbs / Iu / meanTransact once ---
    thProbs = np.arange(0.60, 0.9900001, 0.02, dtype=float)
    Iu_global, meanTransact_global, _ = build_Iu_and_meanTransact(thProbs, ENL, sigma2)

    print(
        f"[dist_th] precomputed threshold tables: "
        f"thProbs.shape={thProbs.shape}, "
        f"Iu.shape={np.asarray(Iu_global).shape}, "
        f"meanTransact.shape={np.asarray(meanTransact_global).shape}",
        flush=True
    )

    # --- persist source arrays once ---
    print("[dist_th] persisting arrays...", flush=True)
    ds.img_lp = ds.img_lp.persist()
    if hasattr(ds, "land_cover"):
        ds.land_cover = ds.land_cover.persist()
    print("[dist_th] persist done", flush=True)

    # --- allocate outputs ---

    matThProb = np.zeros((nR, nC, nCats), dtype=np.float32)
    matCat = np.zeros((nR, nC, nCats), dtype=bool)
    matThI_low = np.zeros((nR, nC), dtype=np.float32)
    matStatus = np.zeros((nR, nC), dtype=bool)

    # --- batching ---
    batch_size = int(getattr(args, "distThBatchSize", 600))
    row_batch_size = max(1, batch_size // max(1, nC))

    print(
        f"[dist_th] batch_size={batch_size}, "
        f"row_batch_size={row_batch_size}",
        flush=True
    )

    c0_all = np.asarray(args.c0, dtype=np.int64).reshape(-1)
    nCols_all = np.asarray(args.nCols, dtype=np.int64).reshape(-1)
    c_min_global = int(c0_all.min())
    c_max_global = int((c0_all + nCols_all).max())

    processed = 0

    for r_start in range(0, nR, row_batch_size):
        r_end = min(r_start + row_batch_size, nR)

        r0_batch = np.asarray(args.r0[r_start:r_end], dtype=np.int64).reshape(-1)
        nRows_batch = np.asarray(args.nRows[r_start:r_end], dtype=np.int64).reshape(-1)

        r_min = int(r0_batch.min())
        r_max = int((r0_batch + nRows_batch).max())

        # print(
        #     f"[dist_th] building row-batch rows {r_start}:{r_end} "
        #     f"(windows {(r_end-r_start)*nC}) "
        #     f"bbox=r[{r_min}:{r_max}], c[{c_min_global}:{c_max_global}]",
        #     flush=True
        # )

        # --- one compute per stripe ---
        if hasattr(ds, "land_cover"):
            Istripe, LCstripe = dask.compute(
                ds.img_lp[r_min:r_max, c_min_global:c_max_global],
                ds.land_cover[r_min:r_max, c_min_global:c_max_global],
            )
        else:
            Istripe = ds.img_lp[r_min:r_max, c_min_global:c_max_global].compute()
            LCstripe = None

        Istripe = np.asarray(Istripe, dtype=np.float64)
        if LCstripe is not None:
            LCstripe = np.asarray(LCstripe)

        # --- build tasks from local numpy stripe ---
        tasks = []

        for iR in range(r_start, r_end):
            rr0 = int(args.r0[iR])
            rrn = int(args.nRows[iR])

            rr0_rel = rr0 - r_min
            row_I = Istripe[rr0_rel:rr0_rel + rrn, :]
            row_LC = LCstripe[rr0_rel:rr0_rel + rrn, :] if LCstripe is not None else None

            for iC in range(nC):
                cc0 = int(args.c0[iC])
                ccn = int(args.nCols[iC])

                cc0_rel = cc0 - c_min_global

                Iwin = row_I[:, cc0_rel:cc0_rel + ccn]
                lcwin = row_LC[:, cc0_rel:cc0_rel + ccn] if row_LC is not None else None

                tasks.append(
                    dask.delayed(dist_th_window_worker)(
                        args,
                        iR,
                        iC,
                        Iwin,
                        lcwin,
                        sigma2,
                        ENL,
                        binW,
                        thProbs,
                        Iu_global,
                        meanTransact_global,
                    )
                )

        # print(f"[dist_th] submitting row-batch rows {r_start}:{r_end}", flush=True)
        batch_results = dask.compute(*tasks)
        # print(f"[dist_th] row-batch rows {r_start}:{r_end} done", flush=True)

        # --- collect results ---
        for item in batch_results:
            if item is None:
                continue

            iR, iC, thProbOpt, catOpt, thI_low = item
            iR = int(iR)
            iC = int(iC)

            thProbOpt = np.asarray(thProbOpt, dtype=np.float32).reshape(-1)
            catOpt = np.asarray(catOpt, dtype=bool).reshape(-1)

            if thProbOpt.size != nCats:
                tmp = np.zeros((nCats,), dtype=np.float32)
                tmp[: min(nCats, thProbOpt.size)] = thProbOpt[: min(nCats, thProbOpt.size)]
                thProbOpt = tmp

            if catOpt.size != nCats:
                tmpb = np.zeros((nCats,), dtype=bool)
                tmpb[: min(nCats, catOpt.size)] = catOpt[: min(nCats, catOpt.size)]
                catOpt = tmpb

            matThProb[iR, iC, :] = thProbOpt
            matCat[iR, iC, :] = catOpt
            matThI_low[iR, iC] = np.float32(thI_low)
            matStatus[iR, iC] = True

        processed += (r_end - r_start) * nC
        if (processed % 5000 == 0) or (processed == total):
            print(
                f"[dist_th] progress: {processed}/{total} windows "
                f"({processed / total * 100:.1f}%)",
                flush=True
            )

    ds.matThProb = matThProb
    ds.matCat = matCat
    ds.matThI_low = matThI_low
    ds.matStatus = matStatus

    print("[dist_th] window-wise thresholding done.", flush=True)
    print("[dist_th] entering dist_th_postprocess...", flush=True)

    args, ds = dist_th_postprocess(args, ds)

    print("[dist_th] dist_th_postprocess finished", flush=True)
    print("[dist_th] done", flush=True)

    return args, ds


def Binary_classification(args, ds):
    with Timer('BC TIME ' + args.filepath[0]):
        rowsImage, colsImage = ds.img_lp.shape

        # -------------------------------------------------
        # Part 1) SBA windows for optimizeDist6
        # -------------------------------------------------
        args.blockSizeSBA = max(int(round(rowsImage / 50)), int(round(colsImage / 50)))

        nR, nC, r0, c0, nRows, nCols, rowsImage, colsImage = get_block(
            rowsImage, colsImage,
            block_size=args.blockSizeSBA,
            overlap=int(round(args.blockSizeSBA * args.overlapRatio / 2)),
            offset=False,
            keep_size=True
        )

        args.nR, args.nC = nR, nC
        args.r0, args.c0 = r0, c0
        args.nRows, args.nCols = nRows, nCols

        print(
            f"[Binary_classification] optimizeDist6 grid: "
            f"blockSizeSBA={args.blockSizeSBA}, "
            f"overlap={int(round(args.blockSizeSBA * args.overlapRatio / 2))}, "
            f"nR={args.nR}, nC={args.nC}",
            flush=True
        )

        # optimizeDist6 uses SBA windows
        args, ds = optimizeDist6(args, ds)

        # -------------------------------------------------
        # Part 2) dist_th windows
        # -------------------------------------------------
        args.SBAWinSize = int(getattr(args, "SBAWinSize", 100))

        dist_block_size = max(1, int(round(args.SBAWinSize / 2.0)))
        dist_overlap = max(0, int(round(args.SBAWinSize / 4.0)))

        nR, nC, r0, c0, nRows, nCols, _, _ = get_block(
            rowsImage, colsImage,
            block_size=dist_block_size,
            overlap=dist_overlap,
            offset=False,
            keep_size=True
        )

        args.nR, args.nC = nR, nC
        args.r0, args.c0 = r0, c0
        args.nRows, args.nCols = nRows, nCols

        print(
            f"[Binary_classification] dist_th grid: "
            f"SBAWinSize={args.SBAWinSize}, "
            f"block_size={dist_block_size}, "
            f"overlap={dist_overlap}, "
            f"nR={args.nR}, nC={args.nC}",
            flush=True
        )

        args, ds = dist_th(args, ds)

        if bool(args.tempfiles):
            for itemp in args.tempfiles:
                print('closing ' + itemp.name)
                itemp.close()

    return args, ds
