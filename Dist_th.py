import numpy as np
import os
import dask
from Statistics import KLD_chi2_frm_sample, validate_gmm, get_block, build_Iu_and_meanTransact
from utils import read_window_from_ds
from scipy.stats import chi2
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


def solve4GMM(nmodes: int, I_linear: np.ndarray, mu0=None, std0=None, pc0=None, fileVal: str | None = None):
    """
    Port of solve4GMM.m
    Input I_linear is LINEAR intensity. Internally convert to dB then standardize and fit GMM.
    Returns mu/sigma/th in dB, pc weights, RH ratio, and AIC.
    """
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm

    I_linear = np.asarray(I_linear, dtype=float).reshape(-1)
    I_linear = I_linear[np.isfinite(I_linear) & (I_linear > 0)]
    if I_linear.size < 10:
        return (
            np.full((nmodes,), np.nan),
            np.full((nmodes,), np.nan),
            np.full((nmodes,), np.nan),
            np.full((max(nmodes - 1, 1),), np.nan),
            np.full((max(nmodes - 1, 1),), np.nan),
            np.nan,
        )

    I_db = 10.0 * np.log10(I_linear)

    I_mean = float(np.mean(I_db))
    I_std = float(np.std(I_db, ddof=0))
    if not np.isfinite(I_std) or I_std <= 0:
        return (
            np.full((nmodes,), np.nan),
            np.full((nmodes,), np.nan),
            np.full((nmodes,), np.nan),
            np.full((max(nmodes - 1, 1),), np.nan),
            np.full((max(nmodes - 1, 1),), np.nan),
            np.nan,
        )

    I_stdzd = (I_db - I_mean) / I_std
    X = I_stdzd.reshape(-1, 1)

    # init handling
    mu0 = np.asarray(mu0).reshape(-1) if mu0 is not None else np.array([])
    std0 = np.asarray(std0).reshape(-1) if std0 is not None else np.array([])
    pc0 = np.asarray(pc0).reshape(-1) if pc0 is not None else np.array([])
    has_init = (mu0.size > 0) and (std0.size > 0) and (pc0.size > 0)

    gmm = GaussianMixture(
        n_components=nmodes,
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=1000,
        n_init=1 if has_init else 20,
        random_state=0,
    )

    if has_init:
        means_init = ((mu0 - I_mean) / I_std).reshape(-1, 1)
        weights_init = pc0 / np.sum(pc0) if np.sum(pc0) > 0 else None
        gmm.means_init = means_init
        gmm.weights_init = weights_init

    gmm.fit(X)

    mu_std = gmm.means_.reshape(-1)
    sigma_std = np.sqrt(np.maximum(gmm.covariances_.reshape(-1), 1e-12))
    pc = gmm.weights_.reshape(-1)
    AIC = float(gmm.aic(X))

    # sort by mean
    idx = np.argsort(mu_std)
    mu_std = mu_std[idx]
    sigma_std = sigma_std[idx]
    pc = pc[idx]

    # histogram peakDensity on dB original scale
    hist_vals, _ = np.histogram(I_db, bins=50, density=True)
    peakDensity = float(np.max(hist_vals)) if hist_vals.size > 0 else 1.0

    thresholds_std = np.zeros((max(nmodes - 1, 0),), dtype=float)
    RH = np.zeros((max(nmodes - 1, 0),), dtype=float)

    for i in range(nmodes - 1):
        mu1 = float(mu_std[i])
        mu2 = float(mu_std[i + 1])
        s1 = float(sigma_std[i])
        s2 = float(sigma_std[i + 1])
        w1 = float(pc[i])
        w2 = float(pc[i + 1])

        def f1(x):
            return w1 * norm.pdf(x, mu1, s1)

        def f2(x):
            return w2 * norm.pdf(x, mu2, s2)

        def eq(x):
            return abs(f1(x) - f2(x))

        res = minimize_scalar(eq, bounds=(mu1, mu2), method="bounded")
        thr = float(res.x) if res.success else 0.5 * (mu1 + mu2)
        thresholds_std[i] = thr

        thDensity = (f1(thr) + f2(thr)) / I_std
        RH[i] = float(thDensity / peakDensity) if peakDensity > 0 else np.nan

    mu = mu_std * I_std + I_mean
    sigma = sigma_std * I_std
    th = thresholds_std * I_std + I_mean

    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    pc = pc.reshape(-1)
    th = th.reshape(-1)
    RH = RH.reshape(-1)

    # optional plot saving is not required for now; skip or keep minimal if you want.

    return mu, sigma, pc, th, RH, AIC


def thresholdGMM_Dual(
    I_linear: np.ndarray,
    probW=None,
    thProb: float | None = None,
    waterOnly: bool = True,
    thPC=(0.02, 0.98),
    thAD=0.9,
    thRH=1.0,
    thDist2Mean=0.1,
    lowestMean: float | None = None,   # dB
):
    """
    Pure-array port of thresholdGMM_Dual.m (wind_inclusive usage).

    Returns:
      th (float): threshold in dB (NaN if invalid, inf if one-mode/global)
      pc (2,)
      mu (2,) in dB
      sigma (2,) in dB
      RH (float)
      AD (float)
      isValid (bool)
      isOneMode (bool)
    """
    isOneMode = False

    I = np.asarray(I_linear, dtype=float)
    basinMask0 = np.isfinite(I) & (I > 0)

    if probW is not None and thProb is not None and np.isfinite(thProb):
        probW = np.asarray(probW, dtype=float)
        basinMask = basinMask0 & (probW >= thProb)
    else:
        basinMask = basinMask0

    if not waterOnly:
        rows, cols = np.where(basinMask)
        if rows.size == 0:
            samples = I[basinMask0]
        else:
            minRow, maxRow = int(rows.min()), int(rows.max())
            minCol, maxCol = int(cols.min()), int(cols.max())
            hH = int(round((maxRow - minRow) / 2))
            hW = int(round((maxCol - minCol) / 2))
            rowTop = max(0, minRow - hH)
            rowBtm = min(I.shape[0] - 1, maxRow + hH)
            colLeft = max(0, minCol - hW)
            colRight = min(I.shape[1] - 1, maxCol + hW)

            I0 = I[rowTop:rowBtm + 1, colLeft:colRight + 1]
            samples = I0[np.isfinite(I0) & (I0 > 0)]
    else:
        samples = I[basinMask]

    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples) & (samples > 0)]

    if samples.size < 100:
        th = np.nan
        pc = np.array([np.nan, np.nan], dtype=float)
        mu = np.array([np.nan, np.nan], dtype=float)
        sigma = np.array([np.nan, np.nan], dtype=float)
        RH = np.nan
        AD = np.nan
        isValid = False
        return th, pc, mu, sigma, RH, AD, isValid, isOneMode

    # fit 2-mode
    mu, sigma, pc, th_arr, RH_arr, _AIC = solve4GMM(2, samples, [], [], [], None)

    th = float(np.asarray(th_arr).reshape(-1)[0]) if np.asarray(th_arr).size > 0 else np.nan
    RH = float(np.asarray(RH_arr).reshape(-1)[0]) if np.asarray(RH_arr).size > 0 else np.nan

    # lowestMean gate (MATLAB: th<-30 or mu(2)<lowestMean => isOneMode, th=inf)
    if lowestMean is not None:
        if (np.isfinite(th) and th < -30) or (mu.size >= 2 and float(mu[1]) < float(lowestMean)):
            isValid = False
            isOneMode = True
            th = float("inf")
            pc = np.array([np.nan, np.nan], dtype=float)
            mu = np.array([np.nan, np.nan], dtype=float)
            sigma = np.array([np.nan, np.nan], dtype=float)
            RH = np.nan
            AD = np.nan
            return th, pc, mu, sigma, RH, AD, isValid, isOneMode

    # validate 2-mode
    isValid, AD = validate_gmm(
        RH=RH,
        pc=pc,
        mu=mu,
        sigma=sigma,
        Iu=th,
        thPC=thPC,
        thAD=thAD,
        thRH=thRH,
        thDist2Mean=thDist2Mean,
        separately=True
    )

    if not isValid:
        # try 3-mode then prune fake mode/threshold
        mu3, sigma3, pc3, th3, RH3, _AIC3 = solve4GMM(3, samples, [], [], [], None)

        mu3 = np.asarray(mu3, dtype=float).reshape(-1)
        sigma3 = np.asarray(sigma3, dtype=float).reshape(-1)
        pc3 = np.asarray(pc3, dtype=float).reshape(-1)
        th3 = np.asarray(th3, dtype=float).reshape(-1)
        RH3 = np.asarray(RH3, dtype=float).reshape(-1)

        maskFakeMode = np.zeros(mu3.shape, dtype=bool)
        maskFakeTh = np.zeros(th3.shape, dtype=bool)

        for it in range(th3.size):
            dist = np.abs(mu3 - th3[it])
            idx = int(np.argmin(dist))
            if dist[idx] < 0.5:
                maskFakeMode[idx] = True
                maskFakeTh[it] = True

        mu3 = mu3[~maskFakeMode]
        sigma3 = sigma3[~maskFakeMode]
        pc3 = pc3[~maskFakeMode]
        th3 = th3[~maskFakeTh]
        RH3 = RH3[~maskFakeTh]

        if pc3.size > 0 and np.sum(pc3) > 0:
            pc3 = pc3 / np.sum(pc3)

        if th3.size > 1:
            th3 = th3[:1]
            mu3 = mu3[:2]
            sigma3 = sigma3[:2]
            RH3 = RH3[:1]
            pc3 = pc3[:2]
            pc3 = pc3 / np.sum(pc3) if np.sum(pc3) > 0 else pc3

        if th3.size == 1 and mu3.size == 2 and np.isfinite(th3[0]):
            th_candidate = float(th3[0])

            if lowestMean is not None:
                if (th_candidate < -30) or (mu3[1] < float(lowestMean)):
                    isValid = False
                    isOneMode = True
                    th = float("inf")
                    pc = np.array([np.nan, np.nan], dtype=float)
                    mu = np.array([np.nan, np.nan], dtype=float)
                    sigma = np.array([np.nan, np.nan], dtype=float)
                    RH = np.nan
                    AD = np.nan
                    return th, pc, mu, sigma, RH, AD, isValid, isOneMode

            RH_candidate = float(np.asarray(RH3).reshape(-1)[0]) if RH3.size > 0 else np.nan
            isValid2, AD2 = validate_gmm(
                RH=RH_candidate,
                pc=pc3,
                mu=mu3,
                sigma=sigma3,
                Iu=th_candidate,
                thPC=thPC,
                thAD=thAD,
                thRH=thRH,
                thDist2Mean=thDist2Mean,
                separately=True
            )

            if isValid2:
                th = th_candidate
                mu = mu3
                sigma = sigma3
                pc = pc3
                RH = RH_candidate
                AD = AD2
                isValid = True

        if not isValid:
            th = np.nan
            pc = np.array([np.nan, np.nan], dtype=float)
            mu = np.array([np.nan, np.nan], dtype=float)
            sigma = np.array([np.nan, np.nan], dtype=float)
            RH = np.nan
            AD = np.nan

    return float(th), np.asarray(pc, dtype=float), np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float), float(RH), float(AD), bool(isValid), bool(isOneMode)


def findThProb(
    runStyle: str,
    thProbs,
    Iu,
    I1,
    sigma2,
    ENL,
    binWidth,
    meanTransact,
    thKLD,
    thR,
    thNSE,
    thDist2Mean: float,
    thSampleSize: int,
    isPW: bool,
    hasPW: bool,
    thPC=(0.02, 0.98),
    thAD=0.9,
    thRH=(1.0, 1.0),
):
    """
    Port of MATLAB findThProb.m

    Parameters
    ----------
    runStyle : str
        'wind_inclusive' or 'glacial_exclusive' (glacial_exclusive still left TODO)
    thProbs : (nProbs,)
    Iu : (nProbs, nCat)
        Threshold intensities in LINEAR domain for each probability and class.
    I1 : (N,)
        Window samples in LINEAR domain.
    sigma2 : (nCat,)
    ENL : (nCat,)
    binWidth : (nCat,)
    meanTransact : (nProbs,)
        Mean of truncated lowest-class distribution (LINEAR domain).
    thKLD, thR, thNSE : scalar or len-2 sequence
        MATLAB worker uses len-2 thresholds. This function supports both.
    thDist2Mean : float
    thSampleSize : int
    isPW, hasPW : bool
    thPC : tuple(float, float)
    thAD : float
    thRH : tuple(float, float)

    Returns
    -------
    found : bool
    thProb_out : (nCat,)
    catSel_out : (nCat,)
    thI : float
        Low-class threshold in LINEAR domain.
        0   -> no lower class found
        inf -> global bimodality applies
    """
    thProbs = np.asarray(thProbs, dtype=float).reshape(-1)
    I1 = np.asarray(I1, dtype=float).reshape(-1)
    I1 = I1[np.isfinite(I1) & (I1 > 0)]

    sigma2 = np.asarray(sigma2, dtype=float).reshape(-1)
    ENL = np.asarray(ENL, dtype=float).reshape(-1)
    binWidth = np.asarray(binWidth, dtype=float).reshape(-1)
    meanTransact = np.asarray(meanTransact, dtype=float).reshape(-1)

    Iu = np.asarray(Iu, dtype=float)
    if Iu.ndim != 2:
        raise ValueError("Iu must be 2D with shape (len(thProbs), nCat).")

    nCat = int(sigma2.size)
    if Iu.shape != (thProbs.size, nCat):
        raise ValueError(f"Iu shape mismatch: expected {(thProbs.size, nCat)}, got {Iu.shape}")

    # support scalar or 2-element thresholds
    thKLD = np.asarray(thKLD, dtype=float).reshape(-1)
    thR = np.asarray(thR, dtype=float).reshape(-1)
    thNSE = np.asarray(thNSE, dtype=float).reshape(-1)

    if thKLD.size == 1:
        thKLD = np.repeat(thKLD, 2)
    if thR.size == 1:
        thR = np.repeat(thR, 2)
    if thNSE.size == 1:
        thNSE = np.repeat(thNSE, 2)

    if thKLD.size < 2 or thR.size < 2 or thNSE.size < 2:
        raise ValueError("thKLD, thR, thNSE must be scalar or length-2.")

    thPC = tuple(thPC)
    thRH = tuple(thRH)

    catSel = np.zeros((nCat, 2), dtype=bool)
    thProb = np.zeros((nCat, 2), dtype=float)
    thI = 0.0

    low_class = int(np.nanargmin(sigma2))

    bimodalTested = False

    for cat in range(nCat):
        for t in range(thProbs.size - 1, -1, -1):
            cut = Iu[t, cat]
            pol_water = I1[I1 < cut]

            if pol_water.size < int(thSampleSize):
                break

            # bimodality test only for the lowest class and larger thProb
            if (cat == low_class) and (thProbs[t] > 0.6):
                real_mean = float(np.mean(pol_water))

                # IMPORTANT: only test when meanTransact is finite
                if (
                    np.isfinite(meanTransact[t])
                    and (real_mean < float(meanTransact[t]))
                    and (not bimodalTested)
                ):
                    bimodalTested = True

                    if runStyle == "wind_inclusive":
                        if isPW:
                            thI = float("inf")
                        else:
                            ref_low_mean_db = 10.0 * np.log10(sigma2[cat])

                            th_db, pc, muGMM, sigmaGMM, RH_gmm, AD_gmm, is_valid, isOneMode = thresholdGMM_Dual(
                                I_linear=I1,
                                probW=None,
                                thProb=None,
                                waterOnly=True,
                                thPC=thPC,
                                thAD=thAD,
                                thRH=thRH[0],
                                thDist2Mean=thDist2Mean,
                                lowestMean=ref_low_mean_db,
                            )

                            if is_valid:
                                muGMM = np.asarray(muGMM, dtype=float).reshape(-1)
                                if muGMM.size > 0 and muGMM[0] < ref_low_mean_db + 3.8:
                                    # MATLAB: thI = 10^(min(thI)/10)
                                    thI = 10.0 ** (float(th_db) / 10.0)
                                elif hasPW:
                                    thI = float("inf")
                                else:
                                    thI = 0.0
                            elif isOneMode:
                                thI = float("inf")
                            else:
                                thI = 0.0

                    elif runStyle == "glacial_exclusive":
                        # TODO: keep empty for now
                        pass
                    else:
                        raise ValueError(f"Unknown runStyle={runStyle}")

            # scalar distribution test
            KLD, _, _, R, NSE = KLD_chi2_frm_sample(
                I=pol_water,
                sigma2=float(sigma2[cat]),
                interval=float(binWidth[cat]),
                n=float(ENL[cat]),
                p=float(thProbs[t]),
                trailing=3,
                real2theo=False,
                out_figure=None,
            )

            pass_vec = np.array([
                (KLD < thKLD[0]) and (R > thR[0]) and (NSE > thNSE[0]),
                (KLD < thKLD[1]) and (R > thR[1]) and (NSE > thNSE[1]),
            ], dtype=bool)

            catSel[cat, 0] = pass_vec[0]
            catSel[cat, 1] = bool(catSel[cat, 1] or pass_vec[1])
            thProb[cat, 0] = float(thProbs[t]) if pass_vec[0] else 0.0

            if (thProb[cat, 1] == 0.0) and pass_vec[1]:
                thProb[cat, 1] = float(thProbs[t])

            if pass_vec[0]:
                break

    mask1 = thProb[:, 0] > 0
    mask2 = (thProb[:, 1] > 0) & (thProb[:, 0] <= 0)
    thProb_out = thProb[:, 0] * mask1 + thProb[:, 1] * mask2

    catSel_out = catSel[:, 0] | catSel[:, 1]
    found = bool(np.any(thProb_out > 0) or (thI > 0))

    return found, thProb_out, catSel_out, float(thI)


def dist_th_window_worker(args, iR, iC, Iwin, lcwin, sigma2, ENL, binW):
    """
    Return: iR, iC, thProbOpt(nCats,), catOpt(nCats,), thI_low(float linear)
    """
    # --- materialize window locally (worker side) ---
    if hasattr(Iwin, "compute"):
        Iw2d = Iwin.compute()
    else:
        Iw2d = Iwin
    Iw2d = np.asarray(Iw2d, dtype=np.float64)

    basinMask = np.isfinite(Iw2d)
    Iw = Iw2d[basinMask]
    Iw = Iw[np.isfinite(Iw) & (Iw > 0)]  # MATLAB basinMask=~isnan(I1); later water-only uses >0

    # if too few pixels, early return
    nCats = int(np.asarray(sigma2).size)
    if Iw.size < max(50, int(getattr(args, "thMinSampleSize", 500) // 10)):
        return int(iR), int(iC), np.zeros((nCats,), float), np.zeros((nCats,), bool), 0.0

    # --- LCC / PW ratio ---
    rPW = 0.0
    isPW = False
    hasPW = False
    if lcwin is not None:
        if hasattr(lcwin, "compute"):
            LCC2d = lcwin.compute()
        else:
            LCC2d = lcwin
        LCC2d = np.asarray(LCC2d)

        LCCw = LCC2d[basinMask]
        # MATLAB: rPW=sum(ismember(LCCw,codePW))/length(LCCw)
        codePW = int(getattr(args, "codePW", 80))
        if LCCw.size > 0:
            rPW = float(np.sum(LCCw == codePW) / LCCw.size)

        # MATLAB: isPW=rPW>rThPW(2); hasPW=rPW>rThPW(1)
        rThPW = getattr(args, "distThRPW", (0.01, 0.9))
        hasPW = rPW > float(rThPW[0])
        isPW = rPW > float(rThPW[1])

    # --- thProbs / Iu / meanTransact (match MATLAB constants) ---
    thProbs = np.arange(0.60, 0.9900001, 0.02, dtype=float)
    Iu, meanTransact, _ = build_Iu_and_meanTransact(thProbs, ENL, sigma2)

    # --- findThProb ---
    found, thProbOpt, catOpt, thI_low = findThProb(
        runStyle=getattr(args, "runStyle", "wind_inclusive"),
        thProbs=thProbs,
        Iu=Iu,
        I1=Iw,                 # 1D samples in window
        sigma2=sigma2,
        ENL=ENL,
        binWidth=binW,
        meanTransact=meanTransact,
        thKLD=float(getattr(args, "distThKLD", (0.1, 0.2))),
        thR=float(getattr(args, "distThR", (0.9, 0.7))),
        thNSE=float(getattr(args, "distThNSE", (0.80, 0.40))),
        thDist2Mean=float(getattr(args, "distThDist2Mean", 0.1)),
        thSampleSize=int(getattr(args, "distThSampleSize", 50)),
        isPW=bool(isPW),
        hasPW=bool(hasPW),
        thPC=getattr(args, "distThPC", (0.02, 0.98)),
        thAD=float(getattr(args, "distThAD", 0.9)),
        thRH=getattr(args, "distThRH", (1.0, 1.0)),
    )

    # even if not found, we still return arrays (MATLAB stores zeros)
    thProbOpt = np.asarray(thProbOpt, dtype=float).reshape(-1)
    catOpt = np.asarray(catOpt, dtype=bool).reshape(-1)

    if thProbOpt.size != nCats:
        tmp = np.zeros((nCats,), float)
        tmp[: min(nCats, thProbOpt.size)] = thProbOpt[: min(nCats, thProbOpt.size)]
        thProbOpt = tmp
    if catOpt.size != nCats:
        tmpb = np.zeros((nCats,), bool)
        tmpb[: min(nCats, catOpt.size)] = catOpt[: min(nCats, catOpt.size)]
        catOpt = tmpb

    return int(iR), int(iC), thProbOpt, catOpt, float(thI_low)


def _ensure_refmeta(args):
    """
    Prefer args.refmeta (serializable) over keeping an open dataset handle.
    If missing, fall back to reading from args.filepath[0] (one-time IO).
    """
    if getattr(args, "refmeta", None) is None:
        with rasterio.open(args.filepath[0]) as src:
            args.refmeta = src.meta.copy()


def _build_coarse_transform_from_blocks(args, refmeta):
    """
    MATLAB builds geoTransMat based on block centers.
    Here we compute coarse pixel size from centers of first two blocks in each direction.
    args.r0/c0/nRows/nCols are 1D "process window" vectors (you already normalized to col=1).
    """
    tr = refmeta["transform"]

    r0 = np.asarray(args.r0).reshape(-1)
    c0 = np.asarray(args.c0).reshape(-1)
    nRows = np.asarray(args.nRows).reshape(-1)
    nCols = np.asarray(args.nCols).reshape(-1)

    nR = int(args.nR)
    nC = int(args.nC)

    # center of first cell (0,0)
    r_center00 = float(r0[0] + nRows[0] / 2.0)
    c_center00 = float(c0[0] + nCols[0] / 2.0)
    x00, y00 = rasterio.transform.xy(tr, r_center00, c_center00, offset="center")

    # estimate coarse resolution using neighbor centers (fallback to ref pixel scale if only 1 cell)
    if nC >= 2:
        c_center01 = float(c0[1] + nCols[1] / 2.0)
        x01, _ = rasterio.transform.xy(tr, r_center00, c_center01, offset="center")
        resX = float(x01 - x00)
    else:
        resX = float(tr.a)

    if nR >= 2:
        r_center10 = float(r0[1] + nRows[1] / 2.0)
        _, y10 = rasterio.transform.xy(tr, r_center10, c_center00, offset="center")
        resY = float(y10 - y00)
    else:
        resY = float(tr.e)

    # origin is top-left corner of coarse pixel [0,0]
    x0 = float(x00 - resX / 2.0)
    y0 = float(y00 - resY / 2.0)

    return Affine(resX, 0.0, x0, 0.0, resY, y0)


def _write_tif(path, arr2d, transform, crs, nodata, dtype="float32"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(arr2d.shape[0]),
        "width": int(arr2d.shape[1]),
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(dtype, copy=False), 1)


def _resample_to_ref(src_path, dst_path, refmeta, resampling=Resampling.bilinear):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    ref_transform = refmeta["transform"]
    ref_crs = refmeta["crs"]
    ref_h = int(refmeta["height"])
    ref_w = int(refmeta["width"])
    ref_nodata = refmeta.get("nodata", 255)

    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

        dst_arr = np.full((ref_h, ref_w), ref_nodata, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=ref_nodata,
            resampling=resampling,
        )

    out_profile = refmeta.copy()
    out_profile.update(driver="GTiff", count=1, dtype="float32", nodata=ref_nodata, compress="deflate", tiled=True)
    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(dst_arr.astype("float32", copy=False), 1)


def _build_matThI(args, ds):
    """
    Build coarse threshold-intensity maps (linear domain).
    MATLAB: nClasses = nCats + 1; last class is matThI_low.
    For each cat, thI = chi2.ppf(thProbOpt, df=2*n) * sigma2 / (2*n).
    """
    matThProb = np.asarray(ds.matThProb, dtype=float)     # (nR,nC,nCats)
    matThI_low = np.asarray(ds.matThI_low, dtype=float)   # (nR,nC)

    sigma2 = np.asarray(ds.sigma2, dtype=float).reshape(-1)
    ENL = np.asarray(ds.n_est, dtype=float).reshape(-1)

    nR, nC, nCats = matThProb.shape
    nClasses = nCats + 1

    matThI = np.full((nR, nC, nClasses), np.nan, dtype=float)

    # per-class threshold from thProb
    for k in range(nCats):
        p = matThProb[:, :, k]
        # valid probabilities
        valid = np.isfinite(p) & (p > 0) & (p < 1)
        if not np.any(valid):
            continue

        Xu = np.full((nR, nC), np.nan, dtype=float)
        Xu[valid] = chi2.ppf(p[valid], df=2.0 * ENL[k])
        matThI[:, :, k] = Xu * sigma2[k] / (2.0 * ENL[k])

    # last class = low class threshold map
    matThI[:, :, -1] = matThI_low

    # MATLAB: conservatively suppress any thILow to below median (50th percentile)
    low = matThI[:, :, -1]
    mask = np.isfinite(low) & (low > 0) & (~np.isinf(low))
    if np.any(mask):
        prc50 = float(np.percentile(low[mask], 50))
        low2 = low.copy()
        low2[np.isinf(low2)] = prc50
        low2[(low2 >= prc50)] = prc50
        matThI[:, :, -1] = low2

    return matThI


def dist_th_postprocess(args, ds, nodata=255):
    """
    Python port of MATLAB dist_th (post ~200 lines):
      1) build matThI (coarse threshold maps in linear domain)
      2) write coarse tif and resample to fine tif
      3) generate initial classification raster (Cat_init.tif) by scanning full-res blocks
      4) collect pol_water samples per class (like MATLAB pol_water{idx(cat)})
    """

    _ensure_refmeta(args)
    refmeta = args.refmeta
    rowsImage = int(refmeta["height"])
    colsImage = int(refmeta["width"])

    # ---- output file roots (aligned with your dup_args layout) ----
    fileThCoarse_root = getattr(args, "fileThCoarse_root", os.path.join(args.dirTempPol, "ThCoarse_"))
    fileThFine_root = getattr(args, "fileThFine_root", os.path.join(args.dirTempPol, "ThFine_"))
    fileCatInit = getattr(args, "fileCatInit", os.path.join(args.dirOut, "Cat_init.tif"))

    # ---- 1) build matThI ----
    matThI = _build_matThI(args, ds)  # (nR,nC,nClasses)
    nR, nC, nClasses = matThI.shape

    # ---- 2) write coarse threshold maps and resample to full-res ----
    coarse_transform = _build_coarse_transform_from_blocks(args, refmeta)
    crs = refmeta["crs"]

    for cat in range(1, nClasses + 1):
        coarse_path = f"{fileThCoarse_root}{cat}.tif"
        fine_path = f"{fileThFine_root}{cat}.tif"

        print(f"interpolate the threshold map for class {cat}...")

        arr = matThI[:, :, cat - 1]
        # write nodata as 255 to match your convention (float raster with nodata=255 is ok)
        out = np.where(np.isfinite(arr), arr.astype(np.float32, copy=False), np.float32(nodata))

        _write_tif(
            coarse_path,
            out,
            transform=coarse_transform,
            crs=crs,
            nodata=nodata,
            dtype="float32",
        )
        _resample_to_ref(coarse_path, fine_path, refmeta, resampling=Resampling.bilinear)

    # ---- 3) scan full-res and produce initial classification ----
    # MATLAB: C(end+1)=inf; idx=sort(C,'ascend');
    C_ext = np.concatenate([np.asarray(ds.sigma2, dtype=float).reshape(-1), [np.inf]])
    idx_sorted = np.argsort(C_ext) + 1  # 1-based class indices used in filenames and output

    # open output classification raster (uint8)
    out_meta = refmeta.copy()
    out_meta.update(driver="GTiff", count=1, dtype="uint8", nodata=nodata, compress="deflate", tiled=True)
    os.makedirs(os.path.dirname(fileCatInit), exist_ok=True)

    # collect pol_water like MATLAB cell array (index by class id 1..nClasses)
    pol_water_lists = [[] for _ in range(nClasses + 1)]

    # block over full image like MATLAB GetBlock(fileInt,1000,0,false,false)
    nR2, nC2, r0b, c0b, nRowsb, nColsb, _, _ = get_block(
        rowsImage, colsImage,
        block_size=1000,
        overlap=0,
        offset=False,
        keep_size=False,
    )

    with rasterio.open(fileCatInit, "w", **out_meta) as dst:
        total = int(nR2) * int(nC2)
        done = 0

        for iR in range(int(nR2)):
            rr0 = int(r0b[iR]); rrn = int(nRowsb[iR])

            for iC in range(int(nC2)):
                done += 1
                cc0 = int(c0b[iC]); ccn = int(nColsb[iC])

                if getattr(args, "showProg", False):
                    prog = done / total
                    print(f"\r{int(prog*100)}%...", end="")

                # intensity block
                Iblk = ds.img_lp[rr0:rr0 + rrn, cc0:cc0 + ccn]
                if hasattr(Iblk, "compute"):
                    Iblk = Iblk.compute()
                Iblk = np.asarray(Iblk, dtype=np.float64)

                basinMask = np.isfinite(Iblk) & (Iblk > 0)

                # waterCat: nodata outside basin; 0 inside basin initially
                waterCat = np.full((rrn, ccn), nodata, dtype=np.uint8)
                waterCat[basinMask] = 0

                for rank in range(nClasses):
                    cls = int(idx_sorted[rank])  # 1..nClasses
                    fine_path = f"{fileThFine_root}{cls}.tif"

                    with rasterio.open(fine_path) as thsrc:
                        th_blk = thsrc.read(
                            1,
                            window=Window(cc0, rr0, ccn, rrn),
                            boundless=True,
                            fill_value=nodata,
                        ).astype(np.float64)

                    # MATLAB: waterMask = I1 < thI; waterCat(waterMask & waterCat==0)=idx(cat)
                    waterMask = basinMask & np.isfinite(th_blk) & (th_blk != nodata) & (Iblk < th_blk)
                    place = waterMask & (waterCat == 0)
                    if np.any(place):
                        waterCat[place] = np.uint8(cls)

                        # collect samples
                        px = Iblk[waterCat == cls]
                        if px.size > 0:
                            pol_water_lists[cls].append(px.reshape(-1))

                dst.write(waterCat, 1, window=Window(cc0, rr0, ccn, rrn))

        if getattr(args, "showProg", False):
            print("\nDone")

    # flatten pol_water
    pol_water = [np.array([], dtype=np.float64) for _ in range(nClasses + 1)]
    for cls in range(1, nClasses + 1):
        if pol_water_lists[cls]:
            pol_water[cls] = np.concatenate(pol_water_lists[cls], axis=0)

    ds.matThI = matThI
    ds.idx_sorted = idx_sorted
    ds.C_ext = C_ext
    ds.pol_water = pol_water
    ds.fileCatInit = fileCatInit
    ds.fileThCoarse_root = fileThCoarse_root
    ds.fileThFine_root = fileThFine_root

    print(f"[dist_th] wrote threshold maps and initial classification -> {fileCatInit}")
    return args, ds
