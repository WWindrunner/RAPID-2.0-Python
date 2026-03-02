import numpy as np
import dask
from Statistics import KLD_chi2_frm_sample, validate_gmm
from utils import read_window_from_ds
from scipy.stats import chi2


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
    thKLD: float,
    thR: float,
    thNSE: float,
    thDist2Mean: float,
    thSampleSize: int,
    isPW: bool,
    hasPW: bool,
):
    """
    Port of findThProb.m (wind_inclusive implemented; glacial_exclusive TODO).

    Returns:
      found (bool), thProb_out (nCat,), catSel_out (nCat,), thI (float linear)
    """
    thProbs = np.asarray(thProbs, dtype=float).reshape(-1)
    I1 = np.asarray(I1, dtype=float).reshape(-1)
    I1 = I1[np.isfinite(I1)]

    sigma2 = np.asarray(sigma2, dtype=float).reshape(-1)
    ENL = np.asarray(ENL, dtype=float).reshape(-1)
    binWidth = np.asarray(binWidth, dtype=float).reshape(-1)
    meanTransact = np.asarray(meanTransact, dtype=float).reshape(-1)

    Iu = np.asarray(Iu, dtype=float)
    if Iu.ndim != 2:
        raise ValueError("Iu must be 2D (len(thProbs), nCat).")

    nCat = int(sigma2.size)
    if Iu.shape != (thProbs.size, nCat):
        raise ValueError(f"Iu shape mismatch: expected {(thProbs.size, nCat)}, got {Iu.shape}")

    catSel = np.zeros((nCat, 2), dtype=bool)
    thProb = np.zeros((nCat, 2), dtype=float)
    thI = 0.0

    low_class = int(np.nanargmin(sigma2))

    # thresholds for bimodality (as in MATLAB)
    thPC = (0.02, 0.98)
    thAD = 0.9
    thRH = (1.0, 1.0)
    bimodalTested = False

    for cat in range(nCat):
        for t in range(thProbs.size - 1, -1, -1):
            cut = Iu[t, cat]
            pol_water = I1[I1 < cut]

            if pol_water.size < int(thSampleSize):
                break

            if (cat == low_class) and (thProbs[t] > 0.6):
                real_mean = float(np.mean(pol_water))

                if (real_mean < float(meanTransact[t])) and (not bimodalTested):
                    bimodalTested = True

                    if runStyle == "wind_inclusive":
                        if isPW:
                            thI = float("inf")
                        else:
                            ref_low_mean_db = 10.0 * np.log10(sigma2[cat])

                            th_db, pc, muGMM, sigmaGMM, RH_gmm, AD_gmm, is_valid, isOneMode = thresholdGMM_Dual(
                                I_linear=I1,               # LINEAR
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
                                    thI = 10.0 ** (float(np.min([th_db])) / 10.0)  # dB -> linear
                                elif hasPW:
                                    thI = float("inf")
                                else:
                                    thI = 0.0
                            elif isOneMode:
                                thI = float("inf")
                            else:
                                thI = 0.0

                    elif runStyle == "glacial_exclusive":
                        # TODO: as requested
                        pass
                    else:
                        raise ValueError(f"Unknown runStyle={runStyle}")

            # distribution test (scalar version)
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

            pass_mask = (KLD < thKLD) and (R > thR) and (NSE > thNSE)

            # keep MATLAB 2-col structure but driven by same boolean for now
            catSel[cat, 0] = bool(pass_mask)
            catSel[cat, 1] = bool(catSel[cat, 1] or pass_mask)

            thProb[cat, 0] = float(thProbs[t]) if pass_mask else 0.0
            if (thProb[cat, 1] == 0.0) and pass_mask:
                thProb[cat, 1] = float(thProbs[t])

            if pass_mask:
                break

    mask1 = thProb[:, 0] > 0
    mask2 = (thProb[:, 1] > 0) & (thProb[:, 0] <= 0)
    thProb_out = thProb[:, 0] * mask1 + thProb[:, 1] * mask2
    catSel_out = catSel[:, 0] | catSel[:, 1]
    found = bool(np.any(thProb_out > 0) or (thI > 0))

    return found, thProb_out, catSel_out, float(thI)


def dist_th_window_worker(args, ds, iR: int, iC: int):
    """
    One SBA window task. Returns:
      iR, iC, thProbOpt(nCat,), catOpt(nCat,), thI_low(float)
    """
    r0 = int(args.r0[iR])
    c0 = int(args.c0[iC])
    nr = int(args.nRows[iR])
    nc = int(args.nCols[iC])

    # ---- read window (numpy) ----
    Iwin = read_window_from_ds(ds, "img_lp", r0, c0, nr, nc)
    I1 = Iwin.reshape(-1)
    I1 = I1[np.isfinite(I1) & (I1 > 0)]

    # land cover window (optional but recommended)
    lc_win = None
    if hasattr(ds, "land_cover"):
        lc_win = read_window_from_ds(ds, "land_cover", r0, c0, nr, nc)
        lc_flat = lc_win.reshape(-1)
        lc_flat = lc_flat[np.isfinite(lc_flat)]
    else:
        lc_flat = np.array([], dtype=float)

    nCat = int(len(ds.sigma2))

    # ---- build candidate thProbs ----
    thProbs = np.arange(
        float(args.thProbMin),
        float(args.thProbMax) + float(args.dThProb) / 2.0,
        float(args.dThProb),
        dtype=float,
    )
    nT = int(thProbs.size)

    # ---- build Iu(t,cat) from chi2 ppf ----
    sigma2 = np.asarray(ds.sigma2, dtype=float).reshape(-1)
    ENL = np.asarray(ds.n_est, dtype=float).reshape(-1)
    binWidth = np.asarray(ds.binWidth, dtype=float).reshape(-1)

    Iu = np.zeros((nT, nCat), dtype=float)
    for t in range(nT):
        df = 2.0 * ENL
        Xu = chi2.ppf(thProbs[t], df=df)
        Iu[t, :] = Xu * sigma2 / (2.0 * ENL)

    # ---- placeholders / rough flags ----
    meanTransact = np.full((nT,), np.inf, dtype=float)  # TODO: replace with real logic

    codePW = int(getattr(args, "codePW", 80))
    hasPW = bool(lc_flat.size > 0 and np.any(lc_flat == codePW))
    # "isPW" (pure water class) - rough: >=95% water pixels in this tile
    if lc_flat.size > 0:
        isPW = bool(np.mean(lc_flat == codePW) >= 0.95)
    else:
        isPW = False

    thR = float(getattr(args, "thR", 0.0))

    # ---- call findThProb ----
    found, thProbOpt, catOpt, thI_low = findThProb(
        runStyle="wind_inclusive",   # glacial_exclusive left TODO
        thProbs=thProbs,
        Iu=Iu,
        I1=I1,
        sigma2=sigma2,
        ENL=ENL,
        binWidth=binWidth,
        meanTransact=meanTransact,
        thKLD=float(args.thKLD),
        thR=thR,
        thNSE=float(args.thNSE),
        thDist2Mean=float(args.thDist2Mean),
        thSampleSize=int(args.thMinSampleSize),
        isPW=isPW,
        hasPW=hasPW,
    )

    return iR, iC, thProbOpt, catOpt, thI_low
