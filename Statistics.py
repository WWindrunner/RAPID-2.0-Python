import numpy as np
from scipy.stats import chi2
from typing import Tuple
from scipy.special import gammainc, gamma


def chi2_truncated_mean(df: float, a: np.ndarray) -> np.ndarray:
    """
    E[X | X <= a] for X ~ ChiSquare(df), vectorized.
    Uses lower incomplete gamma:
      E[X | X<=a] = 2 * γ(k+1, a/2) / γ(k, a/2),  k=df/2
    """
    a = np.asarray(a, dtype=float)
    k = df / 2.0
    x = a / 2.0

    # γ(k,x) = gammainc(k,x) * Γ(k)
    gk = gammainc(k, x) * gamma(k)
    gk1 = gammainc(k + 1.0, x) * gamma(k + 1.0)

    out = np.full_like(a, np.nan, dtype=float)
    mask = (a > 0) & np.isfinite(a) & (gk > 0) & np.isfinite(gk) & np.isfinite(gk1)
    out[mask] = 2.0 * gk1[mask] / gk[mask]
    return out


def build_Iu_and_meanTransact(thProbs: np.ndarray, ENL: np.ndarray, C: np.ndarray):
    """
    Port of MATLAB dist_th_worker precompute:
      thProbs=0.6:0.02:0.99
      Iu(:,cat)=chi2inv(thProbs,2*n(cat))*C(cat)/(2*n(cat))
      meanTransact = E[I | I<=Iu_low(p)] / p   (lowest class only)
    Returns:
      Iu shape (nProbs, nCats)
      meanTransact shape (nProbs,)
      low_class (int)
    """
    thProbs = np.asarray(thProbs, dtype=float).reshape(-1)
    ENL = np.asarray(ENL, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float).reshape(-1)

    nCats = int(C.size)
    nProbs = int(thProbs.size)

    Iu = np.full((nProbs, nCats), np.nan, dtype=float)

    for cat in range(nCats):
        df = 2.0 * ENL[cat]
        Xu = chi2.ppf(thProbs, df=df)                     # chi2inv
        Iu[:, cat] = Xu * C[cat] / (2.0 * ENL[cat])      # Xu*C/(2*n)

    low_class = int(np.nanargmin(C))

    # lowest class truncated mean
    df_low = 2.0 * ENL[low_class]
    Xu_low = chi2.ppf(thProbs, df=df_low)
    s_low = C[low_class] / (2.0 * ENL[low_class])       # I = s*X

    EX_cond = chi2_truncated_mean(df_low, Xu_low)       # E[X | X<=Xu]
    EI_cond = s_low * EX_cond                           # E[I | I<=Iu]
    meanTransact = EI_cond / thProbs                    # MATLAB divides by thProbs'

    return Iu, meanTransact, low_class


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


def KLD_chi2_frm_sample(
    I: np.ndarray,
    sigma2: float,
    interval: float,
    n: float,
    p,
    trailing: int,
    real2theo: bool,
    out_figure: str | None = None,
):
    """
    Compute KL divergence between empirical distribution (from samples I) and chi-square distribution
    with an upper probability cutoff p.

    MATLAB signature:
      [mKLD, Iu, thProb, r, NSE] = KLD_chi2_frm_sample(I, sigma2, interval, n, p, trailing, real2theo, outFigure)

    Parameters
    ----------
    I : array-like
        Sample intensities (linear domain, NOT dB).
    sigma2 : float
        Estimated sigma^2 parameter used in X = 2*n*I/sigma2.
    interval : float
        Bin width in I domain (same as MATLAB 'interval', which is the histogram BinWidth for I).
        Internally converted to X bin width: intX = interval * 2*n / sigma2.
    n : float
        ENL (number of looks).
    p : float or array-like or nan
        Upper probability cutoff(s). If nan or 1, will use 99th percentile init and set p=0.99.
    trailing : int
        1: uniform trailing until 0.99
        2: tiny constant trailing until 0.99
        3: no trailing
    real2theo : bool
        If True, uses same direction for KLD2 as KLD1 (as in MATLAB code).
    out_figure : str or None
        If provided, saves a plot comparing theoretical and empirical PDFs.
        Note: MATLAB saves .fig; here we save a PNG payload even if filename ends with ".fig".

    Returns
    -------
    mKLD : float
    Iu : float
        Optimal cutoff intensity (linear domain).
    thProb : float
        Selected cutoff probability.
    r : float
        Correlation coefficient between empirical and theoretical PDFs (at the selected p).
    NSE : float
        Nash–Sutcliffe efficiency between empirical and theoretical PDFs (at the selected p).
    """
    I = np.asarray(I, dtype=float)
    I = I[np.isfinite(I)]
    if I.size == 0 or sigma2 <= 0 or interval <= 0 or n <= 0:
        return float("inf"), float("nan"), float("nan"), 0.0, float("-inf")

    # ensure p is array
    if p is None:
        p_arr = np.array([np.nan], dtype=float)
    else:
        p_arr = np.atleast_1d(np.array(p, dtype=float))

    nPs = p_arr.size
    KLD1 = np.ones(nPs, dtype=float)
    KLD2 = np.ones(nPs, dtype=float)

    X = 2.0 * n * I / sigma2
    intX = interval * 2.0 * n / sigma2

    # store per-p results so we can pick the best one
    Xu_vals = np.full(nPs, np.nan, dtype=float)
    f_emp_list = [None] * nPs
    f_chi_list = [None] * nPs
    xvals_list = [None] * nPs

    for ip in range(nPs):
        pi = p_arr[ip]

        # Step 2: define empirical PDF of sample (with cutoff)
        if np.isnan(pi) or pi == 1.0:
            Iu = np.percentile(I, 99)
            Xu = 2.0 * n * Iu / sigma2
            pi = 0.99
            p_arr[ip] = pi
        else:
            Xu = chi2.ppf(pi, df=2.0 * n)

        if not np.isfinite(Xu) or Xu <= 0:
            # degenerate
            KLD1[ip] = float("inf")
            KLD2[ip] = float("inf")
            continue

        # bins in X domain: [0, Xu]
        # MATLAB histogram with BinWidth uses data range; we anchor at 0 for stability.
        edges = np.arange(0.0, Xu + intX * 1.0000001, intX)
        if edges.size < 3:
            # too few bins -> fail
            KLD1[ip] = float("inf")
            KLD2[ip] = float("inf")
            continue

        Xc = X[X < Xu]
        if Xc.size < 2:
            KLD1[ip] = float("inf")
            KLD2[ip] = float("inf")
            continue

        hist, bin_edges = np.histogram(Xc, bins=edges, density=True)
        x_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        f_empirical = hist.astype(float)

        f_empirical[f_empirical <= 0] = 1e-4

        if x_values.size < 2:
            # MATLAB early return
            return float("inf"), float("nan"), float("nan"), 0.0, float("-inf")

        dx = x_values[1] - x_values[0]
        Xe = x_values[-1]

        # trailing handling (match MATLAB)
        if trailing == 1:
            XU = chi2.ppf(0.99, df=2.0 * n)
            Xtrailing = np.arange(Xe + dx, XU + dx * 0.5, dx)
            if Xtrailing.size > 0 and XU > Xe:
                pdTrailing = (0.99 - pi) / (XU - Xe)
                f_empirical = np.concatenate([f_empirical, pdTrailing * np.ones(Xtrailing.size)])
                x_values = np.concatenate([x_values, Xtrailing])
            f_empirical = f_empirical * pi

        elif trailing == 2:
            XU = chi2.ppf(0.99, df=2.0 * n)
            Xtrailing = np.arange(Xe + dx, XU + dx * 0.5, dx)
            pdTrailing = 1e-4
            pTrailing = pdTrailing * Xtrailing.size
            if Xtrailing.size > 0:
                f_empirical = np.concatenate([f_empirical * (1.0 - pTrailing), pdTrailing * np.ones(Xtrailing.size)])
                x_values = np.concatenate([x_values, Xtrailing])
            f_empirical = f_empirical * pi

        elif trailing == 3:
            # no trailing
            pass
        else:
            raise ValueError(f"Unknown trailing={trailing}; expected 1,2,3")

        f_chi_square = chi2.pdf(x_values, df=2.0 * n)

        if trailing in (2, 3):
            # MATLAB divides by p for trailing 2 and 3
            f_chi_square = f_chi_square / pi

        # KL divergence
        # (avoid divide-by-zero)
        f_chi_square = np.maximum(f_chi_square, 1e-12)
        f_empirical = np.maximum(f_empirical, 1e-12)

        KLD1[ip] = np.sum(f_empirical * np.log(f_empirical / f_chi_square)) * dx
        if not real2theo:
            KLD2[ip] = np.sum(f_chi_square * np.log(f_chi_square / f_empirical)) * dx
        else:
            KLD2[ip] = np.sum(f_empirical * np.log(f_empirical / f_chi_square)) * dx

        Xu_vals[ip] = Xu
        f_emp_list[ip] = f_empirical
        f_chi_list[ip] = f_chi_square
        xvals_list[ip] = x_values

    KLD = np.maximum(KLD1, KLD2)
    index = int(np.nanargmin(KLD)) if np.any(np.isfinite(KLD)) else 0
    mKLD = float(KLD[index])

    Xu_best = Xu_vals[index]
    if not np.isfinite(Xu_best):
        return float("inf"), float("nan"), float("nan"), 0.0, float("-inf")

    Iu_best = float(Xu_best * sigma2 / (2.0 * n))
    thProb_best = float(p_arr[index])

    f_emp_best = f_emp_list[index]
    f_chi_best = f_chi_list[index]
    x_best = xvals_list[index]

    if f_emp_best is None or f_chi_best is None:
        return float("inf"), float("nan"), float("nan"), 0.0, float("-inf")

    # correlation
    r = float(np.corrcoef(f_emp_best, f_chi_best)[0, 1]) if f_emp_best.size > 1 else 0.0

    # NSE
    denom = np.sum((f_chi_best - np.mean(f_chi_best)) ** 2)
    NSE = float(1.0 - np.sum((f_emp_best - f_chi_best) ** 2) / denom) if denom > 0 else float("-inf")

    # optional figure
    if out_figure is not None:
        import matplotlib.pyplot as plt

        # MATLAB plots in I-domain and converts density units accordingly
        I_axis = x_best * sigma2 / (2.0 * n)
        theo_pdf_I = f_chi_best * (2.0 * n) / sigma2
        emp_pdf_I = f_emp_best * (2.0 * n) / sigma2

        fig = plt.figure()
        plt.plot(I_axis, theo_pdf_I, "-k", label="Chi-square (theo)")
        plt.plot(I_axis, emp_pdf_I, "-r", label="Empirical")
        plt.title(f"KLD={mKLD:.4g}, r={r:.4g}, NSE={NSE:.4g}")
        plt.legend()
        # 保存：即使扩展名是 .fig，也强制写 PNG 数据，方便你后面 delete 文件
        try:
            fig.savefig(out_figure, format="png", dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)

    return mKLD, Iu_best, thProb_best, r, NSE


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