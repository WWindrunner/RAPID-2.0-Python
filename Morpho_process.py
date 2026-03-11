import numpy as np
import dask
import dask.array as da
from itertools import product
import rasterio
import os
import psutil
from scipy import ndimage
import dask.dataframe as dd
import rasterio
from rasterio.windows import Window
from Dist_th import thresholdGMM_Dual
from Statistics import get_block
from utils import *


def _read_window_from_tif(path: str, r0: int, c0: int, nRows: int, nCols: int):
    with rasterio.open(path) as src:
        return src.read(1, window=Window(c0, r0, nCols, nRows))


def countPixels(rasWBCo, rasWBCross, isOcean, targets):
    """
    MATLAB:
      baseMask = (rasWBCo == 0) & (isOcean == 0)
      mask1 = baseMask & (rasWBCross ~= 0)
      VHDetected = nnz(mask1)
      mask2 = baseMask & ismember(rasWBCross, targets)
      VHDetectedHigh = nnz(mask2)
    """
    rasWBCo = np.asarray(rasWBCo)
    rasWBCross = np.asarray(rasWBCross)
    isOcean = np.asarray(isOcean)

    baseMask = (rasWBCo == 0) & (isOcean == 0)
    mask1 = baseMask & (rasWBCross != 0)
    VHDetected = int(np.count_nonzero(mask1))

    targets = np.asarray(targets).reshape(-1)
    mask2 = baseMask & np.isin(rasWBCross, targets)
    VHDetectedHigh = int(np.count_nonzero(mask2))

    return VHDetectedHigh, VHDetected


def patchBimodal(maskSeeds, I, thSizeBimodal, thPC, thAD, thRH, thDist2Mean):
    """
    Python port of patchBimodal.m

    Returns
    -------
    thBimodal_vals : 1D array of valid bimodal thresholds (dB)
    objID : 2D array, optional/debug
    th_all : 1D array, per large patch threshold (NaN if invalid)
    """
    maskSeeds = np.asarray(maskSeeds, dtype=bool)
    I = np.asarray(I, dtype=float)

    rowsBlock, colsBlock = maskSeeds.shape

    # MATLAB binary_regiongrow(...,'codedObject',4,...) -> 4-connectivity
    structure = ndimage.generate_binary_structure(2, 1)
    CMSeeds, nobj = ndimage.label(maskSeeds, structure=structure)

    if nobj == 0:
        return np.array([], dtype=float), np.full(maskSeeds.shape, np.nan), np.array([], dtype=float)

    patch_ids = np.arange(1, nobj + 1)
    patch_sizes = ndimage.sum(np.ones_like(maskSeeds, dtype=np.int32), labels=CMSeeds, index=patch_ids)
    patch_sizes = np.asarray(patch_sizes).reshape(-1)

    large_ids = patch_ids[patch_sizes > thSizeBimodal]

    thBimodal_map = np.zeros(maskSeeds.shape, dtype=float)
    objID = np.full(maskSeeds.shape, np.nan)
    th_list = []

    if large_ids.size == 0:
        return np.array([], dtype=float), objID, np.array([], dtype=float)

    for pid in large_ids:
        maski = (CMSeeds == pid)
        ri, ci = np.where(maski)

        rs = int(ri.min())
        re = int(ri.max())
        cs = int(ci.min())
        ce = int(ci.max())

        rows = re - rs + 1
        cols = ce - cs + 1

        # MATLAB:
        # rsB=max(1,round(rs-rows/2)); reB=min(rowsBlock,round(re+rows/2))
        # csB=max(1,round(cs-cols/2)); ceB=min(colsBlock,round(cs+cols/2))
        rsB = max(0, int(round(rs - rows / 2)))
        reB = min(rowsBlock - 1, int(round(re + rows / 2)))
        csB = max(0, int(round(cs - cols / 2)))
        ceB = min(colsBlock - 1, int(round(ce + cols / 2)))

        Ii = I[rsB:reB + 1, csB:ceB + 1]
        Ii = Ii[np.isfinite(Ii) & (Ii > 0)]

        if Ii.size == 0:
            th_list.append(np.nan)
            continue

        th, pc, mu, sigma, RH, AD, isValid, isOneMode = thresholdGMM_Dual(
            I_linear=Ii,
            probW=None,
            thProb=None,
            waterOnly=True,
            thPC=thPC,
            thAD=thAD,
            thRH=thRH,
            thDist2Mean=thDist2Mean,
            lowestMean=None,
        )

        th_list.append(th)

        if isValid and np.isfinite(th) and not np.isinf(th):
            thBimodal_map[maski] = th
            objID[maski] = pid

    th_list = np.asarray(th_list, dtype=float)
    thBimodal_vals = thBimodal_map[thBimodal_map > 0]

    return thBimodal_vals, objID, th_list


@dask.delayed
def morph_pre_block_worker(
    iR,
    iC,
    r0Load,
    c0Load,
    nRowsLoad,
    nColsLoad,
    fileWB_co,
    fileWB_cross,
    Ipp_block,
    ocean_block,
    idx_co,
    idx_cross,
    nClasses_co,
    nClasses_cross,
    codePW,
    thSizeBimodal,
    thPC,
    thAD,
    thRH,
    thDist2Mean,
):
    """
    One morph_pre block.
    Returns:
      {
        "iR": iR,
        "iC": iC,
        "thBimodal": 1D array,
        "VHDetectedHigh": int,
        "VHDetected": int,
      }
    """
    rasWBCo = _read_window_from_tif(fileWB_co, r0Load, c0Load, nRowsLoad, nColsLoad)
    rasWBCross = _read_window_from_tif(fileWB_cross, r0Load, c0Load, nRowsLoad, nColsLoad)

    Ipp = np.asarray(Ipp_block, dtype=float)
    isOcean = np.asarray(ocean_block, dtype=float)

    # MATLAB:
    # maskSeeds = rasWBCo==idx(1) | rasWBCo==nClasses+1
    target_seed_labels = [int(idx_co[0]), int(nClasses_co + 1)]
    maskSeeds = np.isin(rasWBCo, target_seed_labels)

    # MATLAB:
    # [VHDetectedHigh, VHDetected] = countPixels(rasWBCo, rasWBCross, isOcean, [idx_qp(1), (nClasses_qp+1)])
    targets_cross = [int(idx_cross[0]), int(nClasses_cross + 1)]
    VHDetectedHigh, VHDetected = countPixels(rasWBCo, rasWBCross, isOcean, targets_cross)

    thBimodal, objID, th = patchBimodal(
        maskSeeds=maskSeeds,
        I=Ipp,
        thSizeBimodal=thSizeBimodal,
        thPC=thPC,
        thAD=thAD,
        thRH=thRH,
        thDist2Mean=thDist2Mean,
    )

    return {
        "iR": int(iR),
        "iC": int(iC),
        "thBimodal": np.asarray(thBimodal, dtype=float),
        "VHDetectedHigh": int(VHDetectedHigh),
        "VHDetected": int(VHDetected),
    }


def morph_pre_scene(scene_pack, blockSize=None):
    """
    Python replacement of MATLAB morph_pre + morph_pre_worker.

    Expected scene_pack structure:
    {
        "scene_args": args_scene,
        "scene_ds": ds_scene,
        "pol_results": {
            "VV": (args_vv, ds_vv),
            "VH": (args_vh, ds_vh),   # optional
        }
    }
    """
    pol_results = scene_pack["pol_results"]

    if "VV" not in pol_results or "VH" not in pol_results:
        raise ValueError(
            "morph_pre_scene requires dual-polarization results (VV + VH). "
            "Single-pol morphology is not supported."
        )

    args_co, ds_co = pol_results["VV"]
    args_cross, ds_cross = pol_results["VH"]

    if blockSize is None:
        blockSize = int(getattr(args_co, "morphPreBlockSize", 1000))

    dirMorph = getattr(args_co, "dirMorph", os.path.join(args_co.dirOut, "Morph"))
    os.makedirs(dirMorph, exist_ok=True)

    fileWB_co = getattr(args_co, "fileCatInit", os.path.join(args_co.dirOut, "Cat_init.tif"))

    if args_cross is not None:
        fileWB_cross = getattr(args_cross, "fileCatInit", os.path.join(args_cross.dirOut, "Cat_init.tif"))
        Cqp = np.asarray(ds_cross.sigma2, dtype=float).reshape(-1)
        idx_cross = np.argsort(Cqp) + 1
        nClasses_cross = int(Cqp.size)
    else:
        fileWB_cross = None
        Cqp = np.array([], dtype=float)
        idx_cross = np.array([], dtype=int)
        nClasses_cross = 0

    Cpp = np.asarray(ds_co.sigma2, dtype=float).reshape(-1)
    idx_co = np.argsort(Cpp) + 1
    nClasses_co = int(Cpp.size)

    with rasterio.open(fileWB_co) as src:
        rowsImage = src.height
        colsImage = src.width

    nR, nC, r0, c0, nRows, nCols, _, _ = get_block(
        rowsImage,
        colsImage,
        block_size=blockSize,
        overlap=int(blockSize / 2),
        offset=False,
        keep_size=False,
    )

    # MATLAB worker uses read-window column [:,0]
    r0_read = np.asarray(r0)[:, 0] if np.ndim(r0) == 2 else np.asarray(r0)
    c0_read = np.asarray(c0)[:, 0] if np.ndim(c0) == 2 else np.asarray(c0)
    nRows_read = np.asarray(nRows)[:, 0] if np.ndim(nRows) == 2 else np.asarray(nRows)
    nCols_read = np.asarray(nCols)[:, 0] if np.ndim(nCols) == 2 else np.asarray(nCols)

    thSizeBimodal = int(getattr(args_co, "morphPrePatchSize", 500))
    thPC = tuple(getattr(args_co, "morphPrePC", (0.15, 0.85)))
    thAD = float(getattr(args_co, "morphPreAD", 2.0))
    thRH = float(getattr(args_co, "morphPreRH", 0.3))
    thDist2Mean = float(getattr(args_co, "morphPreDist2Mean", 0.1))
    codePW = int(getattr(args_co, "codePW", 80))

    tasks = []
    for iR in range(int(nR)):
        rr0 = int(r0_read[iR])
        rrn = int(nRows_read[iR])

        for iC in range(int(nC)):
            cc0 = int(c0_read[iC])
            ccn = int(nCols_read[iC])

            Ipp_block = ds_co.img_lp[rr0:rr0 + rrn, cc0:cc0 + ccn]
            ocean_block = ds_co.ocean_mask[rr0:rr0 + rrn, cc0:cc0 + ccn]

            task = morph_pre_block_worker(
                iR=iR,
                iC=iC,
                r0Load=rr0,
                c0Load=cc0,
                nRowsLoad=rrn,
                nColsLoad=ccn,
                fileWB_co=fileWB_co,
                fileWB_cross=fileWB_cross,
                Ipp_block=Ipp_block,
                ocean_block=ocean_block,
                idx_co=idx_co,
                idx_cross=idx_cross,
                nClasses_co=nClasses_co,
                nClasses_cross=nClasses_cross,
                codePW=codePW,
                thSizeBimodal=thSizeBimodal,
                thPC=thPC,
                thAD=thAD,
                thRH=thRH,
                thDist2Mean=thDist2Mean,
            )

            tasks.append(task)

    print("searching for highly credible threshold ...")
    results = dask.compute(*tasks)

    thBimodal_all = []
    VHDetectedHigh = 0
    VHDetected = 0

    for res in results:
        tb = np.asarray(res["thBimodal"], dtype=float).reshape(-1)
        if tb.size > 0:
            thBimodal_all.append(tb)

        VHDetectedHigh += int(res.get("VHDetectedHigh", 0))
        VHDetected += int(res.get("VHDetected", 0))

    if thBimodal_all:
        thBimodal = np.concatenate(thBimodal_all, axis=0)
    else:
        thBimodal = np.array([], dtype=float)

    isWindy = 0
    if VHDetected != 0:
        isWindy = int((VHDetectedHigh / VHDetected) >= 0.5)

    print(f"VHDetectedHigh = {VHDetectedHigh}")
    print(f"VHDetected = {VHDetected}")
    print(f"Windy case: {isWindy}")

    fileOut = os.path.join(dirMorph, "hi_cred_bimod_th.npz")
    np.savez(
        fileOut,
        thBimodal=thBimodal,
        isWindy=isWindy
    )

    if thBimodal.size > 0:
        finite = thBimodal[np.isfinite(thBimodal)]
        if finite.size > 0:
            print(f"median thBimodal = {np.median(finite)}")
            print(f"min thBimodal = {np.min(finite)}")

    # store back into scene_pack
    scene_pack["morph_pre"] = {
        "fileHiCredBimod": fileOut,
        "thBimodal": thBimodal,
        "isWindy": isWindy,
        "VHDetectedHigh": VHDetectedHigh,
        "VHDetected": VHDetected,
    }

    return scene_pack


@dask.delayed
def morph_block_worker_pre(
    iR,
    iC,
    r0Load,
    c0Load,
    nRowsLoad,
    nColsLoad,
    rs, re, cs, ce,
    r0Save, c0Save,
    fileWB_vv,
    fileWB_vh,
    fileThFine_vv,
    fileThFine_vh,
    Ipp_block,
    Iqp_block,
    LCC_block,
    water_occ_block,
    nCatPP,
    nCatQP,
    idx_vv,
    idx_vh,
    thGlobal,
    isWindy,
    WBClasses,
):
    """
    First-half port of morph_worker.m
    Returns intermediate arrays needed by the second half.
    """
    # --- read initial binary classification results ---
    rasWBCo = _read_window_tif(fileWB_vv, r0Load, c0Load, nRowsLoad, nColsLoad)
    rasWBCross = _read_window_tif(fileWB_vh, r0Load, c0Load, nRowsLoad, nColsLoad)

    # --- in-memory inputs ---
    Ipp = Ipp_block.compute() if hasattr(Ipp_block, "compute") else Ipp_block
    Iqp = Iqp_block.compute() if hasattr(Iqp_block, "compute") else Iqp_block
    LCC = LCC_block.compute() if hasattr(LCC_block, "compute") else LCC_block
    water_occ = water_occ_block.compute() if hasattr(water_occ_block, "compute") else water_occ_block

    Ipp = np.asarray(Ipp, dtype=np.float64)
    Iqp = np.asarray(Iqp, dtype=np.float64)
    LCC = np.asarray(LCC)
    water_occ = np.asarray(water_occ, dtype=np.float64)

    rows, cols = rasWBCo.shape
    rasWB = np.zeros((rows, cols), dtype=np.float64)

    # MATLAB:
    # Ipp(Ipp==0)=NaN;
    # Ipp(Iqp==0)=NaN;
    Ipp[Ipp == 0] = np.nan
    Ipp[Iqp == 0] = np.nan
    Iqp[Iqp == 0] = np.nan

    # --- load threshold stacks ---
    thpp = np.zeros((rows, cols, nCatPP + 1), dtype=np.float64)
    thqp = np.zeros((rows, cols, nCatQP + 1), dtype=np.float64)

    for c in range(1, nCatPP + 2):
        thpp[:, :, c - 1] = _read_window_tif(f"{fileThFine_vv}{c}.tif", r0Load, c0Load, nRowsLoad, nColsLoad)

    for c in range(1, nCatQP + 2):
        thqp[:, :, c - 1] = _read_window_tif(f"{fileThFine_vh}{c}.tif", r0Load, c0Load, nRowsLoad, nColsLoad)

    # --- masks ---
    maskOut = np.isnan(rasWBCo) | np.isnan(rasWBCross)
    rasWB[maskOut] = np.nan

    # step 1) generate seeds
    isPW = np.isin(LCC, WBClasses)
    maskPW = isPW & (~maskOut)

    # highly credible persistent water
    maskPWHighCred = maskPW & ((rasWBCo > 0) | (rasWBCross > 0))
    rasWB[maskPWHighCred] = 1

    # highly credible water with positive water occurrence
    if isWindy:
        maskWaterProb = (
            (~maskPWHighCred)
            & (water_occ > 0)
            & ((rasWBCo > 0) | (rasWBCross == idx_vh[0]) | (rasWBCross == (nCatQP + 1)))
            & (~maskOut)
        )
    else:
        maskWaterProb = (
            (~maskPWHighCred)
            & (water_occ > 0)
            & (rasWBCo > 0)
            & (~maskOut)
        )
    rasWB[maskWaterProb] = 2

    # highly credible water at zero water occurrence
    if isWindy:
        maskOtherHighCred = (
            ((rasWBCo == idx_vv[0]) | (rasWBCo == (nCatPP + 1)) |
             (rasWBCross == idx_vh[0]) | (rasWBCross == (nCatQP + 1)))
            & (rasWB == 0)
            & (~maskOut)
        )
    else:
        maskOtherHighCred = (
            ((rasWBCo == idx_vv[0]) | (rasWBCo == (nCatPP + 1)))
            & (rasWB == 0)
            & (~maskOut)
        )
    rasWB[maskOtherHighCred] = 3

    # moderately credible
    if len(idx_vv) >= 2:
        maskOtherModCred = (rasWBCo == idx_vv[1]) & (rasWB == 0) & (~maskOut)
        rasWB[maskOtherModCred] = 4
    else:
        maskOtherModCred = np.zeros_like(rasWB, dtype=bool)

    # low credibility
    rasWB[(rasWBCo > 0) & (rasWB == 0)] = 5

    # --- merge thresholds for PW ---
    thppM = np.zeros((rows, cols), dtype=np.float64)
    for c in range(1, nCatPP + 2):
        maskC = (rasWBCo == c)
        thppC = thpp[:, :, c - 1]
        sel = maskC & maskPWHighCred
        thppM[sel] = thppC[sel]

    thpp_high = np.maximum(thpp[:, :, idx_vv[0] - 1], thpp[:, :, -1])
    thqp_high = np.maximum(thqp[:, :, idx_vh[0] - 1], thqp[:, :, -1])

    thqpM = np.zeros((rows, cols), dtype=np.float64)
    for c in range(1, nCatQP + 2):
        maskC = (rasWBCross == c)
        thqpC = thqp[:, :, c - 1]
        sel = maskC & maskPWHighCred
        thqpM[sel] = thqpC[sel]

    return {
        "iR": int(iR),
        "iC": int(iC),
        "r0Save": int(r0Save),
        "c0Save": int(c0Save),
        "rs": int(rs),
        "re": int(re),
        "cs": int(cs),
        "ce": int(ce),

        "rasWBCo": rasWBCo,
        "rasWBCross": rasWBCross,
        "rasWB_init": rasWB,

        "Ipp": Ipp,
        "Iqp": Iqp,
        "LCC": LCC,
        "water_occ": water_occ,

        "thpp": thpp,
        "thqp": thqp,
        "thppM": thppM,
        "thqpM": thqpM,
        "thpp_high": thpp_high,
        "thqp_high": thqp_high,

        "maskOut": maskOut,
        "maskPW": maskPW,
        "maskPWHighCred": maskPWHighCred,
        "maskWaterProb": maskWaterProb,
        "maskOtherHighCred": maskOtherHighCred,
        "maskOtherModCred": maskOtherModCred,

        "thGlobal": float(thGlobal) if np.isfinite(thGlobal) else np.nan,
        "isWindy": int(isWindy),
    }


def morph_proc_scene_main(scene_pack, blockSize=None):
    """
    Main morph processing after morph_pre.
    This version ports the admin logic + worker first half.
    """
    pol_results = scene_pack["pol_results"]

    if "VV" not in pol_results or "VH" not in pol_results:
        raise ValueError("Morph processing requires dual-polarization results (VV + VH).")

    args_vv, ds_vv = pol_results["VV"]
    args_vh, ds_vh = pol_results["VH"]

    if "morph_pre" not in scene_pack:
        raise ValueError("scene_pack['morph_pre'] is missing. Run morph_pre_scene first.")

    morph_pre = scene_pack["morph_pre"]
    thBimodal = np.asarray(morph_pre.get("thBimodal", []), dtype=float).reshape(-1)
    thBimodal = thBimodal[np.isfinite(thBimodal) & (~np.isinf(thBimodal))]
    isWindy = int(morph_pre.get("isWindy", 0))

    if blockSize is None:
        blockSize = int(getattr(args_vv, "morphPreBlockSize", 1000))

    dirMorph = getattr(args_vv, "dirMorph", os.path.join(args_vv.dirOut, "Morph"))
    os.makedirs(dirMorph, exist_ok=True)

    fileMorphOut = os.path.join(dirMorph, "Morph_final.tif")
    fileMorphRoot = os.path.join(dirMorph, "MorphBlock_")

    # binary outputs
    fileWB_vv = getattr(args_vv, "fileCatInit", os.path.join(args_vv.dirOut, "Cat_init.tif"))
    fileWB_vh = getattr(args_vh, "fileCatInit", os.path.join(args_vh.dirOut, "Cat_init.tif"))

    # fine threshold roots
    fileThFine_vv = getattr(args_vv, "fileThFine_root", os.path.join(args_vv.dirTempPol, "ThFine_"))
    fileThFine_vh = getattr(args_vh, "fileThFine_root", os.path.join(args_vh.dirTempPol, "ThFine_"))

    # class stats
    Cpp = np.asarray(ds_vv.sigma2, dtype=float).reshape(-1)
    Cqp = np.asarray(ds_vh.sigma2, dtype=float).reshape(-1)

    nCatPP = int(Cpp.size)
    nCatQP = int(Cqp.size)
    idx_vv = np.argsort(Cpp) + 1
    idx_vh = np.argsort(Cqp) + 1

    # global threshold from morph_pre
    if thBimodal.size > 0:
        thPrcBimodal = float(getattr(args_vv, "thPrcBimodal", 50.0))
        thGlobal_db = float(np.percentile(thBimodal, thPrcBimodal))
        thGlobal = 10.0 ** (thGlobal_db / 10.0)
    else:
        thGlobal_db = np.nan
        thGlobal = np.nan

    WBClasses = np.asarray(getattr(args_vv, "WBClasses", [5, 6, 50, 60]), dtype=int)

    with rasterio.open(fileWB_vv) as src:
        rowsImage = src.height
        colsImage = src.width

    nR, nC, r0, c0, nRows, nCols, _, _ = get_block(
        rowsImage,
        colsImage,
        block_size=blockSize,
        overlap=int(blockSize / 2),
        offset=False,
        keep_size=False,
    )

    tasks = []
    print("Morphologic processing:")
    for iR in range(int(nR)):
        for iC in range(int(nC)):
            prog = ((iR) * int(nC) + iC + 1) / (int(nR) * int(nC))
            print(f"{int(prog * 100)}% ({iR+1},{iC+1})/({nR},{nC})...")

            # READ window
            r0Load = int(r0[iR, 0])
            nRowsLoad = int(nRows[iR, 0])
            c0Load = int(c0[iC, 0])
            nColsLoad = int(nCols[iC, 0])

            # PROCESS crop inside read window
            rs = int(r0[iR, 1] - r0[iR, 0])
            re = int(rs + nRows[iR, 1] - 1)
            cs = int(c0[iC, 1] - c0[iC, 0])
            ce = int(cs + nCols[iC, 1] - 1)

            r0Save = int(r0[iR, 1])
            c0Save = int(c0[iC, 1])

            Ipp_block = ds_vv.img_lp[r0Load:r0Load + nRowsLoad, c0Load:c0Load + nColsLoad]
            Iqp_block = ds_vh.img_lp[r0Load:r0Load + nRowsLoad, c0Load:c0Load + nColsLoad]
            LCC_block = ds_vv.land_cover[r0Load:r0Load + nRowsLoad, c0Load:c0Load + nColsLoad]
            water_occ_block = ds_vv.water_occurrence[r0Load:r0Load + nRowsLoad, c0Load:c0Load + nColsLoad]

            tasks.append(
                morph_block_worker_pre(
                    iR=iR,
                    iC=iC,
                    r0Load=r0Load,
                    c0Load=c0Load,
                    nRowsLoad=nRowsLoad,
                    nColsLoad=nColsLoad,
                    rs=rs,
                    re=re,
                    cs=cs,
                    ce=ce,
                    r0Save=r0Save,
                    c0Save=c0Save,
                    fileWB_vv=fileWB_vv,
                    fileWB_vh=fileWB_vh,
                    fileThFine_vv=fileThFine_vv,
                    fileThFine_vh=fileThFine_vh,
                    Ipp_block=Ipp_block,
                    Iqp_block=Iqp_block,
                    LCC_block=LCC_block,
                    water_occ_block=water_occ_block,
                    nCatPP=nCatPP,
                    nCatQP=nCatQP,
                    idx_vv=idx_vv,
                    idx_vh=idx_vh,
                    thGlobal=thGlobal,
                    isWindy=isWindy,
                    WBClasses=WBClasses,
                )
            )

    results = dask.compute(*tasks)

    scene_pack["morph_proc"] = {
        "fileMorphOut": fileMorphOut,
        "fileMorphRoot": fileMorphRoot,
        "thGlobal": thGlobal,
        "thGlobal_db": thGlobal_db,
        "isWindy": isWindy,
        "block_results_pre": list(results),
    }

    print("finished morphological process (first half only)")
    return scene_pack
