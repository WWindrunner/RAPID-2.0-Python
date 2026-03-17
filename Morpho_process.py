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
from rasterio.merge import merge
from Dist_th import thresholdGMM_Dual
from Statistics import get_block
from utils import *
from affine import Affine
from collections import deque


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


def _positive_stat(v, mode: str):
    v = np.asarray(v, dtype=float).reshape(-1)
    vp = v[np.isfinite(v) & (v > 0)]
    if vp.size == 0:
        return 0.0

    mode = mode.lower()
    if mode == "median":
        return float(np.median(vp))
    elif mode == "prc99":
        return float(np.percentile(vp, 99))
    elif mode == "prc90":
        return float(np.percentile(vp, 90))
    elif mode == "prc25":
        return float(np.percentile(vp, 25))
    else:
        raise ValueError(f"unknown thresholding option: {mode}")


def _neighbor_offsets(nNeighbor=4, nWin=0):
    """
    Match MATLAB neighbor definition.
    """
    if nNeighbor == 8:
        dr = np.array([-1, -1, -1,  0, 0, 1, 1, 1], dtype=int)
        dc = np.array([-1,  0,  1, -1, 1,-1, 0, 1], dtype=int)
    elif nNeighbor == 4:
        dr = np.array([0, -1, 1, 0], dtype=int)
        dc = np.array([-1, 0, 0, 1], dtype=int)
    elif nWin > 0:
        expected = (nWin * 2 + 1) ** 2 - 1
        if nNeighbor != expected:
            nNeighbor = expected
        rr = np.arange(-nWin, nWin + 1, dtype=int)
        cc = rr.copy()
        dc_grid, dr_grid = np.meshgrid(cc, rr)
        dr = dr_grid.ravel()
        dc = dc_grid.ravel()
        center = (dr == 0) & (dc == 0)
        dr = dr[~center]
        dc = dc[~center]
    else:
        raise ValueError(f"Unsupported neighbor setup: nNeighbor={nNeighbor}, nWin={nWin}")
    return dr, dc


def _single_object_regiongrow(I, rowSeed, colSeed, nNeighbor=4, nWin=0):
    """
    Python equivalent of MATLAB binary_regiongrow(..., 'singleObject', ...)
    Returns boolean mask.
    Assumes rowSeed/colSeed are 0-based.
    """
    I = np.asarray(I)
    rows, cols = I.shape

    # MATLAB logic: if ~islogical(I), I=logical(I)
    # Also reject NaN / False / out-of-mask
    valid_mask = np.isfinite(I) & (I != 0)

    CM = np.zeros((rows, cols), dtype=bool)

    rowSeed = np.asarray(rowSeed, dtype=int).reshape(-1)
    colSeed = np.asarray(colSeed, dtype=int).reshape(-1)

    # keep only valid seeds inside mask
    inside = (rowSeed >= 0) & (rowSeed < rows) & (colSeed >= 0) & (colSeed < cols)
    rowSeed = rowSeed[inside]
    colSeed = colSeed[inside]

    if rowSeed.size == 0:
        return CM

    seed_valid = valid_mask[rowSeed, colSeed]
    rowSeed = rowSeed[seed_valid]
    colSeed = colSeed[seed_valid]

    if rowSeed.size == 0:
        return CM

    dr, dc = _neighbor_offsets(nNeighbor=nNeighbor, nWin=nWin)

    q = deque()
    for r, c in zip(rowSeed, colSeed):
        if not CM[r, c]:
            CM[r, c] = True
            q.append((int(r), int(c)))

    while q:
        r0, c0 = q.popleft()
        for drr, dcc in zip(dr, dc):
            r = r0 + int(drr)
            c = c0 + int(dcc)

            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if CM[r, c]:
                continue
            if not valid_mask[r, c]:
                continue

            CM[r, c] = True
            q.append((r, c))

    return CM


def binary_regiongrow(I, rowSeed, colSeed, opt, nNeighbor, nWin, showProg, C0=None, mCID=None):
    """
    Python port of MATLAB binary_regiongrow.m

    Parameters
    ----------
    I : 2D array
        Boolean-like image / mask.
    rowSeed, colSeed : array-like or scalar
        Seed coordinates, 0-based in Python.
    opt : str
        'singleObject' or e.g. 'codedObject'
    nNeighbor : int
        4, 8, or custom based on nWin.
    nWin : int
        Used when neighborhood is custom.
    showProg : bool
        Ignored in Python version.
    C0 : optional 2D label map
        Existing coded map.
    mCID : optional int
        Starting label after C0.

    Returns
    -------
    CM : 2D bool or 2D int
    optionally c_next if coded mode (to mimic varargout={c})
    """
    I = np.asarray(I)
    rows, cols = I.shape

    rowSeed = np.asarray(rowSeed, dtype=int).reshape(-1)
    colSeed = np.asarray(colSeed, dtype=int).reshape(-1)

    if str(opt).lower() == "singleobject":
        CM = _single_object_regiongrow(I, rowSeed, colSeed, nNeighbor=nNeighbor, nWin=nWin)
        return CM

    # -------------------------------------------------
    # coded-object mode
    # -------------------------------------------------
    CM = np.zeros((rows, cols), dtype=np.int32)
    c = 1

    if C0 is not None:
        C0 = np.asarray(C0)
        if C0.shape != (rows, cols):
            raise ValueError(f"C0 shape mismatch: expected {(rows, cols)}, got {C0.shape}")

        CM[:] = C0.astype(np.int32, copy=False)

        rowSeed0, colSeed0 = np.where(C0 > 0)
        if rowSeed0.size > 0:
            # remove seed-groups already represented in C0
            rowSeed0 = rowSeed0.astype(int)
            colSeed0 = colSeed0.astype(int)

            while rowSeed0.size > 0:
                iCM = _single_object_regiongrow(I, rowSeed0[0], colSeed0[0], nNeighbor=nNeighbor, nWin=nWin)
                CM[iCM] = int(C0[rowSeed0[0], colSeed0[0]])

                # remove seeds covered by this region from rowSeed0/colSeed0
                keep0 = ~iCM[rowSeed0, colSeed0]
                rowSeed0 = rowSeed0[keep0]
                colSeed0 = colSeed0[keep0]

                # also remove covered seeds from rowSeed/colSeed
                if rowSeed.size > 0:
                    keep = ~iCM[rowSeed, colSeed]
                    rowSeed = rowSeed[keep]
                    colSeed = colSeed[keep]

            if mCID is not None:
                c = int(mCID)

    totalSeeds = rowSeed.size

    while rowSeed.size > 0:
        iCM = _single_object_regiongrow(I, rowSeed[0], colSeed[0], nNeighbor=nNeighbor, nWin=nWin)
        CM[iCM] = int(c)
        c += 1

        keep = ~iCM[rowSeed, colSeed]
        rowSeed = rowSeed[keep]
        colSeed = colSeed[keep]

    return CM, c


def extendTh(rSeeds, cSeeds, WBMax, CM, thMatIn, optTh, thBimodal):
    """
       Python port of MATLAB extendTh.m

       Parameters
       ----------
       rSeeds, cSeeds : 1D arrays
           Seed row/col indices, 0-based.
       WBMax : 2D bool array
           Maximum extent mask to region-grow within.
       CM : 2D int array or None
           Connected-component map. If None/empty, generated by binary_regiongrow.
       thMatIn : 2D array
           Input threshold map used to estimate per-patch threshold from seed pixels.
       optTh : str
           One of: 'median', 'prc99', 'prc90', 'prc25'
       thBimodal : 2D array or None
           Optional bimodality threshold map; if provided, take min with seed-stat threshold.

       Returns
       -------
       thMatOut : 2D float array
       maskPatches : 2D bool array
       CM : 2D int array
       """
    WBMax = np.asarray(WBMax, dtype=bool)
    thMatIn = np.asarray(thMatIn, dtype=float)

    rows, cols = WBMax.shape

    rSeeds = np.asarray(rSeeds, dtype=int).reshape(-1)
    cSeeds = np.asarray(cSeeds, dtype=int).reshape(-1)

    # guard empty seeds
    if rSeeds.size == 0 or cSeeds.size == 0:
        if CM is None or (hasattr(CM, "size") and np.size(CM) == 0):
            CM = np.zeros((rows, cols), dtype=np.int32)
        thMatOut = np.zeros((rows, cols), dtype=float)
        maskPatches = np.zeros((rows, cols), dtype=bool)
        return thMatOut, maskPatches, CM

    # build CM if needed
    if CM is None or (hasattr(CM, "size") and np.size(CM) == 0):
        CM, _ = binary_regiongrow(WBMax, rSeeds, cSeeds, 'codedObject', 4, 0, True)

    CM = np.asarray(CM)
    if CM.shape != (rows, cols):
        raise ValueError(f"CM shape mismatch: expected {(rows, cols)}, got {CM.shape}")

    # MATLAB:
    # ind=sub2ind([rows,cols],rSeeds,cSeeds);
    # CMSeeds=CM(ind);
    CMSeeds = CM[rSeeds, cSeeds].astype(int)

    # keep only valid labeled seeds
    valid_seed = CMSeeds > 0
    if not np.any(valid_seed):
        thMatOut = np.zeros((rows, cols), dtype=float)
        maskPatches = CM > 0
        return thMatOut, maskPatches, CM

    rSeeds_v = rSeeds[valid_seed]
    cSeeds_v = cSeeds[valid_seed]
    CMSeeds_v = CMSeeds[valid_seed]

    # th from seed locations
    th = thMatIn[rSeeds_v, cSeeds_v]

    # accumulate threshold statistic per connected patch ID
    max_id = int(np.max(CMSeeds_v))
    thMedian = np.zeros(max_id + 1, dtype=float)

    for pid in np.unique(CMSeeds_v):
        vals = th[CMSeeds_v == pid]
        thMedian[int(pid)] = _positive_stat(vals, optTh)

    # assign patch-level threshold to all pixels in that patch
    thMatOut = np.zeros((rows, cols), dtype=float)
    mask = CM > 0
    thMatOut[mask] = thMedian[CM[mask].astype(int)]

    # optional bimodality threshold refinement
    if thBimodal is not None and np.size(thBimodal) > 0:
        thBimodal = np.asarray(thBimodal, dtype=float)
        if thBimodal.shape != (rows, cols):
            raise ValueError(f"thBimodal shape mismatch: expected {(rows, cols)}, got {thBimodal.shape}")

        CMBimodal = thBimodal[rSeeds_v, cSeeds_v]
        thMinBimodal = np.full(max_id + 1, np.inf, dtype=float)

        for pid in np.unique(CMSeeds_v):
            vals = CMBimodal[CMSeeds_v == pid]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                thMinBimodal[int(pid)] = float(np.min(vals))

        valid_patch = np.isfinite(thMinBimodal)
        mask_bi = mask & valid_patch[CM.astype(int)]
        thMatOut[mask_bi] = np.minimum(
            thMatOut[mask_bi],
            thMinBimodal[CM[mask_bi].astype(int)]
        )

    maskPatches = CM > 0
    return thMatOut, maskPatches, CM


def sub_transform_from_ref(ref_transform, r0, c0):
    """
    Equivalent to MATLAB subTranscoef(geoTrans, r0Save, c0Save)
    where r0, c0 are 0-based pixel offsets.
    """
    return ref_transform * Affine.translation(c0, r0)


def identifyBoundaryMulti(CM, connectivity=4):
    """
    Identify boundary pixels of labeled objects in CM (>0).
    A pixel is boundary if any neighbor under given connectivity has different label or is background.
    """
    CM = np.asarray(CM)
    boundary = np.zeros(CM.shape, dtype=bool)

    if connectivity == 4:
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        shifts = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    rows, cols = CM.shape
    mask_obj = CM > 0

    for dr, dc in shifts:
        shifted = np.zeros_like(CM)
        if dr >= 0:
            rs_src = slice(0, rows - dr)
            rs_dst = slice(dr, rows)
        else:
            rs_src = slice(-dr, rows)
            rs_dst = slice(0, rows + dr)

        if dc >= 0:
            cs_src = slice(0, cols - dc)
            cs_dst = slice(dc, cols)
        else:
            cs_src = slice(-dc, cols)
            cs_dst = slice(0, cols + dc)

        shifted[rs_dst, cs_dst] = CM[rs_src, cs_src]
        boundary |= mask_obj & (shifted != CM)

    return boundary


def RoI(CM, LCC, codeLCC, th, maskHigher, encapOpt):
    """
    Python port of RoI.m

    ratio = overlapping count / total patch pixels
    """
    CM = np.asarray(CM)
    LCC = np.asarray(LCC)
    codeLCC = np.asarray(codeLCC).reshape(-1)

    maskHasObj = CM > 0
    claInd = CM[maskHasObj].astype(int)

    if claInd.size == 0:
        mask2Remove = np.zeros_like(CM, dtype=bool)
        ratioMat = np.full(CM.shape, np.nan, dtype=float)
        return mask2Remove, ratioMat

    uClaInd = np.unique(claInd)
    max_id = int(np.max(uClaInd))
    objSum = np.bincount(claInd, minlength=max_id + 1).astype(float)

    bufSize = 10

    if encapOpt == 'contain':
        coincidence = np.isin(LCC, codeLCC)
        coinVals = coincidence[maskHasObj].astype(float)
        coinSum = np.bincount(claInd, weights=coinVals, minlength=max_id + 1).astype(float)

    elif encapOpt == 'buffer':
        se = ndimage.generate_binary_structure(2, 1)
        # approximate disk radius 10 with repeated dilation
        coinSum = np.zeros(max_id + 1, dtype=float)
        for uCID in uClaInd:
            maskCi = (CM == uCID)
            buffer = maskCi.copy()
            for _ in range(bufSize):
                buffer = ndimage.binary_dilation(buffer, structure=se)
            LCCBuffer = LCC[buffer]
            coincidence = np.isin(LCCBuffer, codeLCC)
            coinSum[int(uCID)] = float(np.sum(coincidence))

    elif encapOpt == 'squared_buffer':
        rows, cols = CM.shape
        coinSum = np.zeros(max_id + 1, dtype=float)
        for uCID in uClaInd:
            maskCi = (CM == uCID)
            row, col = np.where(maskCi)
            if row.size == 0:
                continue
            minRow, maxRow = row.min(), row.max()
            minCol, maxCol = col.min(), col.max()
            minRow = max(0, minRow - bufSize)
            maxRow = min(rows - 1, maxRow + bufSize)
            minCol = max(0, minCol - bufSize)
            maxCol = min(cols - 1, maxCol + bufSize)

            subLCC = LCC[minRow:maxRow + 1, minCol:maxCol + 1]
            coincidence = np.isin(subLCC, codeLCC)
            coinSum[int(uCID)] = float(np.sum(coincidence))
    else:
        raise ValueError(f"Unknown encapOpt={encapOpt}")

    roi = np.full(max_id + 1, np.nan, dtype=float)
    valid_ids = objSum > 0
    roi[valid_ids] = coinSum[valid_ids] / objSum[valid_ids]

    ratioMat = np.full(CM.shape, np.nan, dtype=float)
    ratioMat[maskHasObj] = roi[CM[maskHasObj].astype(int)]

    if maskHigher:
        cid = np.where(roi > th)[0]
    else:
        cid = np.where(roi < th)[0]

    mask2Remove = np.isin(CM, cid)
    return mask2Remove, ratioMat


def compactness(CM):
    """
    Python port of compactness.m
    Input:
      CM: (rows, cols, nClasses)
    Returns:
      compactScore: list of arrays with columns [patchID, score]
      area:         list of arrays with columns [patchID, area]
    """
    CM = np.asarray(CM)
    nClasses = CM.shape[2]

    compactScore = [None] * nClasses
    area = [None] * nClasses

    for credibility in range(nClasses):
        CMc = CM[:, :, credibility].astype(int)

        if np.count_nonzero(CMc) == 0:
            compactScore[credibility] = np.empty((0, 2), dtype=float)
            area[credibility] = np.empty((0, 2), dtype=float)
            continue

        isBoundary = identifyBoundaryMulti(CMc, connectivity=4)
        boundaryClass = CMc[isBoundary]
        boundaryClass = boundaryClass[boundaryClass > 0]

        max_id = int(CMc.max())
        perimeter = np.bincount(boundaryClass, minlength=max_id + 1).astype(float)

        maskPatches = CMc > 0
        patch_ids = CMc[maskPatches]
        area_counts = np.bincount(patch_ids, minlength=max_id + 1).astype(float)

        valid_ids = np.where(area_counts > 0)[0]

        area_arr = np.column_stack([valid_ids, area_counts[valid_ids]]) if valid_ids.size > 0 else np.empty((0, 2), dtype=float)

        score = np.full(max_id + 1, np.nan, dtype=float)
        valid_perim = valid_ids[perimeter[valid_ids] > 0]
        score[valid_perim] = area_counts[valid_perim] / perimeter[valid_perim]

        score_arr = np.column_stack([valid_perim, score[valid_perim]]) if valid_perim.size > 0 else np.empty((0, 2), dtype=float)

        compactScore[credibility] = score_arr
        area[credibility] = area_arr

    return compactScore, area


@dask.delayed
def morph_block_worker(
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
    EXClasses,
    nWBClasses,
    rNWB,
    thSize,
    thCompactness,
    morph_out_dir,
    block_transform,
    crs
):
    """
    Full port of morph_worker.m for one block,
    except helper functions not yet provided are left as placeholders.
    Writes one block tif: fileBlock
    """
    print(f"[morph_block_worker] start ({iR + 1},{iC + 1})", flush=True)

    # ---- read initial water masks ----
    rasWBCo = _read_window_from_tif(fileWB_vv, r0Load, c0Load, nRowsLoad, nColsLoad)
    rasWBCross = _read_window_from_tif(fileWB_vh, r0Load, c0Load, nRowsLoad, nColsLoad)

    # ---- load in-memory arrays ----
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

    # ---- threshold stacks ----
    thpp = np.zeros((rows, cols, nCatPP + 1), dtype=np.float64)
    thqp = np.zeros((rows, cols, nCatQP + 1), dtype=np.float64)

    for c in range(1, nCatPP + 2):
        thpp[:, :, c - 1] = _read_window_from_tif(f"{fileThFine_vv}{c}.tif", r0Load, c0Load, nRowsLoad, nColsLoad)

    for c in range(1, nCatQP + 2):
        thqp[:, :, c - 1] = _read_window_from_tif(f"{fileThFine_vh}{c}.tif", r0Load, c0Load, nRowsLoad, nColsLoad)

    # ---- masks ----
    maskOut = np.isnan(rasWBCo) | np.isnan(rasWBCross)
    rasWB[maskOut] = np.nan

    # step 1) generate seeds
    isPW_mask = np.isin(LCC, WBClasses)
    maskPW = isPW_mask & (~maskOut)

    maskPWHighCred = maskPW & ((rasWBCo > 0) | (rasWBCross > 0))
    rasWB[maskPWHighCred] = 1

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

    if len(idx_vv) >= 2:
        maskOtherModCred = (rasWBCo == idx_vv[1]) & (rasWB == 0) & (~maskOut)
        rasWB[maskOtherModCred] = 4
    else:
        maskOtherModCred = np.zeros_like(rasWB, dtype=bool)

    rasWB[(rasWBCo > 0) & (rasWB == 0)] = 5

    # ---- merge thresholds for PW ----
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

    # =========================================================
    # form high to low-credibility water bodies
    # =========================================================

    # ---- form PW objects and identify ----
    maskSeeds = (rasWB == 1)
    RSeeds, CSeeds = np.where(maskSeeds)
    if RSeeds.size > 0:
        thPWppBlock, maskPatches, CM = extendTh(RSeeds, CSeeds, maskPW, None, thppM, 'prc90', None)
        thPWqpBlock, _, _ = extendTh(RSeeds, CSeeds, maskPW, CM, thqpM, 'prc90', None)

        accept = (Ipp < thPWppBlock) | (Iqp < thPWqpBlock)
        rasWB[maskPatches] = 1 * accept[maskPatches]

    mask_thHigh = (rasWB == 1) & ((rasWBCo == idx_vv[0]) | (rasWBCo == nCatPP + 1))
    if np.any(mask_thHigh):
        thHigh = np.percentile(Ipp[mask_thHigh & np.isfinite(Ipp)], 75)
    else:
        thHigh = np.nan

    # ---- form highly-credible water objects ----
    maskSeeds = (rasWB == 2)
    RSeeds, CSeeds = np.where(maskSeeds)
    if RSeeds.size > 0:
        thPWppBlock, maskPatches, _ = extendTh(RSeeds, CSeeds, maskSeeds, None, thpp_high, 'prc90', None)
        thPWqpBlock, _, _ = extendTh(RSeeds, CSeeds, maskSeeds, None, thqp_high, 'prc90', None)

        thPWppBlock = np.asarray(thPWppBlock, dtype=float)
        maskPatches = np.asarray(maskPatches, dtype=bool)

        fill_val = thGlobal if np.isfinite(thGlobal) else 0.0
        thPWppBlock[maskPatches] = (
                thPWppBlock[maskPatches] * (thPWppBlock[maskPatches] > 0)
                + fill_val * (thPWppBlock[maskPatches] <= 0)
        )

        if isWindy:
            accept = (Ipp < thPWppBlock) | (Iqp < thPWqpBlock)
        else:
            accept = (Ipp < thPWppBlock)
        rasWB[maskPatches] = 2 * accept[maskPatches]

    # ---- form non-PW highly credible objects ----
    maskSeeds = (rasWB == 3)
    RSeeds, CSeeds = np.where(maskSeeds)
    if RSeeds.size > 0:
        if isWindy:
            WBMax = ((rasWBCo > 0) | (rasWBCross == idx_vh[0]) | (rasWBCross == (nCatQP + 1))) & (rasWB > 2)
            thMatOut, maskPatches, _ = extendTh(RSeeds, CSeeds, WBMax, None, thpp_high, 'prc90', None)
            thPWqpBlock, _, _ = extendTh(RSeeds, CSeeds, WBMax, None, thqp_high, 'prc90', None)

            cap_val = np.nanmin([thGlobal, thHigh]) if np.any(np.isfinite([thGlobal, thHigh])) else np.nan
            thMatOut = np.minimum(thMatOut, cap_val) if np.isfinite(cap_val) else thMatOut

            accept = (Ipp < thMatOut) | (Iqp < thPWqpBlock)
            rasWB[maskPatches] = 3 * accept[maskPatches]
        else:
            WBMax = (rasWBCo > 0) & (rasWB > 2)
            thMatOut, maskPatches, _ = extendTh(RSeeds, CSeeds, WBMax, None, thpp_high, 'prc90', None)

            cap_val = np.nanmin([thGlobal, thHigh]) if np.any(np.isfinite([thGlobal, thHigh])) else np.nan
            thMatOut = np.minimum(thMatOut, cap_val) if np.isfinite(cap_val) else thMatOut

            accept = (Ipp < thMatOut)
            rasWB[maskPatches] = 3 * accept[maskPatches]

    # ---- moderate-credible objects ----
    maskSeeds = (rasWB == 4)
    if np.any(maskSeeds):
        RSeeds, CSeeds = np.where(maskSeeds)
        WBMax = (rasWBCo > 0) & (rasWB > 3)
        thMatOut, maskPatches, _ = extendTh(RSeeds, CSeeds, WBMax, None, thpp[:, :, idx_vv[1] - 1], 'prc90', None)

        cap_val = np.nanmin([thGlobal, thHigh]) if np.any(np.isfinite([thGlobal, thHigh])) else np.nan
        thMatOut = np.minimum(thMatOut, cap_val) if np.isfinite(cap_val) else thMatOut

        accept = (Ipp < thMatOut)
        rasWB[maskPatches] = 4 * accept[maskPatches]

    # ---- low-credible objects ----
    maskLow = (rasWB == 5)
    if np.any(maskLow):
        cap_val = np.nanmin([thGlobal, thHigh]) if np.any(np.isfinite([thGlobal, thHigh])) else np.nan
        if np.isfinite(cap_val):
            rasWB[maskLow] = 5 * (Ipp[maskLow] < cap_val)

    # =========================================================
    # step 4) remove small objects / low compactness
    # =========================================================
    nCred = 5
    CM = np.zeros((rows, cols, 4), dtype=np.int32)
    lstRealCred = [None] * nCred

    for credibility in range(1, nCred + 1):
        maskExt = (rasWB >= credibility)
        if credibility > 1:
            for c in range(1, credibility):
                maskExt = maskExt & (CM[:, :, c - 1] == 0)

        maskSeeds = (rasWB == credibility) & maskExt
        RSeeds, CSeeds = np.where(maskSeeds)

        # MATLAB uses only 4 layers in CM even though nCred=5
        if credibility <= 4:
            CM[:, :, credibility - 1], _ = binary_regiongrow(maskExt, RSeeds, CSeeds, 'codedObject', 8, 0, True)
            CMc = CM[:, :, credibility - 1]
        else:
            # keep last credibility using previous CM logic if needed
            CMc = np.zeros_like(rasWB, dtype=np.int32)

        if credibility >= 3:
            if isWindy:
                catVector = rasWBCo.copy()
                temp_mask = ((rasWBCross == idx_vh[0]) | (rasWBCross == (nCatQP + 1))) & (catVector == 0)
                catVector[temp_mask] = 3  # temp number
            else:
                catVector = rasWBCo

            # placeholder logic: realCat depends on connected-component mode
            # requires accumarray/mode equivalent inside a helper later if exact behavior matters
            patch_ids = np.unique(CMc[CMc > 0])
            realCred = []

            for pid in patch_ids:
                vals = catVector[CMc == pid]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                catInd = int(np.bincount(vals.astype(int)).argmax())

                credInd = 0
                if (catInd == idx_vv[0]) or (catInd == nCatPP + 1):
                    credInd = 3
                elif len(idx_vv) >= 2 and (catInd == idx_vv[1]):
                    credInd = 4
                else:
                    low_match = False
                    for ic in range(2, nCatPP):
                        if catInd == idx_vv[ic]:
                            low_match = True
                            break
                    if low_match:
                        credInd = 5

                if credInd > 0:
                    realCred.append((pid, credInd))
                    rasWB[CMc == pid] = credInd

            lstRealCred[credibility - 1] = np.array(realCred, dtype=int) if len(realCred) > 0 else np.empty((0, 2), dtype=int)

        else:
            rasWB[CMc > 0] = credibility
            patch_ids = np.unique(CMc[CMc > 0])
            lstRealCred[credibility - 1] = np.column_stack([
                patch_ids,
                np.full(patch_ids.shape, credibility, dtype=int)
            ]) if patch_ids.size > 0 else np.empty((0, 2), dtype=int)

    # remove objects with high built-up ratio
    mask2Remove = np.zeros((rows, cols), dtype=bool)
    for credibility in range(1, nCred + 1):
        if credibility <= 4:
            mask2Remove_i, _ = RoI(
                CM[:, :, credibility - 1],
                LCC,
                nWBClasses,
                rNWB[credibility - 1],
                True,
                'contain'
            )
            mask2Remove = mask2Remove | mask2Remove_i
    rasWB[mask2Remove] = 0

    # remove small objects / low compactness
    compactScore, area = compactness(CM)
    for credibility in range(1, nCred + 1):
        if credibility > 4:
            continue
        CMc = CM[:, :, credibility - 1]
        realCred = lstRealCred[credibility - 1]
        if realCred is None or len(realCred) == 0:
            continue

        # area{credibility} / compactScore{credibility} expected from compactness()
        # exact structure depends on your later implementation
        try:
            patchID, areaC = area[credibility - 1][:, 0].astype(int), area[credibility - 1][:, 1]
            compactID, compactC = compactScore[credibility - 1][:, 0].astype(int), compactScore[credibility - 1][:, 1]
        except Exception:
            raise NotImplementedError("compactness() output structure is not yet defined.")

        # map patch -> real credibility
        loc = np.searchsorted(realCred[:, 0], patchID)
        loc = np.clip(loc, 0, len(realCred) - 1)
        credC = realCred[loc, 1]

        thArea = np.asarray(thSize)[credC - 1]
        IDSmall = patchID[areaC < thArea]
        maskSmall = np.isin(CMc, IDSmall)

        thCompC = np.asarray(thCompactness)[credC - 1]
        # align compact ids
        IDVoid = compactID[compactC < thCompC]
        maskVoid = np.isin(CMc, IDVoid)

        rasWB[(maskVoid | maskSmall) & (~np.isnan(rasWB))] = 0

    rasWB[maskOut | (Ipp <= 0)] = np.nan

    # ---- return only the PROCESS core (MATLAB writes rasWB(rs:re,cs:ce)) ----
    core = rasWB[rs:re + 1, cs:ce + 1]

    out = np.full(core.shape, 255, dtype=np.uint8)
    valid = np.isfinite(core) & (core >= 0) & (core < 255)
    out[valid] = core[valid].astype(np.uint8)

    # --- write block result ---
    block_file = os.path.join(
        morph_out_dir,
        f"morph_block_{iR}_{iC}.tif"
    )

    with rasterio.open(
            block_file,
            "w",
            driver="GTiff",
            height=core.shape[0],
            width=core.shape[1],
            count=1,
            dtype=core.dtype,
            transform=block_transform,
            crs=crs
    ) as dst:
        dst.write(core, 1)

    print(f"[morph_block_worker] finished block ({iR + 1},{iC + 1})", flush=True)

    return {
        "iR": iR,
        "iC": iC,
        "file": block_file
    }


def morph_proc_scene_main(scene_pack, blockSize=None):
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

    fileWB_vv = getattr(args_vv, "fileCatInit", os.path.join(args_vv.dirOut, "Cat_init.tif"))
    fileWB_vh = getattr(args_vh, "fileCatInit", os.path.join(args_vh.dirOut, "Cat_init.tif"))

    fileThFine_vv = getattr(args_vv, "fileThFine_root", os.path.join(args_vv.dirTempPol, "ThFine_"))
    fileThFine_vh = getattr(args_vh, "fileThFine_root", os.path.join(args_vh.dirTempPol, "ThFine_"))

    Cpp = np.asarray(ds_vv.sigma2, dtype=float).reshape(-1)
    Cqp = np.asarray(ds_vh.sigma2, dtype=float).reshape(-1)

    nCatPP = int(Cpp.size)
    nCatQP = int(Cqp.size)
    idx_vv = np.argsort(Cpp) + 1
    idx_vh = np.argsort(Cqp) + 1

    if thBimodal.size > 0:
        thPrcBimodal = float(getattr(args_vv, "thPrcBimodal", 50.0))
        thGlobal_db = float(np.percentile(thBimodal, thPrcBimodal))
        thGlobal = 10.0 ** (thGlobal_db / 10.0)
    else:
        thGlobal_db = np.nan
        thGlobal = np.nan

    WBClasses = np.asarray(getattr(args_vv, "WBClasses", [5, 6, 50, 60]), dtype=int)
    EXClasses = np.asarray(getattr(args_vv, "EXClasses", []), dtype=int)
    nWBClasses = np.asarray(getattr(args_vv, "nWBClasses", [8, 80]), dtype=int)

    rNWB = np.asarray(getattr(args_vv, "rNWB_morph", [0.3, 0.4, 0.4, 0.4, 0.4]), dtype=float)
    thSize = np.asarray(getattr(args_vv, "thSize_morph", [10, 10, 10, 10, 10]), dtype=float)
    thCompactness = np.asarray(getattr(args_vv, "thCompactness", [0, 0, 0.25, 1, 2]), dtype=float)

    with rasterio.open(fileWB_vv) as src:
        rowsImage = src.height
        colsImage = src.width
        ref_transform = src.transform
        ref_crs = src.crs

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
            print(f"[morph_proc_scene_main] queue block ({iR+1},{iC+1})/({nR},{nC})", flush=True)

            r0Load = int(r0[iR, 0])
            nRowsLoad = int(nRows[iR, 0])
            c0Load = int(c0[iC, 0])
            nColsLoad = int(nCols[iC, 0])

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
                morph_block_worker(
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
                    EXClasses=EXClasses,
                    nWBClasses=nWBClasses,
                    rNWB=rNWB,
                    thSize=thSize,
                    thCompactness=thCompactness,
                    morph_out_dir=dirMorph,
                    block_transform=ref_transform,
                    crs=ref_crs
                )
            )

    print("[morph_proc_scene_main] submitting morph tasks...", flush=True)
    results = dask.compute(*tasks)
    print("[morph_proc_scene_main] all morph blocks finished", flush=True)

    # ---- mosaic ----
    print("[morph_proc_scene_main] start mosaic", flush=True)

    block_files = [r["file"] for r in results]
    src_files = [rasterio.open(f) for f in block_files]
    try:
        mosaic, out_trans = rasterio.merge.merge(src_files)

        with rasterio.open(
                fileMorphOut,
                "w",
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                count=1,
                dtype=mosaic.dtype,
                crs=src_files[0].crs,
                transform=out_trans,
                nodata=255,
                compress="deflate",
                tiled=True,
        ) as dst:
            dst.write(mosaic)
    finally:
        for src in src_files:
            src.close()
    print("[morph_proc_scene_main] mosaic finished", flush=True)

    print("[morph_proc_scene_main] deleting block files", flush=True)
    for f in block_files:
        try:
            os.remove(f)
        except:
            pass
    print("[morph_proc_scene_main] cleanup finished", flush=True)

    scene_pack["morph_proc"] = {
        "fileMorphOut": fileMorphOut,
        "thGlobal": thGlobal,
        "thGlobal_db": thGlobal_db,
        "isWindy": isWindy,
    }

    print(f"finished morphological process, wrote final mosaic -> {fileMorphOut}")
    return scene_pack
