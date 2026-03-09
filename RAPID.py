import json
import os
import argparse
import random
import datetime
import copy
import rasterio
from utils import *
from Load_Data import *
from Binary_classification import *
from Morpho_process import *
from Compensation import *
from NDFI import *
from quick_look import *
from Self_supervised import *
import subprocess
import warnings
import dask

warnings.filterwarnings("ignore")

'''
Developed by: Maopu Xu, Qing Yang, Xinyi Shen. University of Wisconsin-Milwaukee.

Modified for scene-level dual-pol workflow:
- one scene = one Load_Data
- Binary_classification runs separately for VV and VH
- subsequent processing works on the scene package
'''


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse(config_json=None):
    """
    Parse the arguments with default values.
    Read the config file if provided.
    Compatible with old RAPID parameters + current SBA/bimodality parameters.
    """
    parser = argparse.ArgumentParser()

    # ---- existing / legacy args ----
    parser.add_argument('--showDist', type=_str2bool, default=False)
    parser.add_argument('--Iter', type=_str2bool, default=False)
    parser.add_argument('--granules', nargs='*', default=[])

    parser.add_argument('--MODE', type=str, default='single')
    parser.add_argument('--thProb', type=float, default=0.87)
    parser.add_argument('--thProbMax', type=float, default=0.89)
    parser.add_argument('--thProbMin', type=float, default=0.87)
    parser.add_argument('--dThProb', type=float, default=0.01)
    parser.add_argument('--thSeed', type=float, default=90)
    parser.add_argument('--NSCE_th', type=float, default=0.2)
    parser.add_argument('--PDFScaling', type=float, default=1.0)
    parser.add_argument('--th_grid_ratio', type=float, default=0.3)
    parser.add_argument('--ENL', type=float, default=4.9)

    parser.add_argument('--blockSize', type=int, default=1024)
    parser.add_argument('--nCores', type=int, default=1)
    parser.add_argument('--nInter', type=int, default=100)
    parser.add_argument('--morpho_seed_type', type=str, default='both')

    parser.add_argument('--nWBClasses', nargs='*', type=int, default=[8, 80])
    parser.add_argument('--WBClasses', nargs='*', type=int, default=[5, 6, 50, 60])

    parser.add_argument('--thSensPre', type=float, default=0.2)
    parser.add_argument('--thSensIn', type=float, default=0.1)
    parser.add_argument('--rNWB_WST_Org', type=float, default=0.3)
    parser.add_argument('--rNWB_WST', type=float, default=0.4)
    parser.add_argument('--rNWB_ICD', type=float, default=0.8)
    parser.add_argument('--thSize', type=int, default=10)
    parser.add_argument('--thInund', type=float, default=0.3)
    parser.add_argument('--rInund', type=float, default=1.0)
    parser.add_argument('--rProb', type=float, default=0.5)

    parser.add_argument('--imported', type=_str2bool, default=False)
    parser.add_argument('--optIterAlg', type=_str2bool, default=True)
    parser.add_argument('--log_file_err', type=str, default='')
    parser.add_argument('--showProgress', type=_str2bool, default=False)
    parser.add_argument('--bInitialized', type=_str2bool, default=False)
    parser.add_argument('--bExport', type=_str2bool, default=True)
    parser.add_argument('--floodimage', type=_str2bool, default=False)
    parser.add_argument('--failed', type=_str2bool, default=False)

    # ---- SBA / bimodality / sampling ----
    parser.add_argument('--overlapRatio', type=float, default=0.3,
                        help='Overlap ratio for SBA windows (0-1). Default 0.3.')
    parser.add_argument('--thOccLow', type=float, default=5,
                        help='Lower bound of water occurrence threshold (0-100). Default 5.')
    parser.add_argument('--thOccHigh', type=float, default=99,
                        help='Upper bound of water occurrence threshold (0-100). Default 99.')
    parser.add_argument('--thMinSampleSize', type=int, default=500,
                        help='Minimum number of seed pixels required to start sampling within a tile. Default 500.')
    parser.add_argument('--thIPrc', type=float, default=50,
                        help='Percentile used as reference in the stopping test. Default 50.')
    parser.add_argument('--thFrac', type=float, default=0.6,
                        help='Stopping fraction. Default 0.6.')
    parser.add_argument('--binWidthInit', type=float, default=0.2,
                        help='Histogram bin width (dB) for peakWater estimation. Default 0.2.')
    parser.add_argument('--codePW', type=int, default=80,
                        help='ESA WorldCover class code for water. Default 80.')
    parser.add_argument('--min_windowsWater', type=int, default=20,
                        help='Minimum number of initialization windows required. Default 20.')
    parser.add_argument('--rec_windowsWater', type=int, default=150,
                        help='Recommended number of initialization windows to keep. Default 150.')

    parser.add_argument('--writeSampleMap', type=_str2bool, default=True,
                        help='Write initial SBA sample maps (default: True).')
    parser.add_argument('--showProg', type=_str2bool, default=True,
                        help='Show progress percentage for SBA seed selection.')

    parser.add_argument('--thPC', nargs=2, type=float, default=[0.001, 0.999],
                        help='PC range for valid GMM. Two floats: low high.')
    parser.add_argument('--thAD', type=float, default=0.0)
    parser.add_argument('--thRH', type=float, default=0.5)
    parser.add_argument('--thDist2Mean', type=float, default=0.1)

    parser.add_argument('--maxENL', type=float, default=7.0,
                        help='Upper bound of ENL search in findInitDist. Default 7.0.')
    parser.add_argument('--thKLD', type=float, default=0.2,
                        help='Remove categories if KLD > thKLD. Default 0.2.')
    parser.add_argument('--thNSE', type=float, default=0.7,
                        help='Remove categories if NSE < thNSE. Default 0.7.')
    parser.add_argument('--thR2', type=float, default=0.8,
                        help='Remove categories if R2 < thR2. Default 0.8.')
    parser.add_argument('--writeSampleDist', type=_str2bool, default=True,
                        help='Write chi-square vs empirical distribution figures for each category. Default True.')

    # -----------------------------
    # dist_th / worker thresholds
    # -----------------------------
    parser.add_argument('--distThKLD', nargs=2, type=float, default=[0.1, 0.2],
                        help='KLD thresholds used in dist_th worker. Two floats: strict loose. Default [0.1, 0.2].')

    parser.add_argument('--distThR', nargs=2, type=float, default=[0.9, 0.7],
                        help='Correlation thresholds used in dist_th worker. Two floats: strict loose. Default [0.9, 0.7].')

    parser.add_argument('--distThNSE', nargs=2, type=float, default=[0.80, 0.40],
                        help='NSE thresholds used in dist_th worker. Two floats: strict loose. Default [0.80, 0.40].')

    parser.add_argument('--distThDist2Mean', type=float, default=0.1,
                        help='Distance-to-mean threshold used in dist_th bimodality test. Default 0.1.')

    parser.add_argument('--distThSampleSize', type=int, default=50,
                        help='Minimum sample size used in dist_th worker. Default 50.')

    parser.add_argument('--distThRPW', nargs=2, type=float, default=[0.01, 0.9],
                        help='PW ratio thresholds [hasPW, isPW] used in dist_th worker. Default [0.01, 0.9].')

    parser.add_argument('--distThPC', nargs=2, type=float, default=[0.02, 0.98],
                        help='PC range used in dist_th bimodality GMM validation. Default [0.02, 0.98].')

    parser.add_argument('--distThAD', type=float, default=0.9,
                        help='AD threshold used in dist_th bimodality GMM validation. Default 0.9.')

    parser.add_argument('--distThRH', nargs=2, type=float, default=[1.0, 1.0],
                        help='RH thresholds [wind_inclusive, glacial_exclusive] used in dist_th bimodality GMM validation. Default [1.0, 1.0].')

    # ---- current workflow / io ----
    parser.add_argument('--OUT_PATH', type=str, default='.')
    parser.add_argument('--filepath_list', nargs='*', default=[])

    args, unparsed = parser.parse_known_args()

    # ---- normalize / derived fields ----
    args.thOccLim = (float(args.thOccLow), float(args.thOccHigh))
    args.blockSize = int(args.blockSize)
    args.chunks = args.blockSize

    args.thFrac = float(args.thFrac)
    args.showProg = bool(args.showProg)
    args.writeSampleMap = bool(args.writeSampleMap)
    args.thPC = (float(args.thPC[0]), float(args.thPC[1]))
    args.maxENL = float(args.maxENL)
    args.thKLD = float(args.thKLD)
    args.thNSE = float(args.thNSE)
    args.thR2 = float(args.thR2)
    args.writeSampleDist = bool(args.writeSampleDist)

    args.distThKLD = (float(args.distThKLD[0]), float(args.distThKLD[1]))
    args.distThR = (float(args.distThR[0]), float(args.distThR[1]))
    args.distThNSE = (float(args.distThNSE[0]), float(args.distThNSE[1]))
    args.distThDist2Mean = float(args.distThDist2Mean)
    args.distThSampleSize = int(args.distThSampleSize)
    args.distThRPW = (float(args.distThRPW[0]), float(args.distThRPW[1]))
    args.distThPC = (float(args.distThPC[0]), float(args.distThPC[1]))
    args.distThAD = float(args.distThAD)
    args.distThRH = (float(args.distThRH[0]), float(args.distThRH[1]))

    # ---- config override ----
    if config_json is not None:
        with open(config_json, 'r') as f:
            loaded_args = json.load(f)

        for k, v in loaded_args.items():
            setattr(args, k, v)

        # normalize again after override
        args.Iter = _str2bool(getattr(args, "Iter", args.Iter))
        args.showDist = _str2bool(getattr(args, "showDist", args.showDist))
        args.writeSampleMap = _str2bool(getattr(args, "writeSampleMap", args.writeSampleMap))
        args.showProg = _str2bool(getattr(args, "showProg", args.showProg))
        args.showProgress = _str2bool(getattr(args, "showProgress", args.showProgress))
        args.imported = _str2bool(getattr(args, "imported", args.imported))
        args.optIterAlg = _str2bool(getattr(args, "optIterAlg", args.optIterAlg))
        args.bInitialized = _str2bool(getattr(args, "bInitialized", args.bInitialized))
        args.bExport = _str2bool(getattr(args, "bExport", args.bExport))
        args.floodimage = _str2bool(getattr(args, "floodimage", args.floodimage))
        args.failed = _str2bool(getattr(args, "failed", args.failed))
        args.writeSampleDist = _str2bool(getattr(args, "writeSampleDist", args.writeSampleDist))

        args.thOccLow = float(getattr(args, "thOccLow", args.thOccLow))
        args.thOccHigh = float(getattr(args, "thOccHigh", args.thOccHigh))
        args.thOccLim = (args.thOccLow, args.thOccHigh)

        args.thPC = tuple(getattr(args, "thPC", args.thPC))
        args.blockSize = int(getattr(args, "blockSize", args.blockSize))
        args.chunks = args.blockSize

    return args


def dup_args(args, filepath, granuele_name, wotif=None):
    """
    Duplicate the arguments for each scene.
    NOTE:
    filepath is now treated as one SCENE input:
      - local dual-pol: [vv_path, vh_path]
      - ASF auto-download: [url]
    """
    args_new = argparse.Namespace(**vars(args))
    args_new.filepath = filepath
    args_new.wotif = wotif

    args_new.dirOut = os.path.join(args_new.WORK_PATH, granuele_name)
    args_new.fileSampleMap = os.path.join(args_new.dirOut, 'SampleMap.tif')
    args_new.logFile = os.path.join(args_new.dirOut, 'sta_log.csv')

    # scene-level temp root
    args_new.dirTempPol = os.path.join(args_new.dirOut, "TempPol")
    args_new.fileThCoarse_root = os.path.join(args_new.dirTempPol, "ThCoarse_")
    args_new.fileThFine_root = os.path.join(args_new.dirTempPol, "ThFine_")
    args_new.fileCatInit = os.path.join(args_new.dirOut, "Cat_init.tif")

    os.makedirs(args_new.dirTempPol, exist_ok=True)
    os.makedirs(args_new.dirOut, exist_ok=True)

    if args.blockSize == 'auto':
        args_new.chunks = 'auto'
    else:
        args_new.chunks = args.blockSize

    return args_new


def setup_dask_from_slurm():
    """
    Setup Dask LocalCluster based on SLURM environment.

    Default: THREAD MODE
    Can switch to process mode by setting:
        export DASK_USE_PROCESSES=1
    """
    import os
    try:
        from dask.distributed import Client, LocalCluster
    except Exception as e:
        print("WARNING: dask.distributed not available, fallback to default scheduler.", e)
        return None

    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    use_processes = os.environ.get("DASK_USE_PROCESSES", "0") == "1"

    if not use_processes:
        n_workers = 1
        threads_per_worker = ncpus
        if mem_mb > 0:
            mem_limit = f"{int(mem_mb * 0.85)}MB"
        else:
            mem_limit = "70GB"

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=False,
            memory_limit=mem_limit,
            dashboard_address=":8787",
        )
        client = Client(cluster)

        print("Dask LocalCluster started (THREAD MODE).")
        print("  workers:", n_workers)
        print("  threads_per_worker:", threads_per_worker)
        print("  memory_limit:", mem_limit)
        print("  dashboard:", client.dashboard_link)
    else:
        n_workers = int(os.environ.get("DASK_NWORKERS", str(ncpus)))
        threads_per_worker = int(os.environ.get("DASK_THREADS_PER_WORKER", "1"))
        mem_limit = os.environ.get("DASK_MEM_LIMIT", "4GB")

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=mem_limit,
            dashboard_address=":8787",
        )
        client = Client(cluster)

        print("Dask LocalCluster started (PROCESS MODE).")
        print("  workers:", n_workers)
        print("  threads_per_worker:", threads_per_worker)
        print("  memory_limit:", mem_limit)
        print("  dashboard:", client.dashboard_link)

    return client


def _make_pol_args(args_scene, pol):
    """
    Build per-polarization args so Binary_classification can run twice safely.
    """
    args_pol = argparse.Namespace(**vars(args_scene))
    args_pol.pol = pol

    args_pol.dirTempPol = os.path.join(args_scene.dirOut, f"TempPol_{pol}")
    os.makedirs(args_pol.dirTempPol, exist_ok=True)

    args_pol.fileSampleMap = os.path.join(args_pol.dirTempPol, f"SampleMap_{pol}.tif")
    args_pol.fileThCoarse_root = os.path.join(args_pol.dirTempPol, f"ThCoarse_{pol}_")
    args_pol.fileThFine_root = os.path.join(args_pol.dirTempPol, f"ThFine_{pol}_")
    args_pol.fileCatInit = os.path.join(args_pol.dirTempPol, f"Cat_init_{pol}.tif")

    # optional files for later stages / debugging
    args_pol.fileSBAInitUnCat = os.path.join(args_pol.dirTempPol, f"SBAInitUnCat_{pol}.mat")
    args_pol.fileOutBinary = os.path.join(args_pol.dirTempPol, f"Binary_{pol}.tif")

    return args_pol


def _make_single_pol_ds(ds_scene, pol):
    """
    Create a single-pol view of scene ds.
    Binary_classification only sees one polarization as img_lp.
    Shared ancillary arrays are kept.
    """
    class c:
        pass

    ds_pol = c()

    # copy all attributes except image arrays first
    for k, v in vars(ds_scene).items():
        if k in ('img_lp', 'img_cp'):
            continue
        setattr(ds_pol, k, v)

    if pol == "VV":
        ds_pol.img_lp = ds_scene.img_lp
    elif pol == "VH":
        ds_pol.img_lp = ds_scene.img_cp
    else:
        raise ValueError(f"Unsupported pol: {pol}")

    # keep a placeholder img_cp in case some downstream function expects it
    ds_pol.img_cp = ds_pol.img_lp

    return ds_pol


def _infer_scene_name(filepath):
    """
    Infer a scene name from one scene input.

    filepath is one scene entry:
      - local dual-pol: [vv_path, vh_path]
      - local single file: [file_path]
      - remote ASF url: [url]
    """
    first = filepath[0]

    # local multi-file scene: use parent folder name
    if len(filepath) > 1 and not first.startswith("http"):
        return os.path.basename(os.path.dirname(first))

    # remote url: use zip basename without extension
    if first.startswith("http"):
        base = os.path.basename(first)
        for ext in [".zip", ".SAFE", ".tar", ".gz", ".tif"]:
            if base.endswith(ext):
                base = base[:-len(ext)]
        return os.path.splitext(base)[0]

    # local single file: use file stem
    return os.path.splitext(os.path.basename(first))[0]


@tim
def Binary_classification_dualpol(args_scene, ds_scene):
    """
    Run Binary_classification twice on one scene:
      - once for VV
      - once for VH (if available)
    Return a scene package for later morphology/fusion.
    """
    print(f"[Binary_classification_dualpol] start scene: {args_scene.granules}", flush=True)

    scene_pack = {
        "scene_args": args_scene,
        "scene_ds": ds_scene,
        "pol_results": {}
    }

    pols = ["VV"]
    if getattr(args_scene, "dual_pol", False):
        pols.append("VH")

    for pol in pols:
        print(f"[Binary_classification_dualpol] start {pol}", flush=True)
        args_pol = _make_pol_args(args_scene, pol)
        ds_pol = _make_single_pol_ds(ds_scene, pol)

        args_pol, ds_pol = Binary_classification(args_pol, ds_pol)

        scene_pack["pol_results"][pol] = (args_pol, ds_pol)
        print(f"[Binary_classification_dualpol] done {pol}", flush=True)

    print(f"[Binary_classification_dualpol] finished scene: {args_scene.granules}", flush=True)
    return scene_pack


@tim
def Morph_proc_scene(scene_pack):
    """
    Scene-level morphology wrapper.

    Current temporary rule:
      - if both VV and VH exist, keep both results in the package
      - call old Morph_proc only on VV branch as a temporary placeholder
      - you can later replace this with a real VV/VH fusion strategy

    Because we have not rewritten Morpho_process.py yet, this wrapper is only
    defining the scene-level interface.
    """
    if "VV" not in scene_pack["pol_results"]:
        raise ValueError("VV result is required in scene_pack['pol_results'].")

    args_vv, ds_vv = scene_pack["pol_results"]["VV"]

    print("[Morph_proc_scene] temporary behavior: run old Morph_proc on VV only", flush=True)
    args_vv, ds_vv = Morph_proc(args_vv, ds_vv)
    scene_pack["pol_results"]["VV"] = (args_vv, ds_vv)

    return scene_pack


@tim
def quick_look_scene(scene_output):
    """
    Temporary quick-look wrapper.

    Current behavior:
      - use VV branch quick_look-compatible output only
      - keep scene-level structure unchanged for later dual-pol fusion
    """
    vv_only_output = []
    for scene_pack in scene_output:
        if "VV" in scene_pack["pol_results"]:
            vv_only_output.append(scene_pack["pol_results"]["VV"])
    return quick_look(vv_only_output)


@tim
def RAPID(*args, **kwargs):
    if len(args) > 1 and len(args) <= 4:
        # terminal input:
        #   python RAPID.py vv.tif vh.tif out_dir
        # or
        #   python RAPID.py asf_url out_dir
        filepath = list(args[:-1])
        OUT_PATH = args[-1]
        config_file = None
        args = parse(config_file)
        args.OUT_PATH = OUT_PATH
        args.filepath_list = [filepath]
    elif len(args) == 1:
        # config / geojson style input
        config_file = args[0]
        if not os.path.exists(config_file):
            raise ValueError(f"{config_file} does not exist!")
        args = parse(config_file)
    else:
        raise ValueError("Wrong number of input arguments!")

    # output root
    args.WORK_PATH = args.OUT_PATH
    os.makedirs(args.WORK_PATH, exist_ok=True)

    filepath_list = args.filepath_list

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # dask
    client = setup_dask_from_slurm()

    # ----------------------------------
    # Build scene args list
    # Each filepath entry is one scene:
    #   [vv, vh]  or  [url]
    # ----------------------------------
    args_list = []
    for idx, filepath in enumerate(filepath_list):
        if len(filepath_list) == len(args.granules):
            granuele_name = args.granules[idx]
        else:
            granuele_name = _infer_scene_name(filepath)

        args1 = dup_args(args, filepath, granuele_name, wotif=None)
        args1.granules = granuele_name
        args_list.append(args1)

    if len(args_list) == 0:
        raise ValueError("No scene input found in filepath_list.")

    # mark the first scene as flood image if old code still checks this flag
    args_list[0].floodimage = True

    if args_list[0].MODE == 'swd':
        print('Running swd!')
        output = dask.compute(*[dask.delayed(Load_Data)(args_list[0])])
        output[0][1].Compensate_mask, output[0][0].fileOutCompensate, output[0][1].direct_mask = swd(
            output[0][0], output[0][1]
        )
        output[0][0].MODE = 'single'
        out_filepath = quick_look(output)
        return out_filepath

    if args_list[0].MODE == 'desert':
        print('Running desert mode!')
        output = dask.compute(*[dask.delayed(Load_Data)(arggs) for arggs in args_list])
        args_ndfi, ds_ndfi = NDFI(output)
        args_ndfi, ds_ndfi = NDFI_morph(args_ndfi, ds_ndfi)
        args_ndfi, ds_ndfi = NDFI_compensate(args_ndfi, ds_ndfi)
        return args_ndfi, ds_ndfi

    # ----------------------------------
    # WORKFLOW:
    # scene-level Load_Data
    # scene-level dual-pol Binary
    # scene-level Morph
    # ----------------------------------
    with Timer('Load_Data Total'):
        scene_loaded = dask.compute(*[dask.delayed(Load_Data)(arggs) for arggs in args_list])

    with Timer('TOTAL BC TIME'):
        scene_output = []
        for args_scene, ds_scene in scene_loaded:
            scene_output.append(Binary_classification_dualpol(args_scene, ds_scene))

    with Timer('Morph_proc_scene'):
        scene_output = [Morph_proc_scene(scene_pack) for scene_pack in scene_output]

    out_filepath = quick_look_scene(scene_output)
    return out_filepath


if __name__ == '__main__':
    RAPID(*sys.argv[1:])