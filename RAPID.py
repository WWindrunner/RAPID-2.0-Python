import json
import os
import argparse
import random
import datetime
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
# Suppress all warnings
import warnings

warnings.filterwarnings("ignore")

'''
Developed by: Qing Yang, Xinyi Shen. University of Wisconsin-Milwaukee.
Contact: yang2473@uwm.edu

Updated 05/15/2023, Adaptive Initialization.
Created 03/01/2023, RAPID python version.
'''


def mosaic_pre(output):
    '''
    Mosaic the pre-event water mask for the dry reference
    Since we are not directly implementing change detection in terms of backscattering values, 
    the dry and flood images do not need to be in the same geoextent.
    '''
    # if len(output)==1 and output[0][0].MODE == 'normal':
    #    output[0][0].floodimage = False
    #    print('No dry images, consider changing the mode to single!')
    if len(output) == 1 or output[0][0].MODE == 'single':
        output[0][1].pre_mod_mask = da.where(output[0][1].water_occurrence > 50, 1, 0)
        print('Single flood image mode, build pre-event water with ESA water occurrence >50')
    else:
        flood_src = output[0][0].ref_src
        dry_srcs = [out[0].ref_src for out in output[1:]]
        mod_masks = [out[1].Mod_mask for out in output[1:]]
        mosaic_results = []
        for dry_ras, dry_src in zip(mod_masks, dry_srcs):
            dry_src_meta = dry_src.meta.copy()
            dry_src_meta.update(dtype=rasterio.uint8, nodata=255)
            mosaic_results.append(dask.delayed(reproject_clip_direct)(dry_ras, dry_src_meta, flood_src.meta.copy()))
        mosaic_results = dask.compute(*mosaic_results)
        mesaic_raw = da.sum(da.stack(mosaic_results, axis=0), axis=0).squeeze()
        output[0][1].pre_mod_mask = da.where(mesaic_raw > 0, 1, 0)
        del mosaic_results, mesaic_raw
    return output


def parse(config_json=None):
    '''
    Parse the arguments with default values
    Read the config file if provided
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--showDist', default=False)
    parser.add_argument('--Iter', default=False)
    parser.add_argument('--granules', default=[])
    parser.add_argument('--MODE', type=str, default='single')
    parser.add_argument('--thProb', type=float, default=0.87)
    parser.add_argument('--thProbMax', type=float, default=0.89)
    parser.add_argument('--thProbMin', type=float, default=0.87)
    parser.add_argument('--dThProb', type=float, default=0.01)
    parser.add_argument('--thSeed', type=float, default=90)
    parser.add_argument('--NSCE_th', type=float, default=0.2)
    parser.add_argument('--PDFScaling', type=float, default=1)
    parser.add_argument('--th_grid_ratio', type=float, default=0.3)
    parser.add_argument('--ENL', type=float, default=4.9)
    parser.add_argument('--blockSize', default=1024)
    parser.add_argument('--nInter', type=int, default=100)
    parser.add_argument('--morpho_seed_type', type=str, default='both')
    parser.add_argument('--nWBClasses', default=[8, 80])
    parser.add_argument('--WBClasses', default=[5, 6, 50, 60])
    parser.add_argument('--thSensPre', type=float, default=0.2)
    parser.add_argument('--thSensIn', type=float, default=0.1)
    parser.add_argument('--rNWB_WST_Org', type=float, default=0.3)
    parser.add_argument('--rNWB_WST', type=float, default=0.4)
    parser.add_argument('--rNWB_ICD', type=float, default=0.8)
    parser.add_argument('--thSize', type=int, default=10)
    parser.add_argument('--thInund', type=float, default=0.3)
    parser.add_argument('--rInund', type=float, default=1)
    parser.add_argument('--rProb', type=float, default=0.5)
    parser.add_argument('--imported', default=False)
    parser.add_argument('--optIterAlg', default=True)
    parser.add_argument('--log_file_err', type=str, default='')
    parser.add_argument('--showProgress', default=False)
    parser.add_argument('--bInitialized', default=False)
    parser.add_argument('--bExport', default=True)
    parser.add_argument('--floodimage', default=False)
    parser.add_argument('--failed', default=False)

    # parser.add_argument('--blockSizeSBA', type=int, default=1024,
    #                     help='Block size for SBA window creation. Default 1024.')

    parser.add_argument('--overlapRatio', type=float, default=0.3,
                        help='Overlap ratio for SBA windows (0-1). Default 0.3.')

    parser.add_argument('--thOccLow', type=float, default=5,
                        help='Lower bound of water occurrence threshold (0-100). Default 5.')
    parser.add_argument('--thOccHigh', type=float, default=99,
                        help='Upper bound of water occurrence threshold (0-100). Default 99.')

    parser.add_argument('--thMinSampleSize', type=int, default=500,
                        help='Minimum number of seed pixels required to start sampling within a tile. Default 500.')

    parser.add_argument('--thIPrc', type=float, default=50,
                        help='Percentile used as reference in the stopping test (e.g., 50=median). Default 50.')

    parser.add_argument('--thFrac', type=float, default=0.6,
                        help='Stopping fraction for newly added pixels being higher than I_prc. '
                             'Default 0.6 (matches thIPrc=50 -> 1-0.5+0.1).')

    parser.add_argument('--binWidthInit', type=float, default=0.2,
                        help='Histogram bin width (dB) for peakWater estimation. Default 0.2.')

    parser.add_argument('--codePW', type=int, default=80,
                        help='ESA WorldCover class code for water (PW). Default 80.')

    parser.add_argument('--showProg', action='store_true',
                        help='Show progress percentage for SBA seed selection.')
    parser.add_argument('--noShowProg', action='store_true',
                        help='Disable progress display (overrides --showProg).')

    args, unparsed = parser.parse_known_args()
    # pack low/high into one tuple like MATLAB thOccLim=[5;99]
    args.thOccLim = (args.thOccLow, args.thOccHigh)

    # if user changed thIPrc but didn't change thFrac, optionally auto-derive thFrac
    # (keep explicit thFrac if you prefer; this is the safest behavior)
    args.thFrac = float(args.thFrac)

    # showProg flags priority
    if getattr(args, "noShowProg", False):
        args.showProg = False
    else:
        # if neither flag given, showProg will be False by default unless you set it elsewhere
        args.showProg = bool(getattr(args, "showProg", False))

    if config_json is not None:
        with open(config_json, 'r') as f:
            loaded_args = json.load(f)
        for arg in loaded_args:
            setattr(args, arg, loaded_args[arg])
    return args


def dup_args(args, filepath, granuele_name, wotif=None):
    '''
    Duplicate the arguments for each input SAR image (dual-pol)
    '''
    args_new = argparse.Namespace(**vars(args))
    args_new.filepath = filepath
    args_new.wotif = wotif
    args_new.dirOut = os.path.join(args_new.WORK_PATH, granuele_name)
    args_new.fileSampleMap = os.path.join(args_new.dirOut, 'SampleMap.tif')  # intermedian result, not used for now
    args_new.logFile = os.path.join(args_new.dirOut, 'sta_log.csv')
    os.makedirs(args_new.dirOut, exist_ok=True)
    if args.blockSize == 'auto':
        args_new.chunks = 'auto'
    else:
        args_new.chunks = args.blockSize
    return args_new


@tim
def RAPID(*args, **kwargs):
    if len(args) > 1 and len(args) <= 4:
        # input SAR files path and output folder from terminal
        # keep the last one as output folder
        filepath = list(args[:-1])
        OUT_PATH = args[-1]
        config_file = None
        args = parse(config_file)
        args.OUT_PATH = OUT_PATH
        args.filepath_list = [filepath]
    elif len(args) == 1:
        # input geojson file
        # mode and dry referen should be specified in config file
        config_file = args[0]
        if not os.path.exists(config_file):
            raise ValueError(f"{config_file} does not exist!")
        args = parse(config_file)
    else:
        raise ValueError("Wrong number of input arguments!")

    # Refine ouput path if already exists
    raw_out_path = os.path.join(args.OUT_PATH, os.path.splitext(args.filepath_list[0][0].split('/')[-1])[0])
    if os.path.exists(raw_out_path):
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        rand_num = random.randint(1, 1000000)
        args.WORK_PATH = raw_out_path + f'_{now_str}_{rand_num}'
    else:
        args.WORK_PATH = raw_out_path
    os.makedirs(args.WORK_PATH, exist_ok=True)
    filepath_list = args.filepath_list
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Initial args list for each image
    args_list = []
    for idx, filepath in enumerate(filepath_list):
        # set granule name
        if len(filepath_list) == len(args.granules):
            granuele_name = args.granules[idx]
            args1 = dup_args(args, filepath, granuele_name, wotif=None)
            args1.granules = granuele_name
        else:
            granuele_name = os.path.splitext(filepath[0].split('/')[-1])[0]
            args1 = dup_args(args, filepath, granuele_name, wotif=None)
            args1.granules = granuele_name
        args_list.append(args1)
    # set flood image
    args_list[0].floodimage = True

    if args_list[0].MODE == 'swd':
        '''
        Self-supervised Water Detection. An alternative algorithm for test purpose.
        '''
        print('Running swd!')
        output = dask.compute(*[dask.delayed(Load_Data)(args_list[0])])
        output[0][1].Compensate_mask, output[0][0].fileOutCompensate, output[0][1].direct_mask = swd(output[0][0],
                                                                                                     output[0][1])
        output[0][0].MODE = 'single'
        out_filepath = quick_look(output)
    else:
        # load image and get ancillary data
        with Timer('Load_Data Total'):
            output = dask.compute(*[dask.delayed(Load_Data)(arggs) for arggs in args_list])
            # output = [[args_list[i], ds_dask[i]] for i in range(len(args_list))]
        if args_list[0].MODE != 'desert':
            # Normal mode
            with Timer('TOTAL BC TIME'):
                output = dask.compute(*[dask.delayed(Binary_classification)(*single_out) for single_out in output])

            with Timer('Mosaic_pre TIME'):
                output = mosaic_pre(output)
            with Timer('Morph_proc'):
                output = dask.compute(*[dask.delayed(Morph_proc)(*single_out) for single_out in output])
            # with Timer('Compensation'):
            #    output = dask.compute(*[dask.delayed(Compensate)(*single_out) for single_out in output])
            # produce quick look figures
            out_filepath = quick_look(output)
        else:
            print('Running desert mode!')
            args_ndfi, ds_ndfi = NDFI(output)
            args_ndfi, ds_ndfi = NDFI_morph(args_ndfi, ds_ndfi)
            args_ndfi, ds_ndfi = NDFI_compensate(args_ndfi, ds_ndfi)


if __name__ == '__main__':
    RAPID(*sys.argv[1:])
