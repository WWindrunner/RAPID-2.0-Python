import os
import numpy as np
from utils import *
from AncillaryData import *
import tempfile
import dask
import dask.array as da
import rasterio
import zipfile
from io import BytesIO
import requests
from scipy import ndimage


def tosigmma(arr):
    return np.power(10, (2 * np.log10(arr) - 8.3))


def readfileio(file, chunks=(1, 512, 512)):
    # Open the GeoTIFF file using rasterio
    with rasterio.open(file) as src:
        # Read the image in chunks using dask.array
        img = da.from_array(src.read(1), chunks=chunks)
    return (img)


def readhdf(file, chunks=(1, 512, 512)):
    # Not use anymore, 05012023
    # f = h5py.File(file)
    f = None
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    d = f[a_group_key]
    x = da.from_array(d, chunks=chunks)
    return x


def ZipreadMemorTemp(src, args):
    if args.datamode == 'mem':
        temp_lp = da.from_array(src.read(1), chunks=args.chunks)
    else:
        kwargs = src.meta.copy()
        temp_lp = tempfile.NamedTemporaryFile()
        print(temp_lp.name)
        print('open temp_lp.name')
        with rasterio.open(temp_lp.name, 'w', **kwargs) as dst:
            dst.write(src.read(1))
    return temp_lp


def VV_VHread(zip_file, file_name, args, ds, newfilepath, VHindex):
    with zip_file.open(file_name) as tiff_file:
        with rasterio.open(BytesIO(tiff_file.read())) as src:
            temp_lp = ZipreadMemorTemp(src, args)
            args.ref_src = src
            print(temp_lp)
            if args.datamode == 'tem':
                args.tempfiles.append(temp_lp)
                newfilepath[VHindex] = temp_lp.name
            else:
                newfilepath[VHindex] = src
                setattr(ds, args.varnames[VHindex], temp_lp)
    return args, ds, newfilepath


def inc_read(zip_file, file_name, args, ds):
    with zip_file.open(file_name) as tiff_file:
        with rasterio.open(BytesIO(tiff_file.read())) as src:
            temp_lp = ZipreadMemorTemp(src, args)
            print(temp_lp)
            if args.datamode == 'tem':
                args.tempfiles.append(temp_lp)
            else:
                setattr(ds, 'inc', temp_lp)
    return args, ds


@tim
def retrieveRTC(url, args, ds):
    args.tempfiles = []
    newfilepath = [None] * 2
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    for file_name in zip_file.namelist():
        if 'VV' in file_name or 'HH' in file_name:
            if '.xml' not in file_name:
                print('reading VV or HH ' + file_name)
                args, ds, newfilepath = VV_VHread(zip_file, file_name, args, ds, newfilepath, 0)
        if 'VH' in file_name or 'HV' in file_name:
            if '.xml' not in file_name:
                print('reading VH or HV ' + file_name)
                args, ds, newfilepath = VV_VHread(zip_file, file_name, args, ds, newfilepath, 1)
        if 'inc_map' in file_name:
            if '.xml' not in file_name:
                print('reading incidence angle ' + file_name)
                args, ds = inc_read(zip_file, file_name, args, ds)
    return newfilepath, args, ds


def filepathreader(ds, varnames, infilepath, chunks):
    with rasterio.open(infilepath[0]) as src:
        # Access the number of bands
        num_bands = src.count
    if num_bands > 1:
        # VV, VH, Inc angle in one file. 
        # make sure the Inc is the last band
        with rasterio.open(infilepath[0]) as src:
            for idx, name in enumerate(varnames):
                setattr(ds, name, da.from_array(src.read(idx + 1), chunks=chunks))
    else:
        varnames = ['img_lp', 'img_cp']
        pol_index = 0
        for idx, filename in enumerate(infilepath):
            if 'inc' in infilepath[idx] or 'INC' in infilepath[idx]:
                setattr(ds, 'inc', readfileio(filename, chunks))
            else:
                setattr(ds, varnames[pol_index], readfileio(filename, chunks))
                if 'amp' in filename or 'AMP' in filename:
                    print(f"Transfer to sigmma: {filename}")
                    setattr(ds, varnames[pol_index], tosigmma(getattr(ds, varnames[pol_index])))
                pol_index += 1
    return ds


def get_src(filepath, args):
    # Open the GeoTIFF file using rasterio
    with rasterio.open(filepath) as src:
        args.refmeta = src.meta.copy()
        args.transform = src.transform
        args.crs_wkt = src.crs.to_wkt() if src.crs else None
        args.height = src.height
        args.width = src.width
        args.nodata = src.nodata
        args.count = src.count

    args.ref_src = None
    return args


def set_nan(ds, args):
    ds.img_lp = da.where(ds.img_lp == args.nodata, 0, ds.img_lp)
    ds.img_cp = da.where(ds.img_cp == args.nodata, 0, ds.img_cp)
    print('set nan for BRD noise by erosion with 5 pixels')
    valid_mask = da.where(ds.img_lp > 0, 1, 0)
    valid_BRD_mask = ndimage.binary_erosion(valid_mask, iterations=5)
    ds.img_lp = da.where(valid_BRD_mask == 1, ds.img_lp, 0)
    ds.img_cp = da.where(valid_BRD_mask == 1, ds.img_cp, 0)
    del valid_mask, valid_BRD_mask
    return ds


def normalize_inc(ds):
    if ds.inc is not None:
        print('Normalizing incidence angle')
        ds.img_lp = ds.img_lp / np.cos(ds.inc) ** 2
        ds.img_cp = ds.img_cp / np.cos(ds.inc) ** 2
    return ds


# Convert the ESA LCC value to GP LCC
def ESALCC2GPLCC(arr):
    '''
    Convert the ESA LCC value to GP LCC, 05/08/2023
    So we don't have to change the predefined LCC value
    '''
    new_arr = np.zeros_like(arr)
    # Define the original and target value ranges
    origin_LCC = [40, 10, 30, 20, 90, 80, 100, 50, 60, 70, 95]
    Valuerange_LCC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 20]
    for i in range(len(origin_LCC)):
        new_arr[(arr == origin_LCC[i])] = Valuerange_LCC[i]
    return new_arr


@tim
def Load_Data(args):
    filepath = args.filepath
    wotif = args.wotif
    LCCtif = None
    oceantif = None
    args.varnames = ['img_lp', 'img_cp', 'inc']

    class c:
        pass

    ds = c()
    # initial inc
    ds.inc = None
    args.datamode = 'mem'
    from urllib.parse import urlparse
    if urlparse(filepath[0]).scheme == 'https' and '.tif' not in filepath[0]:
        print('Input with ASF url ' + filepath[0])
        infilepath, args, ds = retrieveRTC(filepath[0], args, ds)
        print(infilepath)
        if args.datamode != 'mem':
            ds = filepathreader(ds, args.varnames, infilepath, args.chunks)
        args = get_src(infilepath[0], args)
    else:
        infilepath = filepath
        args = get_src(filepath[0], args)
        args.tempfiles = []
        ds = filepathreader(ds, args.varnames, infilepath, args.chunks)
    # Output are binary mask so just one band
    args.refmeta.update(count=1)
    # Duplicate ds.img_lp to ds.img_cp if ds.img_cp is not exist, 08-05-2023
    if not hasattr(ds, 'img_cp'):
        ds.img_cp = ds.img_lp.copy()
        args.dual_pol = False
        print('Single-pol SAR input!')
    else:
        args.dual_pol = True
    #
    ds = set_nan(ds, args)
    ds = normalize_inc(ds)

    # water occurrence file
    if wotif == None:
        if args.datamode == 'mem':
            setattr(ds, 'water_occurrence',
                    AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getWO().squeeze())
            # args.tempfiles = None
        else:
            tempfileWO = AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getWO()
            setattr(ds, 'water_occurrence', readhdf(tempfileWO.name, args.chunks).squeeze())
            args.tempfiles.append(tempfileWO)
    else:
        setattr(ds, 'water_occurrence', readfileio(wotif, args.chunks))
        # args.tempfiles = None

    # align chunks before combining arrays, to avoid repeated rechunk-merge graphs later
    if hasattr(ds.water_occurrence, "rechunk"):
        ds.water_occurrence = ds.water_occurrence.rechunk(ds.img_lp.chunks)

    if hasattr(ds.img_cp, "rechunk") and ds.img_cp.chunks != ds.img_lp.chunks:
        ds.img_cp = ds.img_cp.rechunk(ds.img_lp.chunks)

    mask_wo = ds.water_occurrence > 100
    zero_lp = da.zeros_like(ds.img_lp)
    zero_cp = da.zeros_like(ds.img_cp)

    ds.img_lp = da.where(mask_wo, zero_lp, ds.img_lp)
    ds.img_cp = da.where(mask_wo, zero_cp, ds.img_cp)

    # land cover file
    if LCCtif == None:
        if args.datamode == 'mem':
            setattr(ds, 'land_cover', AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getLCC().squeeze())
        else:
            tempfileWO = AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getLCC()
            setattr(ds, 'land_cover', readhdf(tempfileWO.name, args.chunks).squeeze())
            args.tempfiles.append(tempfileWO)
    else:
        setattr(ds, 'land_cover', readfileio(wotif, args.chunks))
        # args.tempfiles = None
    # Convert ESA LCC to GP LCC
    # ds.land_cover = ESALCC2GPLCC(ds.land_cover)

    # ocean mask file
    if args.datamode == 'mem':
        setattr(ds, 'ocean_mask', AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getOceanMask().squeeze())
    else:
        tempfileOM = AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getOceanMask()
        setattr(ds, 'ocean_mask', readhdf(tempfileOM.name, args.chunks).squeeze())
        args.tempfiles.append(tempfileOM)

    # DEM file
    if getattr(args, "use_dem", True):
        if args.datamode == 'mem':
            setattr(ds, 'dem',
                    AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getDEM().squeeze())
        else:
            tempfileDEM = AncillaryData(ref=infilepath[0], chunk_size=args.chunks).getDEM()
            setattr(ds, 'dem', readhdf(tempfileDEM.name, args.chunks).squeeze())
            args.tempfiles.append(tempfileDEM)

        if hasattr(ds.dem, "rechunk"):
            ds.dem = ds.dem.rechunk(ds.img_lp.chunks)

    del ds.inc

    # print("img_lp chunks:", getattr(ds.img_lp, "chunks", None))
    # print("img_cp chunks:", getattr(ds.img_cp, "chunks", None))
    # print("water_occurrence chunks:", getattr(ds.water_occurrence, "chunks", None))
    return args, ds
