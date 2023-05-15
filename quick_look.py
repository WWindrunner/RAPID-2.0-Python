import numpy as np
import dask
import dask.array as da
from osgeo import gdal, osr
import rasterio
from rasterio import Affine
import os
import shutil
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches

def GetGeoInfo(FileName):
    SourceDS = gdal.Open(FileName)
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = SourceDS.GetRasterBand(1).DataType
    del SourceDS
    return NDV, xsize, ysize, GeoT, Projection, DataType

def CreateGeoTiff(NewFileName, Array, driver, NDV,
                  xsize, ysize, GeoT, Projection, DataType):
    # Set nans to the original No Data Value
    if NDV is not None:
        try:
            Array[np.isnan(Array)] = NDV
        except:
            print('better no nan in this raster')
    # Set up the dataset
    DataSet = driver.Create(NewFileName, xsize, ysize, 1, DataType, options=['COMPRESS=DEFLATE'])
    # the '1' is for band 1.
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projection.ExportToWkt())
    # Write the array
    DataSet.GetRasterBand(1).WriteArray(Array)
    if NDV is not None:
        DataSet.GetRasterBand(1).SetNoDataValue(NDV)
    DataSet.FlushCache() 
    return NewFileName

def quicklook_color():
    # color for GP LCC
    palette_MergeProduct = np.array([[230, 230, 0],  # Crop land
                                 [56, 168, 0],  # Forest
                                 [76, 230, 0],  # Grassland
                                 [163, 255, 115],  # Shrubland
                                 [0, 255, 197],  # Wetland
                                 [0, 92, 230],  # Water
                                 [178, 178, 178],  # Tundra
                                 [0, 0, 0],  # Impervious surface
                                 [230, 152, 0],  # Bareland
                                 [255, 190, 232],  # Snow/Ice
                                 [0, 197, 255],  # Pre-event Water
                                 [219, 0, 0], # flood
                                 [255, 215, 0]])# River Ice
    palette_MergeProductRGB=palette_MergeProduct/255
    cmap = ListedColormap(palette_MergeProductRGB)
    norm = colors.BoundaryNorm([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], 13)
    NameofLabel=['Cropland','Forest','Grassland','Shrubland',\
                 'Wetland','Water','Tundra','Impervious surface',\
                 'Bareland', 'Snow/Ice',\
                 'Pre-event Water','Flood inundation','River Ice']
    patchList = []
    for idx, val in enumerate(palette_MergeProductRGB):
        data_Patch = mpatches.Patch(color=val, label=NameofLabel[idx])
        patchList.append(data_Patch)    
    GPLCC_PLTpara = [patchList, cmap, norm]
    return GPLCC_PLTpara

def composition(output):
    if len(output)==1 or output[0][0].MODE == 'single':
        output[0][1].pre_compensate_mask = None
    else:
        flood_src = output[0][0].ref_src
        dry_srcs = [out[0].ref_src for out in output[1:]]
        mod_masks = [out[1].Compensate_mask for out in output[1:]]
        mosaic_results = []
        for dry_ras,dry_src in zip(mod_masks,dry_srcs):
            dry_src_meta = dry_src.meta.copy()
            dry_src_meta.update(dtype=rasterio.uint8, nodata=255)
            mosaic_results.append(dask.delayed(reproject_clip_direct)(dry_ras,dry_src_meta,flood_src.meta.copy()))
        mosaic_results = dask.compute(*mosaic_results)
        mesaic_raw = da.sum(da.stack(mosaic_results, axis=0),axis=0).squeeze()
        output[0][1].pre_compensate_mask = da.where(mesaic_raw>0,1,0)
        del mosaic_results,mesaic_raw
    return output

def purge2array(output):
    flood_arr = output[0][1].Compensate_mask.compute()
    WOP_arr = output[0][1].water_occurrence.compute()
    LCC_arr = output[0][1].land_cover.compute()
    valid_mask = da.where(output[0][1].img_lp>0,True,False)
    dry_arr = output[0][1].pre_compensate_mask.compute() if output[0][1].pre_compensate_mask is not None else None
    dire_arr = output[0][1].direct_mask.compute() if output[0][1].direct_mask is not None else None
    return flood_arr,dry_arr,dire_arr,WOP_arr,LCC_arr,valid_mask

def RAPID_quicklookshow(savepath, arr_list, \
                              Mergeresult_PLTpara, \
                              geotrans, titleName):
    # compute the number of rows and columns
    tarr1 = arr_list
    shape_arr1 = tarr1.shape
    # setup the plot
    if shape_arr1[0]>shape_arr1[1]:
        adj=shape_arr1[0]/shape_arr1[1]
        fig = plt.figure(figsize=(7 , 7  * adj))
    else:
        adj = shape_arr1[1] / shape_arr1[0]
        fig = plt.figure(figsize=(7 *adj, 7))
    # loop through each subplot and plot values there
    ax = fig.add_subplot(111)
    xticks = np.arange(0, shape_arr1[1], int(shape_arr1[1]+1)/7)
    yticks = np.arange(0, shape_arr1[0], int(shape_arr1[0]+1)/7)
    xticksLnum=geotrans[0]+xticks*geotrans[1]
    yticksLnum=geotrans[3]+yticks*geotrans[5]
    xticksLnumaround=np.around(xticksLnum, decimals=2)
    yticksLnumaround = np.around(yticksLnum, decimals=2)
    xticksLnumaroundstr=xticksLnumaround.astype(str)
    yticksLnumaroundstr = yticksLnumaround.astype(str)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticksLnumaroundstr,fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticksLnumaroundstr,fontsize=8)
    ax.imshow(tarr1, interpolation='nearest', cmap=Mergeresult_PLTpara[1], norm=Mergeresult_PLTpara[2])
    lg_merge = fig.legend(handles=Mergeresult_PLTpara[0], loc='upper left',bbox_to_anchor=(0.9, 0.88),\
                          ncol=int(1))
    st = fig.suptitle(titleName,fontsize=16)
    st.set_position([.5, 0.925])
    plt.savefig(savepath[0],
                dpi=300,
                format='png',
                bbox_extra_artists=(lg_merge,st),
                bbox_inches='tight')
    plt.savefig(savepath[1],
            dpi=300,
            format='jpeg',
            bbox_extra_artists=(lg_merge,st),
            bbox_inches='tight')

def png_look(flood_arr,dry_arr,dire_arr,WOP_arr,LCC_arr,Dirout,fname,src,valid_mask):
    arr_quicklook = LCC_arr.copy()
    mask_pw = np.logical_or(WOP_arr>=25, LCC_arr==60)
    arr_quicklook[mask_pw]=60
    Mask_flood = np.logical_and( flood_arr==1,np.logical_and(LCC_arr!=50, arr_quicklook!=60))
    arr_quicklook[Mask_flood] = 120
    if dry_arr is not None:
        Mask_prefloodwater = np.logical_and(dry_arr==1, np.logical_and(LCC_arr!=50,LCC_arr!=60))
        arr_quicklook[Mask_prefloodwater] = 110
        fname = '3classes_' + fname
    else:
        fname = '2classes_' + fname
    if dire_arr is not None:
        liquidmask = np.logical_and(dire_arr!=1, mask_pw)
        liquidmask [~valid_mask] = False
        arr_quicklook[liquidmask] = 130
        fname = 'RiverIce_' + fname
    PLT_plate = quicklook_color()
    crs = src.crs
    xsize, ysize = src.shape
    transform = src.transform
    xmin, ymin, xmax, ymax = src.bounds
    if crs != rasterio.crs.CRS.from_epsg(4326):
        xmin_wgs84, ymin_wgs84, xmax_wgs84, ymax_wgs84 = transform_bounds(src.crs, {'init': 'EPSG:4326'}, xmin, ymin, xmax, ymax)
        transform_wgs84 = Affine(transform.a, transform.b, xmin_wgs84, transform.d, transform.e, ymax_wgs84)
    geotrans = [0]*6
    geotrans[0] = transform_wgs84[2]
    geotrans[3] = transform_wgs84[5]
    geotrans[1] = (xmax_wgs84-xmin_wgs84)/xsize
    geotrans[5] = (ymin_wgs84-ymax_wgs84)/ysize
    savepathraw = os.path.join(Dirout,'RAPID_'+fname)
    savepath=[savepathraw+'.png',savepathraw+'.jpg']
    RAPID_quicklookshow(savepath, arr_quicklook, PLT_plate, \
                        geotrans, fname)

def flood_per_pre_save(savepath_RGBfile,mask_profile,floodarr,woparr,LCCarr,prearr,liquidarr,valid_mask):
    output_arr = floodarr.copy()
    mask_pw = np.logical_or(woparr>=25, LCCarr==60)
    floodmask = np.logical_and(floodarr==1,~np.logical_or(mask_pw,prearr==1))
    output_arr[floodmask]=3
    if prearr is not None:
        premask = np.logical_and(prearr==1, ~mask_pw)
        output_arr[premask]=2
    output_arr[mask_pw]=1
    if liquidarr is not None:
        liquidmask = np.logical_and(liquidarr==0, mask_pw)
        liquidmask[~valid_mask]=False
        output_arr[liquidmask]=4
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    output_arr[~valid_mask] = mask_profile['nodata']
    output_arr = output_arr.astype(mask_profile['dtype'])
    output_arr[output_arr==0] = 255
    with rasterio.open(savepath_RGBfile, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(output_arr,indexes=1)
    ds = gdal.Open(savepath_RGBfile, 1)
    band = ds.GetRasterBand(1)
    colors = gdal.ColorTable()
    Valuerange = [1,2,3,4]
    palette = [[0, 92, 230],[0, 197, 255],[219, 0, 0],[255, 215, 0]]
    for idx,i in enumerate(Valuerange):
        colors.SetColorEntry(i, tuple(palette[idx]))
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetNoDataValue(255)
    band.WriteArray(output_arr)
    del band, ds

def quick_look(output):
    output = composition(output)
    flood_arr,dry_arr,dire_arr,WOP_arr,LCC_arr,valid_mask = purge2array(output)
    Dirout = os.path.split(output[0][0].dirOut)[0]
    fname = output[0][0].granules
    src = output[0][0].ref_src
    src_meta = output[0][0].refmeta
    two_classes_RGBfile = os.path.join(Dirout,'RAPID_2classes_RGB_'+fname+'.tif')
    out_filepath = two_classes_RGBfile
    flood_per_pre_save(two_classes_RGBfile,src_meta,flood_arr,WOP_arr,LCC_arr,None,None,valid_mask)
    png_look(flood_arr,None, None, WOP_arr,LCC_arr,Dirout,fname,src,valid_mask)
    print(f'RGB done, {two_classes_RGBfile}')
    if dire_arr is not None:
        Two_classes_RiverIce_RGBfile = os.path.join(Dirout,'RAPID_2lasses_RiverIce_RGB_'+fname+'.tif')
        flood_per_pre_save(Two_classes_RiverIce_RGBfile,src_meta,flood_arr,WOP_arr,LCC_arr,None,dire_arr,valid_mask)
        print(f'RGB done, {Two_classes_RiverIce_RGBfile}')
        png_look(flood_arr,None,dire_arr,WOP_arr,LCC_arr,Dirout,fname,src,valid_mask)
    if dry_arr is not None:
        Three_classes_RGBfile = os.path.join(Dirout,'RAPID_3classes_RGB_'+fname+'.tif')
        flood_per_pre_save(Three_classes_RGBfile,src_meta,flood_arr,WOP_arr,LCC_arr,dry_arr,None,valid_mask)
        print(f'RGB done, {Three_classes_RGBfile}')
        png_look(flood_arr,dry_arr,None,WOP_arr,LCC_arr,Dirout,fname,src,valid_mask)
    shutil.copyfile(output[0][0].fileOutCompensate,os.path.join(Dirout,'RAPID_raw_'+fname+'.tif'))
    return out_filepath