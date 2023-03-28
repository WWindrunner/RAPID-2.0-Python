import os,glob,re,argparse,time,shutil,sys,random
from osgeo import gdal,osr,ogr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
import pandas as pd
from datetime import datetime,timedelta
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform
from utils import *
from AncillaryData import *
import tempfile
import dask
import dask.array as da
import contextily as cx
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
from affine import Affine

class imagelook():
    def __init__(self, workingfolder=None,ref=None,ini_res=0.001,refext=None,cropfile=None):
        self.workingfolder = workingfolder
        os.makedirs(self.workingfolder, exist_ok=True)
        self.ref = ref
        self.refext = refext
        self.cropfile = cropfile
        self.reftif = os.path.join(self.workingfolder, 'shp2raster.tif')
        self.delref = True
        self.ResX=ini_res #use coaser resolution to do quick loop, modified it according to area size
        self.ResY=ini_res
        self.Nodata=-9999
        self.bigext = False
        self.drcfile = None
        self.getname()

    def getcomtif(self):
        df_flood,df_dry = floodanddryproducts(self.ref)
        df_dry = df_dry.where(df_dry['Datetimeimage']<=df_dry['Datetimefolder']-timedelta(hours=15)) ##mask out flood images
        df_dry = df_dry[df_dry['RAPID'].notna()]
        df_dry = df_dry.reset_index(drop=True)
        self.name = df_flood.name[0]
        return df_flood,df_dry

    def quicklook_RAPID(self):
        df_flood,df_dry = self.getcomtif()
        del_list = []
        if self.refext == None:
            print('quick look whole RAPID extent')
            self.drcfile = df_flood.iloc[0].RAPID
            self.ref = self.drcfile
            xsize,ysize,GeoT,Projection,Hcrop_Ex = self.extentandgeoT()
            if len(df_dry)==0:
                self.dryfile=None
            else:
                self.dryrawfile = df_dry.iloc[0].RAPID
                self.dryfile = crop_withEx(self.workingfolder,self.dryrawfile,
                            xsize,ysize,Projection,ext=Hcrop_Ex)
                del_list.append(self.dryfile)
        else:
            self.cropfile = df_flood.iloc[0].RAPID
            self.ref = None
            xsize,ysize,GeoT,Projection,Hcrop_Ex = self.extentandgeoT()
            self.drcfile = crop_withEx(self.workingfolder,self.cropfile,
                        xsize,ysize,Projection,ext=Hcrop_Ex)
            del_list.append(self.drcfile)
            if len(df_dry)==0:
                self.dryfile=None
            else:                
                self.dryrawfile = df_dry.iloc[0].RAPID
                self.dryfile = crop_withEx(self.workingfolder,self.dryrawfile,
                            xsize,ysize,Projection,ext=Hcrop_Ex)
                del_list.append(self.dryfile)
        flood_arr = readgdal(self.drcfile)
        if self.dryfile == None:
            dry_arr=np.zeros(flood_arr.shape)
        else:
            dry_arr = readgdal(self.dryfile)
        #LCCcrop = LoopCrop(self.workingfolder,self.drcfile).MosaicandCrop_GPLCC()
        #WOPcrop = LoopCrop(self.workingfolder,self.drcfile).DownLandMosaic_ESAwater()
        #LCC_arr = readgdal(LCCcrop)
        #WOP_arr = readgdal(WOPcrop)
        WOP_arr = AncillaryData(ref = self.drcfile, chunk_size = 'auto').getWO().squeeze().compute()
        LCC_arr = AncillaryData(ref = self.drcfile, chunk_size = 'auto').getLCC().squeeze().compute()
        arr_quicklook = LCC_arr.copy()
        Mask_flood = np.logical_and( flood_arr==1,np.logical_and(LCC_arr!=50,LCC_arr!=60))
        arr_quicklook[Mask_flood] = 120
        Mask_prefloodwater = np.logical_and(dry_arr==1, np.logical_and(LCC_arr!=50,LCC_arr!=60))
        arr_quicklook[Mask_prefloodwater] = 110
        PLT_plate = quicklook_color()
        _, _, _, geotrans, _, _ = GetGeoInfo(self.drcfile)
        with rasterio.open(self.drcfile) as src:
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
        savepathraw = os.path.join(self.workingfolder,'RAPID_'+self.name)
        savepath=[savepathraw+'.png',savepathraw+'.jpg']
        RAPID_quicklookshow(savepath, arr_quicklook, PLT_plate, \
                              geotrans, self.name)
        savepath_Multiclasses = os.path.join(self.workingfolder,'RAPID_multiclasses_'+self.name+'.tif')
        #dtStart,_,_,_,_,SARid = dateinfo(self.name)
        #RGBfilename = 'RAPID-Sentinel-1_'+dtStart.replace("T", "_")+'.tif'
        RGBfilename = 'RAPID-RGB_'+self.name+'.tif'
        print(RGBfilename)
        savepath_RGBfile = os.path.join(self.workingfolder,RGBfilename)
        flood_per_pre_save(savepath_Multiclasses,savepath_RGBfile,self.drcfile,flood_arr,WOP_arr,dry_arr)
        for ifile in del_list:
            os.remove(ifile)
        #shutil.rmtree(os.path.join(self.workingfolder,'LCCmosaic'))
        #shutil.rmtree(os.path.join(self.workingfolder,'WOPmosaic'))

    def plotSAR_Hist(self):
        SARname = ['Sigma0_VH.img','Sigma0_VV.img']
        WOPfile = 1
        fig=plt.figure(figsize = (8, len(SARname)*3))
        plt.title(self.name)
        gs1 = gridspec.GridSpec(len(SARname),1)
        for idx,isn in enumerate(SARname):
            self.reftif = os.path.join(self.ref,isn)
            SARband = readgdal(self.reftif)
            NDV, _, _, _, _, _ = GetGeoInfo(self.reftif)
            SARband [ SARband == NDV ] = np.nan
            SARband = linear_to_decibel(SARband)
            if WOPfile == 1:
                WOPfile = LoopCrop(workingfolder=self.workingfolder,ref=self.reftif).DownLandMosaic_ESAwater()
                WOParr = readgdal(WOPfile)
                WOParr=WOParr.astype('float32')
                WOParr [ np.isnan(SARband) ] = np.nan
            waterSARband = SARband.copy()
            waterSARband[WOParr<70] = np.nan
            ax1 = plt.subplot(gs1[idx])
            ax1.hist(SARband.flatten(), bins=1000,label='all_pixels',alpha=0.5,color='blue')
            ax1.hist(waterSARband.flatten(), bins=1000,label='water_pixels',alpha=0.5,color='red')
            ax1.legend()
        fig.suptitle(self.name)
        plt.savefig(os.path.join(self.workingfolder,'HIST2_'+self.name + '.png'))#

    def get2polSAR(self):
        SARname = ['Sigma0_VH.img','Sigma0_VV.img']
        for isn in SARname:
            self.reftif = os.path.join(self.ref,isn)
            print(self.reftif)
            if os.path.exists(self.reftif):
                self.plotSAR()

    def plotSAR(self):
        #self.reftif = self.ref if self.drcfile is None else self.drcfile
        xsize,ysize,GeoT,Projection,Hcrop_Ex = self.extentandgeoT()
        SARband = readgdal(self.reftif)
        NDV, xsize, ysize, geotrans, _, DataType = GetGeoInfo(self.reftif)
        print(np.max(SARband),np.min(SARband))
        SARband [ SARband == NDV ] =np.nan
        SARband = linear_to_decibel(SARband)
        print(np.nanmax(SARband),np.nanmin(SARband))
        savepath = os.path.join(self.workingfolder,self.name+ os.path.split(self.reftif)[1] + '.png')
        geoplot(SARband,self.name,Hcrop_Ex,geotrans,savepath)
        imgfolder = os.path.split(self.reftif)[0]
        imgname = os.path.splitext(os.path.split(self.reftif)[1])[0]
        imgext = os.path.splitext(os.path.split(self.reftif)[1])[1]
        imgdB = os.path.join(self.workingfolder,self.name+os.path.split(self.reftif)[1]+'dB'+imgext)
        driver = gdal.GetDriverByName('GTiff')
        CreateGeoTiff(imgdB, SARband, driver, NDV, xsize, ysize, geotrans, Projection, DataType)
   
    def getname(self):
        if self.ref is None:
            self.name = os.path.split(self.cropfile)[1]
            self.workingfolder = os.path.join(self.workingfolder,self.name)
        else:
            nameraw = os.path.split(self.ref)[1]
            self.name = os.path.splitext(nameraw)[0]
    
    def drcResolution(self):
        #get resolution from first HAND file, assume all files have the same resolution
        self.HAND_list = globfiles(self.HANDfolder,'hand.tif')
        _, _, _, GeoT, _, _ = GetGeoInfo(self.HAND_list[0])
        self.ResX=GeoT[1]
        self.ResY=GeoT[5]
        
    def rasterize(self):
        if os.path.exists(self.reftif):#delete the old one
            os.remove(self.reftif)
        '''burn rasterize with system command'''
        gdalrasterize(self.ref,self.reftif,self.ResX,self.Nodata)
    
    def extentandgeoT(self):
        if self.ref is None:
            _, _, _, GeoT, Projection, _ = GetGeoInfo(self.cropfile)
            cropex = [float(di) for di in self.refext.split(',')]
            xsize = (cropex[2]-cropex[0])//GeoT[1]
            ysize = (cropex[1]-cropex[3])//abs(GeoT[5])
            GeoT = list(GeoT)
            GeoT[0] = cropex[0]
            GeoT[3] = cropex[1]
            GeoT = tuple(GeoT)
            Hcrop_Ex = [0] * 4
            Hcrop_Ex[0], Hcrop_Ex[3] = GeoT[0], GeoT[3]
            Hcrop_Ex[2], Hcrop_Ex[1] = GeoT[0] + GeoT[
                1] * xsize, GeoT[3] + GeoT[5] * ysize
        else:
            if os.path.splitext(self.ref)[1] == '.shp':
                '''rasterize'''
                self.rasterize()
            elif not os.path.exists(self.reftif):
                self.reftif = self.ref
                self.delref = False
            '''get the Ext and GeoT from refshp'''
            _, xsize, ysize, GeoT, Projection, _ = GetGeoInfo(self.reftif)
            Hcrop_Ex = [0] * 4
            Hcrop_Ex[0], Hcrop_Ex[3] = GeoT[0], GeoT[3]
            Hcrop_Ex[2], Hcrop_Ex[1] = GeoT[0] + GeoT[
                1] * xsize, GeoT[3] + GeoT[5] * ysize
        return xsize,ysize,GeoT,Projection,Hcrop_Ex

def floodanddryplot(imgflood,imgdry,title,workingfolder,savepath):
    xsize,ysize,GeoT,Projection,Hcrop_Ex = geoextinfo(imgdry)
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_title(title)
    #_, imgflood, _ = reproject_raster(in_path=imgflood, band_to_read=1, out_folder=workingfolder, destination_crs='epsg:3857')
    #_, imgdry, _ = reproject_raster(in_path=imgdry, band_to_read=1, out_folder=workingfolder, destination_crs='epsg:3857')
    #basemap, basemap_extent = cx.bounds2img(*src_flood.bounds)
    #src_extent = src_flood.bounds.left, src_flood.bounds.right, src_flood.bounds.bottom, src_flood.bounds.top
    w, s, e, n = Hcrop_Ex[0], Hcrop_Ex[1], Hcrop_Ex[2], Hcrop_Ex[3]
    _ = cx.bounds2raster(w, s, e, n,
                        ll=True,
                        path=os.path.join(workingfolder,'basemap.tif'),
                        source=cx.providers.OpenStreetMap.Mapnik)
    cropbasefile = os.path.join(workingfolder,'cropbasemap.img')
    crop_withEx(cropbasefile,os.path.join(workingfolder,'basemap.tif'),
                    xsize,ysize,Projection,ext=Hcrop_Ex)
    basmap3b = readgdal3bands(cropbasefile)
    #print(src_flood.read(1))
    ax.imshow(basmap3b)
    flood_arr = readgdal(imgflood).astype('float32')
    flood_arr [ flood_arr == 0 ] = np.nan
    print(flood_arr)
    dry_arr = readgdal(imgdry).astype('float32')
    dry_arr [ dry_arr == 0 ] = np.nan
    ax.imshow(flood_arr, cmap=colors.ListedColormap('red'))
    ax.imshow(dry_arr, cmap=colors.ListedColormap('blue'))
    plt.savefig(savepath,dpi=300,format='png',bbox_inches='tight')#

def geoplot(img,title,BBox,geotrans,savepath):
    fig, ax = plt.subplots(figsize = (8,7))
    ax.set_title(title)
    shape_arr1 = img.shape
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
    #ax.set_xlim(BBox[0],BBox[2])
    #ax.set_ylim(BBox[3],BBox[1])
    im = ax.imshow(img,cmap='gray')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    fig.colorbar(im, cax=cax)
    plt.savefig(savepath,dpi=300,format='png',bbox_inches='tight')#

def gdalrasterize(vector_fn,raster_fn,pixel_size,NoData_value):
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    inp_srs = source_layer.GetSpatialRef()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(inp_srs.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

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

def crop_withEx(work_path,src_file,X,Y,Projection,ext=None):
    #crop function, based on gdal.Warp
    if os.path.splitext(work_path)[1] == '.tif' or os.path.splitext(work_path)[1] == '.img':#direct file path or save folder
        drc_file = work_path
    else:
        drc_file = os.path.join(work_path,'resclip2Ex'+
                                str(random.randint(1,1000000))+'.tif')#give random name for intermediate result
    ds_drcaw = gdal.Open(src_file)
    gdal.Warp(drc_file, ds_drcaw, \
        dstSRS=Projection, width=X, height=Y, format='GTiff', \
        outputBounds=[ext[0], ext[1], 
                      ext[2], ext[3]], \
        resampleAlg=gdal.GRA_NearestNeighbour)
    del ds_drcaw
    return drc_file

def readgdal(file):
    #read raster band value as np array
    ds = gdal.Open(file)
    array = ds.GetRasterBand(1).ReadAsArray()
    NDV, _, _, _, _, _ = GetGeoInfo(file)
    del ds
    return array

def readgdal3bands(file):
    #read raster band value as np array
    ds = gdal.Open(file)
    array = None
    for i in range(3):
        if array is None:
            array = ds.GetRasterBand(i+1).ReadAsArray()
        else:
            array = np.dstack((array,ds.GetRasterBand(i).ReadAsArray()))
    del ds
    return array

def globfiles(folder,string):
    List_files = []
    for root, dirs, files in os.walk(folder, topdown=True):
       for name in files:
           if string in name:
               List_files.append(os.path.join(root, name))
    return List_files

def decibel_to_linear(band):
    return np.power(10,np.array(band)/10)

def linear_to_decibel(band):
    return 10*np.log10(band)

def floodanddryproducts(Folder_RAPIDFIM):
    List_RAPIDs = []
    List_RAPIDnames = []
    for root, dirs, files in os.walk(Folder_RAPIDFIM, topdown=True):
       for name in files:
           if 'ompensate' in name:
               List_RAPIDs.append(os.path.join(root, name))
               List_RAPIDnames.append(max(root.split('/'), key=len))
    zipped = list(zip(List_RAPIDs, List_RAPIDnames))
    df_RAPIDs = pd.DataFrame(zipped, columns=['RAPID','name'])
    df_RAPIDs['Datetimeimage']=pd.to_datetime(df_RAPIDs['name'], format='_%Y%m%d',exact=False)
    #df_RAPIDs['Datetimeimage']= df_RAPIDs['name'].map(lambda x : datetime.strptime(x.split('_')[4],'%Y%m%d'))
    df_RAPIDs['Datetimefolder']=pd.to_datetime(df_RAPIDs['RAPID'],format='/%Y-%m-%d/',exact=False)
    df_flood = df_RAPIDs.where(df_RAPIDs['Datetimeimage']==df_RAPIDs['Datetimefolder'])
    df_flood = df_flood[df_flood['RAPID'].notna()]
    df_flood = df_flood.reset_index(drop=True)
    df_dry = df_RAPIDs.where(df_RAPIDs['Datetimeimage']!=df_RAPIDs['Datetimefolder'])
    df_dry = df_dry[df_dry['RAPID'].notna()]
    df_dry = df_dry.reset_index(drop=True)
    return df_flood, df_dry

def geoextinfo(reftif):
    _, xsize, ysize, GeoT, Projection, _ = GetGeoInfo(reftif)
    Hcrop_Ex = [0] * 4
    Hcrop_Ex[0], Hcrop_Ex[3] = GeoT[0], GeoT[3]
    Hcrop_Ex[2], Hcrop_Ex[1] = GeoT[0] + GeoT[
    1] * xsize, GeoT[3] + GeoT[5] * ysize
    return xsize,ysize,GeoT,Projection,Hcrop_Ex

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
                                 [219, 0, 0]])# flood
    palette_MergeProductRGB=palette_MergeProduct/255
    cmap = ListedColormap(palette_MergeProductRGB)
    norm = colors.BoundaryNorm([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130], 12)
    NameofLabel=['Cropland','Forest','Grassland','Shrubland',\
                 'Wetland','Water','Tundra','Impervious surface',\
                 'Bareland', 'Snow/Ice',\
                 'Pre-event Water','Flood inundation']
    patchList = []
    for idx, val in enumerate(palette_MergeProductRGB):
        data_Patch = mpatches.Patch(color=val, label=NameofLabel[idx])
        patchList.append(data_Patch)
    GPLCC_PLTpara = [patchList, cmap, norm]
    return GPLCC_PLTpara

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
                dpi=600,
                format='png',
                bbox_extra_artists=(lg_merge,st),
                bbox_inches='tight')
    plt.savefig(savepath[1],
            dpi=600,
            format='jpeg',
            bbox_extra_artists=(lg_merge,st),
            bbox_inches='tight')

def flood_per_pre_save(savepath_Multiclasses,savepath_RGBfile,floodRAPID,floodarr,woparr,prearr):
    output_arr = floodarr.copy()
    floodmask = np.logical_and(floodarr==1,~np.logical_or(woparr>=75,prearr==1))
    premask = np.logical_and(prearr==1, woparr<75)
    output_arr[floodmask]=3
    output_arr[premask]=2
    output_arr[woparr>=75]=1
    NDV_f, xsize, ysize, GeoT, Projection, DataType = GetGeoInfo(floodRAPID)
    output_arr [floodarr==NDV_f]= NDV_f
    driver = gdal.GetDriverByName('GTiff')
    CreateGeoTiff(savepath_Multiclasses, output_arr, driver, NDV_f,
        xsize, ysize, GeoT, Projection, DataType)
    arr = output_arr.astype(np.uint8)
    arr[arr==0] = 255
    CreateGeoTiff(savepath_RGBfile, arr, driver, None,
                                xsize, ysize, GeoT, Projection, gdal.GDT_Byte)
    ds = gdal.Open(savepath_RGBfile, 1)
    band = ds.GetRasterBand(1)
    colors = gdal.ColorTable()
    Valuerange = [1,2,3]
    palette = [[0, 92, 230],[0, 197, 255],[219, 0, 0]]
    for idx,i in enumerate(Valuerange):
        colors.SetColorEntry(i, tuple(palette[idx]))
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetNoDataValue(255)
    band.WriteArray(arr)
    del band, ds


def fodler2RAPIDfilesdf(npyfolder):
    List_RAPIDs = []
    List_RAPIDnames = []
    for root, dirs, files in os.walk(npyfolder, topdown=True):
        for name in files:
            if 'compensated.tif' in name:
                List_RAPIDs.append(os.path.join(root, name))
                List_RAPIDnames.append(max(root.split('/'), key=len))
    zipped = list(zip(List_RAPIDs, List_RAPIDnames))
    df_RAPIDs = pd.DataFrame(zipped, columns=['RAPID','filename'])
    df_RAPIDs['Datetimeimage']=pd.to_datetime(df_RAPIDs['RAPID'],format='_%Y%m%d',exact=False)
    df_RAPIDs['Datetimefolder']=pd.to_datetime(df_RAPIDs['RAPID'],format='/%Y-%m-%d/',exact=False)
    return df_RAPIDs

def renameRAPID(df,newfront,outputfolder):
    for index, row in df.iterrows():
        newname = newfront+row.filename
        shutil.copyfile(row.RAPID, os.path.join(outputfolder,newname+'.tif'))

def getRAPIDtif(infolder,outfolder):
    df_RAPIDs = fodler2RAPIDfilesdf(infolder)
    df_flood = df_RAPIDs.where(df_RAPIDs['Datetimeimage']==df_RAPIDs['Datetimefolder'])
    df_flood = df_flood[df_flood['RAPID'].notna()]
    df_dry = df_RAPIDs.where(df_RAPIDs['Datetimeimage']!=df_RAPIDs['Datetimefolder'])
    df_dry = df_dry[df_dry['RAPID'].notna()]
    renameRAPID(df_flood,'RAPID_flood_',outfolder)
    #renameRAPID(df_dry,'RAPID_dry_',outputfolder)

def dateinfo(SARname):
    dateAndMode=SARname.split('_')
    satCode=dateAndMode[0]
    orbit=int(dateAndMode[6])
    if satCode == 'S1A':
        satType=1
        orbit= (orbit-73) % 175 +1
    else:
        satType=2
        orbit= (orbit-27) % 175 +1
    SAR_mode=dateAndMode[1]
    if 'S' in SAR_mode:
        SAR_mode='SM'
    dtStart=dateAndMode[4]
    dtEnd=dateAndMode[5]    
    return dtStart,dtEnd,SAR_mode,satType,orbit, dateAndMode[-1]

class Timer(object):
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--o', type=str, default='/sloth/xshen123/src/Sentinel/quicklook_RAPID/Omaharegion_new',
            help='quick look map output folder')
    parser.add_argument('--i', type=str, default='/sloth/xshen123/RAPID_results/Omaharegion/2019-06-15/2',
            help='RAPID group results folder')
    parser.add_argument('--e', type=str, default='None',
            help='quick look extent')
    args, unparsed = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args=parse()
    with Timer('RAPID quick look map done! in '+args.o):
        if args.e == 'None':
            args.e = None
        imagelook(workingfolder=args.o,ref=args.i,refext=args.e).quicklook_RAPID()
        getRAPIDtif(args.o,args.o)