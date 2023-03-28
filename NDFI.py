import os
import dask.array as da
import numpy as np
from utils import *
from AncillaryData import *
from scipy.stats.distributions import chi2
from scipy.special import gamma, iv
import rasterio
from scipy import ndimage
import dask.dataframe as dd
from Compensation import *

def remove_object(boolean_list,groubyobj,labeled_flood_mask,CM):
    remove_list = groubyobj.loc[boolean_list].index.tolist()
    mask = da.isin(labeled_flood_mask, remove_list)
    morph_mask_raw = da.where(mask, 0, CM)
    return morph_mask_raw

@tim
def NDFI(input):
    if len(input)==1:
        raise ValueError(f"No dry images, NDFI quit")
    else:
        flood_src = input[0][0].ref_src
        dry_srcs = [out[0].ref_src for out in input[1:]]
        dry_lps = [out[1].img_lp for out in input[1:]]
        mosaic_results = []
        for dry_ras,dry_src in zip(dry_lps,dry_srcs):
                mosaic_results.append(dask.delayed(reproject_clip_direct)(dry_ras,dry_src.meta.copy(),flood_src.meta.copy()))
        mosaic_results = dask.compute(*mosaic_results)
        dryI = da.sum(da.stack(mosaic_results, axis=0),axis=0).squeeze()
    floodI_raw = input[0][1].img_lp
    floodI = da.where(floodI_raw>0, floodI_raw, np.nan)
    meanDry = da.nanmean(dryI,axis=0).squeeze()
    minDry = da.nanmin(dryI,axis=0).squeeze()
    #maxDry = da.nanmax(dryI,axis=0).squeeze()
    mask = da.isnan(meanDry) | da.isnan(floodI)
    dryI = None
    NDFI = (meanDry - da.minimum(minDry, floodI)) / (meanDry + da.minimum(minDry, floodI))
    NDFI = da.where(mask, np.nan, NDFI)
    #NDFVI = (da.maximum(maxDry, floodI) - meanDry) / (da.maximum(maxDry, floodI) + meanDry)
    #NDFVI = da.where(mask, np.nan, NDFVI) 
    args = input[0][0]
    ds = input[0][1]
    Mod_mask_raw = da.where(NDFI>0.3, 1, 0)
    setattr(ds, 'Mod_mask', da.where(mask, np.nan, Mod_mask_raw))
    High_mask_raw = da.where(NDFI>0.7, 1, 0)
    setattr(ds, 'High_mask', da.where(mask, np.nan, da.where(mask, np.nan, High_mask_raw)))
    #ds.NDFI = NDFI
    #ds.NDFVI = NDFVI
    #nbI = da.sum(~mask)
    #sNDFI = da.sum(NDFI[~mask])
    #sNDFVI = da.sum(NDFVI[~mask])
    #sqNDFI = da.sum(NDFI[~mask]**2)
    #sqNDFVI = da.sum(NDFVI[~mask]**2)
    return args,ds

@tim
def NDFI_morph(args,ds):
    maskOut = da.where(np.isnan(ds.High_mask),True,False)
    rasWB = ds.High_mask
    rasWB[maskOut] = 0
    labeled_flood_mask, num_labels = ndimage.label(rasWB)
    print('Detected %d water bodies...' % num_labels)
    CM = da.from_array(labeled_flood_mask, chunks=args.chunks)
    isWater=da.where(CM>0,1,0)
    ddf = dd.concat([dd.from_dask_array(c) for c in [CM.flatten(),
    isWater.flatten()]], axis = 1)
    ddf.columns = ['CM', 'isWater']
    objSum = ddf.groupby('CM').isWater.sum().compute()
    boolean_list = objSum < args.thSize
    morph_mask_raw = remove_object(boolean_list,objSum,labeled_flood_mask,CM)
    morph_mask_raw[morph_mask_raw>0]=1
    _, num_labels_after = ndimage.label(morph_mask_raw)
    print('Detected %d water bodies... after NDFI morph ' % num_labels_after)
    mask_profile = args.refmeta
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    morph_mask = da.where(ds.img_lp>0, morph_mask_raw, np.nan)
    morph_mask_write = da.where(ds.img_lp>0, morph_mask_raw, mask_profile['nodata']).astype(mask_profile['dtype'])
    setattr(ds, 'Morph_mask', morph_mask)
    return args,ds

@tim
def NDFI_compensate(args,ds):
    maskOut = da.where(np.isnan(ds.High_mask),True,False)
    rasWB_morph = ds.Morph_mask
    #In NDFI the post_exlow is 
    rasWB_post_exlow = ds.Mod_mask
    buffer = ndimage.binary_dilation(rasWB_morph,iterations=14, structure = ndimage.generate_binary_structure(2, 2))
    valid_morph = da.where(rasWB_morph==1,True,False)
    buffer[valid_morph]=0
    background = da.where(buffer==1,rasWB_post_exlow,0)
    #valid_buffer = da.where(buffer==1,True,False)
    rasWB_morph_buffer = boundary_compensate(rasWB_morph.compute(), background.compute())
    _, num_labels = ndimage.label(rasWB_morph_buffer)
    print('Detected %d water bodies... after compensation %s' % (num_labels,args.filepath[0]))
    CM = da.where(maskOut, 0, rasWB_morph_buffer)
    CM[CM>0]=1
    mask_profile = args.refmeta
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    morph_mask = da.where(ds.img_lp>0, CM, np.nan)
    morph_mask_write = da.where(ds.img_lp>0, CM, mask_profile['nodata']).astype(mask_profile['dtype'])
    setattr(ds, 'Compensate_mask', morph_mask)
    args.fileOutMorph=os.path.join(args.dirOut,'Compensate.tif')
    with rasterio.open(args.fileOutMorph, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(morph_mask_write.compute(),indexes=1)
    return args,ds
    
