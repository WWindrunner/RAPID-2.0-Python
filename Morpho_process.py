import numpy as np
import dask
import dask.array as da
from itertools import product
import rasterio
import os
import psutil
from scipy import ndimage
import dask.dataframe as dd
from utils import *

def process_memory():
    """
    Get current memory usage.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss      

def remove_object(boolean_list,groubyobj,labeled_flood_mask,CM):
    remove_list = groubyobj.loc[boolean_list].index.tolist()
    mask = da.isin(labeled_flood_mask, remove_list)
    morph_mask_raw = da.where(mask, 0, CM)
    return morph_mask_raw

@tim
def Morph_proc(args,ds):
    if not args.floodimage:
        # step 1: initialize water mask and seeds 
        maskOut = ds.img_lp <= 0
        rasWB = ds.High_mask
        rasWB[maskOut] = 0
        #print(np.count_nonzero(rasWB==1).compute())
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'LCC_only':
            LCC = ds.land_cover
            #LCC[maskOut] = np.nan
            rasWB[da.isin(LCC, args.nWBClasses)] = 0 
            rasSeedsLCC=da.isin(LCC, args.WBClasses)
        else:
            #rasSeedsLCC=da.where(rasWB==1,False,False)
            rasSeedsLCC = da.zeros(rasWB.shape, dtype=bool)
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'prob_only':
            rasWBProb = ds.water_occurrence
            #rasWBProb[maskOut] = np.nan
            rasSeedsProb = rasWBProb >= args.thSeed
            #wopseedmask = da.logical_or(np.isnan(rasSeedsProb), rasSeedsProb<args.thSeed)
            #rasSeedsProb[wopseedmask]=False
        else:
            #rasSeedsProb=da.where(rasWB==1,False,False)
            rasSeedsLCC = da.zeros(rasWB.shape, dtype=bool)
        rasSeeds = da.logical_or(rasSeedsProb,rasSeedsLCC)
        # step 2: find water body bojects with region growth
        flood_mask=da.where(rasWB==1,True,False)
        #s = ndimage.generate_binary_structure(2,2)
        labeled_flood_mask, num_labels = ndimage.label(flood_mask)
        print('Detected %d water bodies...' % num_labels)
        CM = da.from_array(labeled_flood_mask, chunks=args.chunks)
        # step 3: get feature statistic for candidate water bodies
        isWater=da.where(CM>0,1,0)
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'prob_only':
            coincidence_raw=da.logical_and(isWater==1, rasWBProb>0 )
        else:
            coincidence_raw=da.logical_and(isWater==1, rasSeeds)
        rasurbanLCC_value=da.where(da.isin(LCC, args.nWBClasses),1,0)
        coincidence = da.where(coincidence_raw,1,0)
        ddf = dd.concat([dd.from_dask_array(c) for c in [CM.ravel(),
            isWater.ravel(),coincidence.ravel(),rasurbanLCC_value.ravel()]], axis = 1)
        ddf.columns = ['CM', 'isWater','coincidence','LCC']
        ddf = ddf.loc[ddf.CM != 0]
        grouped = ddf.groupby('CM')
        objSum = grouped.isWater.sum()
        coinSum = grouped.coincidence.sum()
        objSumNWB = grouped.LCC.sum()
        obj_rNWB = objSumNWB / objSum
        sensitivity = coinSum / objSum
        sensitivity, obj_rNWB = dd.compute(sensitivity, obj_rNWB)
        # step 4: apply filter rules and get dry morph mask
        boolean_list = ((sensitivity < args.thSensPre) | (obj_rNWB > args.rNWB_WST_Org))
        morph_mask_raw = remove_object(boolean_list,objSum.compute(),labeled_flood_mask,CM)
        morph_mask_raw[morph_mask_raw>0]=1
        _, num_labels_after = ndimage.label(morph_mask_raw)
        print('Detected %d water bodies... after morph dry' % num_labels_after)
        mask_profile = args.refmeta
        mask_profile.update(dtype=rasterio.uint8, nodata=255)
        morph_mask = da.where(ds.img_lp>0, morph_mask_raw, 0).astype(np.uint8)
        morph_mask_write = da.where(ds.img_lp>0, morph_mask_raw, mask_profile['nodata']).astype(mask_profile['dtype'])
        setattr(ds, 'Morph_mask', morph_mask)
        args.fileOutMorph=os.path.join(args.dirOut,'Morph.tif')
        with rasterio.open(args.fileOutMorph, 'w', **mask_profile, compress='deflate') as dst:
            dst.write(morph_mask_write.compute(),indexes=1)
        del sensitivity, obj_rNWB
    else:
        # step 1: initialize water mask and seeds 
        maskOut = ds.img_lp <= 0
        rasWB_post_high = ds.High_mask
        rasWB_post_low = ds.Low_mask
        rasWB_post_high[maskOut] = 0
        rasWB_post_low[maskOut] = 0
        rasWB_pre = ds.pre_mod_mask
        rasWB_pre[maskOut]=0
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'LCC_only':
            LCC = ds.land_cover
            #LCC[maskOut] = np.nan
            rasWB_post_low[da.logical_and(da.isin(LCC, args.nWBClasses),rasWB_post_high==0)] = 0
            rasSeedsLCC=da.isin(LCC, args.WBClasses)
        else:
            rasSeedsLCC = da.zeros(rasWB_post_high.shape, dtype=bool)
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'prob_only':
            rasWBProb = ds.water_occurrence
            #rasWBProb[maskOut] = np.nan
            rasSeedsProb= rasWBProb>= args.thSeed
        else:
            rasSeedsProb = da.zeros(rasWB_post_low.shape, dtype=bool)
        rasSeeds = da.logical_or(rasSeedsProb,rasSeedsLCC)
        # step 2: find water body bojects with region growth 
        flood_mask= rasWB_post_low==1
        #s = ndimage.generate_binary_structure(2,2)
        labeled_flood_mask, num_labels = ndimage.label(flood_mask)
        print('Detected %d water bodies... morph flood part I' % num_labels)
        CM = da.from_array(labeled_flood_mask, chunks=args.chunks)
        # step 3: get feature statistic for candidate water bodies
        isWater=da.where(CM>0,1,0)
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'prob_only':
            coincidence_raw=da.logical_and(isWater==1, rasWBProb>0 ).astype(np.uint8)
        else:
            coincidence_raw=da.logical_and(isWater==1, rasSeeds).astype(np.uint8)
        rasurbanLCC_value=da.where(da.isin(LCC, args.nWBClasses),1,0)
        coincidence = da.where(coincidence_raw,1,0)
        ddf = dd.concat([dd.from_dask_array(c) for c in [CM.ravel(),
            isWater.ravel(),coincidence.ravel(),rasurbanLCC_value.ravel()]], axis = 1)
        ddf.columns = ['CM', 'isWater','coincidence','LCC']
        ddf = ddf.loc[ddf.CM != 0]
        grouped = ddf.groupby('CM')
        objSum = grouped.isWater.sum()
        coinSum = grouped.coincidence.sum()
        objSumNWB = grouped.LCC.sum()
        obj_rNWB_Org=objSumNWB / objSum
        sensitivity=coinSum / objSum
        obj_rNWB=objSumNWB / (objSum-coinSum)
        sensitivity, coinSum, obj_rNWB, obj_rNWB_Org = dd.compute(sensitivity, coinSum, obj_rNWB, obj_rNWB_Org)
        # step 4: apply filter rules and get primary flood morph mask
        boolean_list = ( (sensitivity < args.thSensIn) | (coinSum < args.thSize) | ((obj_rNWB > args.rNWB_WST) & (obj_rNWB_Org > args.rNWB_WST_Org)) )
        morph_mask_raw = remove_object(boolean_list,objSum.compute(),labeled_flood_mask,CM)
        morph_mask_raw[morph_mask_raw>0]=1
        _, num_labels_after_pri = ndimage.label(morph_mask_raw)
        print('Detected %d water bodies... after morph flood part I' % num_labels_after_pri)
        # Improve change detection
        # step 5: locate change object
        rasBkg=rasWB_post_low-CM
        WBDiffHigh=rasWB_post_high-rasWB_pre
        WBDiffHigh[WBDiffHigh<0]=0
        labeled_flood_mask_ICD, num_labels_ICD = ndimage.label(WBDiffHigh)
        print('Detected %d water bodies... morph flood part II' % num_labels_ICD)
        CM2 = da.from_array(labeled_flood_mask_ICD, chunks=args.chunks)
        # step 6: get feature statistic for ICD
        mask_CM2=da.where(CM2>0,1,0)
        WBDiffLow=mask_CM2-rasWB_pre
        WBDiffLow[WBDiffHigh<0]=0
        if args.morpho_seed_type == 'both' or args.morpho_seed_type == 'prob_only':
            coincidence=da.logical_and(mask_CM2==1, rasWBProb>0 ).astype(np.uint8)
        else:
            coincidence=da.logical_and(mask_CM2==1, rasSeeds).astype(np.uint8)
        rasurbanLCC_value=da.where(da.isin(LCC, args.nWBClasses),1,0)
        coincidence = da.where(coincidence_raw,1,0)
        ddf = dd.concat([dd.from_dask_array(c) for c in [CM.ravel(),
            mask_CM2.ravel(),coincidence.ravel(),WBDiffLow.ravel(),
            rasWB_post_high.ravel(), rasurbanLCC_value.ravel()]], axis = 1)
        ddf.columns = ['CM', 'mask_CM2','coincidence','WBDiffLow','rasWB_post_high','LCC']
        ddf = ddf.loc[ddf.CM != 0]
        grouped = ddf.groupby('CM')
        objSum = grouped.mask_CM2.sum()
        coinSum = grouped.coincidence.sum()
        objSumNWB = grouped.LCC.sum()
        objPost = grouped.WBDiffLow.sum()
        objHigh = grouped.rasWB_post_high.sum()
        obj_rInund = objPost / objSum
        obj_rProb = objHigh / objSum
        obj_rNWB = objSumNWB /(objSum-coinSum)
        #import pdb; pdb.set_trace()
        #obj_rNWB [objSum==coinSum] = 0 # just to mask out np.inf, not neccessary.
        obj_rInund, obj_rProb, objSum, obj_rNWB = dd.compute(obj_rInund, obj_rProb, objSum, obj_rNWB)
        # step 7: apply filter rules and get secondary flood morph mask
        boolean_list_P2 = ( (obj_rInund < args.thInund) | (obj_rInund < args.thInund * obj_rNWB) | (obj_rProb < args.rProb) | (objSum < args.thSize) | (obj_rNWB > args.rNWB_ICD) )
        morph_mask_raw_M2 = remove_object(boolean_list_P2,objSum,labeled_flood_mask_ICD,CM2)
        morph_mask_raw[morph_mask_raw_M2>0]=2
        _, num_labels_after_P2 = ndimage.label(morph_mask_raw)
        print('Detected %d water bodies... after morph flood part II' % num_labels_after_P2)
        mask_profile = args.refmeta
        mask_profile.update(dtype=rasterio.uint8, nodata=255)
        morph_mask = da.where(ds.img_lp>0, morph_mask_raw, 0).astype(np.uint8)
        morph_mask_write = da.where(ds.I_pol1>0, morph_mask_raw, mask_profile['nodata']).astype(mask_profile['dtype'])
        setattr(ds, 'Morph_mask', morph_mask)
        args.fileOutMorph=os.path.join(args.dirOut,'Morph.tif')
        with rasterio.open(args.fileOutMorph, 'w', **mask_profile, compress='deflate') as dst:
            dst.write(morph_mask_write.compute(),indexes=1)
        del obj_rInund, obj_rProb, objSum, obj_rNWB, sensitivity, coinSum, obj_rNWB_Org
    return args,ds
