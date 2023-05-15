import os
import dask
import dask.array as da
import numpy as np
from Initial_sample import *
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from dask_ml.wrappers import ParallelPostFit
from scipy import ndimage

def get_random_pixels(arr, labels, num_pixels=1000):
    indices = []
    for label in labels:
        # Find all indices where the array has the given label
        label_indices = np.where(arr == label)
        # If there are fewer than num_pixels pixels with this label, just use all of them
        if len(label_indices[0]) < num_pixels:
            indices.extend(list(zip(label_indices[0], label_indices[1])))
        else:
            # Otherwise, randomly select num_pixels indices from the list of all indices with this label
            random_indices = np.random.choice(len(label_indices[0]), size=num_pixels, replace=False)
            for idx in random_indices:
                indices.append((label_indices[0][idx], label_indices[1][idx]))
    return indices

def BI_method(args,ds):
    args, ds = GetStatisticsFrmSample(args, ds)
    direct_mask_raw = da.where(da.logical_and((ds.img_lp<=args.I1u), (ds.img_cp<=args.I2u)),1,0).astype(np.uint8)
    direct_mask = da.where(ds.img_lp>0, direct_mask_raw, 0)
    PW = ds.isSample.astype(np.uint8)
    WI = da.stack([ds.img_lp,ds.img_cp], axis=0)
    num_PW = da.count_nonzero(PW==1).compute()
    Valid_sigma = da.logical_and((ds.img_lp > 0) , (ds.img_cp > 0))
    LCC_raw = da.where(Valid_sigma,ds.land_cover,0)
    LCC = da.where(da.logical_or((ds.img_lp<args.I1u), (ds.img_cp<args.I2u)),0,LCC_raw)
    NW_indices = get_random_pixels(LCC.compute(), [10, 20, 30, 40, 80, 90], int(min(int(np.ceil(num_PW/25)),3e5))) #for GP LCC
    PW_indices = get_random_pixels(PW.compute(), [1], len(NW_indices))
    Total_Sample_indices = NW_indices + PW_indices
    targets = da.concatenate([da.zeros(len(NW_indices)), da.ones(len(PW_indices))]) 
    In_features = WI.compute()[:,np.array(Total_Sample_indices)[:, 0], np.array(Total_Sample_indices)[:, 1]].T
    In_features = da.from_array(In_features,chunks='auto')
    print(f'Total sample size for SWD: {len(targets)}')
    #ML method, RF, could also use something like grid search.
    clf = ParallelPostFit(LogisticRegressionCV(cv=5), scoring="accuracy")
    #clf = ParallelPostFit(SVC(kernel='rbf', C=1.0, random_state=0), scoring="accuracy")
    clf.fit(In_features, targets)
    print(f'Fitting score: {clf.score(In_features, targets)}')
    #import pdb; pdb.set_trace()
    values = da.reshape(WI,(WI.shape[0],-1)).T
    #prediction
    print(f'Water body detecting...')
    predictions = clf.predict(values)
    # Reshape the predicted target variable to match the shape of the `values_arr` mask
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        predicted_mask = da.reshape(predictions, (1,WI.shape[1],WI.shape[2])).squeeze()
    return predicted_mask,direct_mask

def origin_method(args,ds):
    WI = da.stack([ds.img_lp,ds.img_cp], axis=0)
    PW = ds.isSample.astype(np.uint8)
    num_PW = da.count_nonzero(PW==1).compute()
    non_valid_mask = ds.img_lp <= 0
    LCC_raw = ds.land_cover
    wop_raw = ds.water_occurrence
    LCC_raw = da.where(non_valid_mask,0,LCC_raw)
    wop_raw = da.where(non_valid_mask,0,wop_raw)
    PW = da.where(wop_raw>=90,True,False)#some how import, but the optimal is mysterious.
    num_PW = da.count_nonzero(PW==1).compute()
    buffer_size = max(int(np.sqrt(num_PW//1e3+1)//2),3)
    PW_buffer = ndimage.binary_dilation(PW, iterations=buffer_size, structure = ndimage.generate_binary_structure(2, 2))
    LCC = da.where(PW_buffer,0,LCC_raw)
    NW_indices = get_random_pixels(LCC.compute(), [10, 20, 30, 40, 80, 90], int(min(int(np.ceil(num_PW/25)),3e5))) #for GP LCC
    PW_indices = get_random_pixels(PW.compute(), [1], len(NW_indices))
    Total_Sample_indices = NW_indices + PW_indices
    targets = da.concatenate([da.zeros(len(NW_indices)), da.ones(len(PW_indices))]) 
    In_features = WI.compute()[:,np.array(Total_Sample_indices)[:, 0], np.array(Total_Sample_indices)[:, 1]].T
    In_features = da.from_array(In_features,chunks='auto')
    print(f'Total sample size for SWD: {len(targets)}')
    #ML method, RF, could also use something like grid search.
    clf = ParallelPostFit(LogisticRegressionCV(cv=5), scoring="accuracy")
    #clf = ParallelPostFit(SVC(kernel='rbf', C=1.0, random_state=0), scoring="accuracy")
    clf.fit(In_features, targets)
    print(f'Fitting score: {clf.score(In_features, targets)}')
    #import pdb; pdb.set_trace()
    values = da.reshape(WI,(WI.shape[0],-1)).T
    #prediction
    print(f'Water body detecting...')
    predictions = clf.predict(values)
    # Reshape the predicted target variable to match the shape of the `values_arr` mask
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        predicted_mask = da.reshape(predictions, (1,WI.shape[1],WI.shape[2])).squeeze()
    return predicted_mask

# Get the predicted probabilities of being in class 1
#probabilities = clf.predict_proba(values)[:, 1]

def swd(args,ds):
    Valid_mask = ds.img_lp > 0
    predicted_mask,direct_mask = BI_method(args, ds)
    mask_profile = args.refmeta
    mask_profile.update(dtype=rasterio.uint8, nodata=255, count=1)
    compen_mask = da.where(Valid_mask, predicted_mask, 0).astype(np.uint8) 
    predicted_mask_write = da.where(Valid_mask, predicted_mask, mask_profile['nodata'],).astype(mask_profile['dtype'])
    #write_output(outputfile, predicted_mask_write, mask_profile)
    BIswd_wmfile=os.path.join(args.dirOut,'BISWD_wm.tif')
    with rasterio.open(BIswd_wmfile, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(predicted_mask_write,indexes=1)
    direct_mask_write = da.where(Valid_mask, direct_mask, mask_profile['nodata'],).astype(mask_profile['dtype'])
    direct_wmfile=os.path.join(args.dirOut,'directthreshold_wm.tif')
    with rasterio.open(direct_wmfile, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(direct_mask_write,indexes=1)
    predicted_mask = origin_method(args, ds)
    predicted_mask_write = da.where(Valid_mask, predicted_mask, mask_profile['nodata'],).astype(mask_profile['dtype'])
    Oriswd_wmfile=os.path.join(args.dirOut,'Originswd_wm.tif')
    with rasterio.open(Oriswd_wmfile, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(predicted_mask_write,indexes=1)
    return compen_mask, BIswd_wmfile, direct_mask
