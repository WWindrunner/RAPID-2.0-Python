import numpy as np
import dask
import dask.array as da
import rasterio
import os
from scipy import ndimage
import dask.dataframe as dd
import tqdm
from utils import *

def find_surrounding_four(row, col, img_size):
    # Create arrays of indices for the neighboring pixels
    left = col - 1
    right = col + 1
    top = row - 1
    bottom = row + 1
    # Set the boundary conditions
    left = np.clip(left, 0, img_size[1] - 1)
    right = np.clip(right, 0, img_size[1] - 1)
    top = np.clip(top, 0, img_size[0] - 1)
    bottom = np.clip(bottom, 0, img_size[0] - 1)
    # Use NumPy indexing to get the neighboring pixels
    surrounding_row = np.array([row, row, top, bottom])
    surrounding_col = np.array([left, right, col, col])
    return surrounding_row, surrounding_col

def boundary_compensate(mask: np.array, background: np.array) -> np.array:
    # First erosion
    Remain = ndimage.binary_erosion(mask)
    # Identify boundary pixels for each mask using ndimage.binary_erosion
    boundaries = np.logical_xor(mask, Remain)
    # Assign extrme large value for ousiders
    #result = np.zeros(background.shape, dtype=np.uint8)
    result = mask
    # inital dem value for boundary pixels
    left_mask = background.astype(np.uint8)
    # While loop with erosion
    total = np.sum(left_mask)
    pbar = tqdm.tqdm(total=total, desc="boundary_compensate")
    while np.sum(boundaries)>0:
        # zero mask
        # Find the surrounding pixels for all boundary pixels
        row_indices, col_indices = np.where(boundaries == 1)
        ts_rows, ts_cols = find_surrounding_four(row_indices[:, np.newaxis], col_indices[:, np.newaxis], background.shape)
        C_bound_rows = ts_rows.transpose((1, 0, 2)).squeeze()
        C_bound_cols = ts_cols.transpose((1, 0, 2)).squeeze()
        # combine row and column indices into a single array of tuples
        indices = np.transpose(np.vstack((C_bound_rows.flatten(), C_bound_cols.flatten())))
        unique_indices = np.unique(indices, axis=0)
        # keep the new boundaries (new keys) based on continuity.
        mask = np.extract(left_mask[unique_indices[:,0], unique_indices[:,1]] == 1, range(len(unique_indices)))
        unique_indices_filtered = unique_indices[mask]
        # remove those pixels from background
        left_mask[unique_indices_filtered[:,0],unique_indices_filtered[:,1]]=0
        boundaries = np.zeros_like(left_mask)
        boundaries [unique_indices_filtered[:,0],unique_indices_filtered[:,1]] = 1
        result [unique_indices_filtered[:,0],unique_indices_filtered[:,1]] = 1
        # Update progress bar
        processed = total - np.sum(left_mask)
        pbar.update(processed - pbar.n)
    return result

@tim
def Compensate(args,ds):
    maskOut = ds.img_lp <= 0
    rasWB_morph = ds.Morph_mask
    rasWB_post_exlow = ds.Low_mask
    buffer = ndimage.binary_dilation(rasWB_morph, iterations=14, structure = ndimage.generate_binary_structure(2, 2))
    valid_morph = rasWB_morph==1
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
    morph_mask = da.where(ds.img_lp>0, CM, 0).astype(np.uint8)    
    morph_mask_write = da.where(ds.I_pol1>0, CM, mask_profile['nodata']).astype(mask_profile['dtype'])
    setattr(ds, 'Compensate_mask', morph_mask)
    args.fileOutCompensate=os.path.join(args.dirOut,'Compensate.tif')
    with rasterio.open(args.fileOutCompensate, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(morph_mask_write.compute(),indexes=1)
    return args,ds