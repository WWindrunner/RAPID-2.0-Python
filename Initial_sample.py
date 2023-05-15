import os
import dask
import dask.array as da
import numpy as np
from utils import *
from scipy.stats.distributions import chi2
from EM_threshold import *
from scipy.ndimage import median_filter

def to_db(arr):
    result = 10 * da.log10(arr)
    return da.where(result==-np.inf, np.nan, result)

def tail_adjustment(Iu,Id,I,maxP=0.99,minP=0.001):
    '''
    Adjust the tailing threshold to make sure the tailing area is within the range of maxP and minP
    '''
    ratio=1
    scaling=1
    Iu0=Iu
    while ratio>1-maxP:
        Iu=scaling*Iu0
        scaling = scaling+0.5
        mainArea = da.count_nonzero((I<Iu) & (I>Id))
        tailingArea = da.count_nonzero((I>=Iu) & (I<5*Iu))
        ratio=tailingArea/(tailingArea+mainArea)
    ratio=1
    scaling=1
    Id0=Id
    while ratio>1-minP:
        Id1=Id0/scaling
        scaling = scaling+0.5
        mainArea = da.count_nonzero((I<Iu) & (I>Id))
        tailingArea = da.count_nonzero((I>Id1/5 ) & (I<Id))
        ratio=tailingArea/(tailingArea+mainArea)
        Id=Id1
    return Iu,Id

def tile_EM(tiles,hand_candidates):
    '''
    Thresholding from tiles using Expectation Maximization
    '''
    selected_tiles = select_backscatter_tiles(tiles, hand_candidates)
    tiles = np.log10(tiles) + 30. 
    scaling = 256 / (np.mean(tiles) + 3 * np.std(tiles))
    gaussian_threshold = determine_em_threshold(tiles[selected_tiles, :, :], scaling)
    threshold_db = 10. * (gaussian_threshold - 30.)
    return threshold_db

def tile_EM_withselect(tiles,selected_tiles):
    '''
    Thresholding from tiles that meets the bimodal test
    using Expectation Maximization
    '''
    tiles = np.log10(tiles) + 30. 
    scaling = 256 / (np.mean(tiles) + 3 * np.std(tiles))
    gaussian_threshold = determine_em_threshold(tiles[selected_tiles, :, :], scaling)
    threshold_db = 10. * (gaussian_threshold - 30.)
    return threshold_db

def bound_adj(I_PW, threshold_db, ENL, maxP, minP):
    I_min = np.min(I_PW)
    I_max = np.power(10, threshold_db/10.)
    if I_min >= I_max:
        raise ValueError('The minimum value of the available water pixels is larger than the maximum value of the available water pixels.')
    h, bins = np.histogram(I_PW, bins=500, range=[I_min,I_max])
    Ip = bins[np.argmax(h)]
    xu = chi2.ppf(maxP, df=2*ENL)
    xd = chi2.ppf(minP, df=2*ENL)
    x = np.linspace(xd, xu, 500)
    pd=chi2.pdf(x,2*ENL)
    xp = x[pd.argmax()]
    Iu=(xu/xp*Ip)
    Id=(xd/xp*Ip)
    I_PW_chi2 = I_PW[(I_PW<=I_max) & (I_PW>=I_min)]
    Iu,Id = tail_adjustment(Iu,Id,I_PW_chi2)
    if Iu >= I_max:
        Iu = I_max
    #if Id <= I_min:
    #    Id = I_min
    Id = I_min
    avai_I = I_PW_chi2[(I_PW_chi2<Iu) & (I_PW_chi2>Id)]
    ENLa = np.mean(avai_I)**2/np.var(avai_I)
    return Iu,Id,ENLa
    
@tim
def findRange_EM(I_raw,wo_raw,thProbSeeds,ENL,PW_ratio = [0.2,0.6],nodata=0,tile_shape = (80,80), max_db_threshold=0.0,
                 maxP = 0.99, minP = 0.0001, max_arr = 0.1, min_arr = 0.001, L_water_tiles = 15):
    '''
    New version of findRange, which achive the Adaptive Initialization of Liquid water.
    '''
    I = I_raw.compute()
    wo = wo_raw.compute()
    non_mask = I<=0
    mean_value = np.mean(I[~non_mask])
    I[non_mask] = mean_value
    #I = median_filter(I, size=(5,5))
    I[non_mask]=0
    hand_candidates = []
    array = np.ma.masked_values(I, nodata)
    array_wo = np.where(I>min_arr, wo, 0)
    thProbSeeds = thProbSeeds + 2
    threshold_db = None
    selected_tiles = []
    while threshold_db == None and thProbSeeds >= 2:
        thProbSeeds = thProbSeeds - 2
        if thProbSeeds <= 0:
            thProbSeeds = 1
        hand_tiles = tile_array_overlap(array_wo, tile_shape=tile_shape, pad_value=np.nan)
        hand_candidates = select_water_tiles(hand_tiles, thProbSeeds, PW_ratio)
        if len(hand_candidates) >= L_water_tiles:
            array_tseed = np.ma.masked_where(wo<thProbSeeds, array)
            tiles = tile_array_overlap(array_tseed, tile_shape=tile_shape, pad_value=np.nan)
            tiles = np.ma.masked_less_equal(tiles, min_arr)
            selected_tiles = select_tiles(tiles, hand_candidates,lower_median=True)
            if len(selected_tiles) >= 5:
                threshold_db = tile_EM_withselect(tiles,selected_tiles)
    print(f'Found {len(selected_tiles)} water tiles at Persistent Water Seeds = {thProbSeeds}')
    I_PW = I[(wo>=thProbSeeds) & (I>min_arr)]
    array = np.ma.masked_where(wo<thProbSeeds, array)
    tiles = tile_array(array, tile_shape=tile_shape, pad_value=np.nan)
    tiles = np.ma.masked_less_equal(tiles, min_arr)
    #tiles = np.ma.masked_greater_equal(tiles, max_arr)
    back_up_threshold = tile_EM(tiles,np.arange(tiles.shape[0]))
    back_up_Iu, back_up_Id, back_up_ENLa = bound_adj(I_PW, back_up_threshold, ENL, maxP, minP)
    if threshold_db is not None:
        print(f'Main upbound {threshold_db} db, back up upbound {back_up_threshold} db')
        if threshold_db > max_db_threshold:
            print(f'Threshold too high! Using maximum threshold {max_db_threshold} db')
            threshold_db = max_db_threshold
        else:
            print(f'Using threshold {threshold_db} db')
        Iu,Id,ENLa = bound_adj(I_PW, threshold_db, ENL, maxP, minP)
    else:
        print(f'Can not initiliaze the main bounds, Using back up threshold {back_up_threshold} db')
        Iu,Id,ENLa = back_up_Iu, back_up_Id, back_up_ENLa
    print(f'Iu: {Iu}, Id: {Id}, ENLa: {ENLa}, back up Iu: {back_up_Iu}, back up Id: {back_up_Id}, back up ENLa: {back_up_ENLa}')
    return Iu,Id,ENLa,thProbSeeds,back_up_Iu, back_up_Id, back_up_ENLa

@tim
def findRange(I,ENL = 4.9, minOut = 1e-9, maxPrc = 90, maxP = 0.99, minP = 0.0001):
    '''
    Original version of initialization
    '''
    # find the lower and upper limits from the sample
    # output
    # Iu: the upper 0.5 percentile of the class
    # Id: the lower 0.5 percentile of the class
    # ENL_a: the actual ENL
    # input
    # I n-look intensity
    # ENL: equivalent number of looks
    # minOut: a lower limit to rule out marginal pixels of zero value. Default value: 1e-9
    # maxPrc: a safe upper percentile to rule out other classes. Default 90.
    I_max = da.percentile(I[I>minOut],maxPrc)
    I_min=(I_max+da.min(I))/10
    h, bins = da.histogram(I, bins=500, range=[I_min[0],I_max[0]])
    Ip = bins[da.argmax(h)]
    xu = chi2.ppf(maxP, df=2*ENL)
    xd = chi2.ppf(minP, df=2*ENL)
    x = da.linspace(xd, xu, 500)
    pd=chi2.pdf(x,2*ENL)
    xp = x[pd.argmax()]
    Iu=xu/xp*Ip
    Id=xd/xp*Ip
    # final adjustment
    ratio=1
    scaling=1
    Iu0=Iu
    while ratio>1-maxP:
        Iu=scaling*Iu0
        scaling = scaling+0.5
        mainArea = da.count_nonzero((I<Iu) & (I>Id))
        tailingArea = da.count_nonzero((I>=Iu) & (I<5*Iu))
        ratio=tailingArea/(tailingArea+mainArea)
    ratio=1
    scaling=1
    Id0=Id
    while ratio>1-minP:
        Id1=Id0/scaling
        scaling = scaling+0.5
        mainArea = da.count_nonzero((I<Iu) & (I>Id))
        tailingArea = da.count_nonzero((I>Id1/5 ) & (I<Id))
        ratio=tailingArea/(tailingArea+mainArea)
        Id=Id1
    avai_I = I[(I<Iu) & (I>Id)]
    E = da.nanmean(avai_I)
    Var = (da.nanstd(avai_I)**2)
    ENLa=(E**2/Var)
    return Iu.compute(),Id.compute(),ENLa.compute()

@tim
def GetStatisticsFrmSample(args, ds):
    '''Get the statistics from the persistent water pixel sample.'''
    ENL = args.ENL 
    thProbSeeds=args.thSeed
    pol1_max,pol1_min,n1,thPseedn1,back_pol1_max, back_up_pol1_min, back_up_n1 = findRange_EM(ds.img_lp, ds.water_occurrence, thProbSeeds, \
                                                  ENL,max_db_threshold=-2, \
                                                    max_arr = 0.1, min_arr = 1e-6)
    pol2_max,pol2_min,n2,thPseedn2, back_pol2_max, back_up_pol2_min, back_up_n2 = findRange_EM(ds.img_cp, ds.water_occurrence, \
                                                  thProbSeeds, ENL,max_db_threshold=-10, \
                                                    max_arr = 0.01, min_arr = 1e-6)
    thProbSeeds = (thPseedn1+thPseedn2)/2
    nNew=((n1+n2)/2)
    print(f'New:{nNew}, Ori:{ENL}, for {args.filepath[0]}')
    ENL = nNew
    if abs(nNew-ENL)/ENL>0.2:
        #Wether to use the new ENL or not.
        print(f'Too much deviation between estimated and theoretical ENLs.')
        #ENL = nNew
    Valid_sigma = da.logical_and((ds.img_lp > 0) , (ds.img_cp > 0))
    Valid_WOmask = da.logical_and(ds.water_occurrence >= thProbSeeds, Valid_sigma)
    args.mean_pol1  = da.nanmean(ds.img_lp[Valid_WOmask])
    args.mean_pol2 = da.nanmean(ds.img_cp[Valid_WOmask])
    basinMask = da.logical_and(da.logical_and(ds.img_lp>pol1_min, ds.img_cp>pol2_min), da.logical_and(ds.img_lp<pol1_max, ds.img_cp<pol2_max))
    isSample = da.logical_and(basinMask,Valid_WOmask)
    ini_pol1_water = ds.img_lp[isSample]
    ini_pol2_water = ds.img_cp[isSample]
    C0 = da.array([da.mean(ini_pol1_water), da.mean(ini_pol2_water)]).compute()
    if C0[0] < args.mean_pol1/20 or C0[1] < args.mean_pol2/20:
        print(f'Warning: The initial boundary is too small compared to the image mean, use boundary from SBA instead.')
        pol1_min = back_up_pol1_min
        pol2_min = back_up_pol2_min
        pol1_max = back_pol1_max
        pol2_max = back_pol2_max
        basinMask = da.logical_and(da.logical_and(ds.img_lp>pol1_min, ds.img_cp>pol2_min), da.logical_and(ds.img_lp<pol1_max, ds.img_cp<pol2_max))
        isSample = da.logical_and(basinMask,Valid_WOmask)
        ini_pol1_water = ds.img_lp[isSample]
        ini_pol2_water = ds.img_cp[isSample]
        C0 = da.array([da.mean(ini_pol1_water), da.mean(ini_pol2_water)]).compute()
    rho_c2 = ((da.mean(ini_pol1_water * ini_pol2_water) - C0[0] * C0[1]) / (da.nanstd(ini_pol1_water) * da.nanstd(ini_pol2_water))).compute()
    #rho_c2 = 0.5
    if rho_c2 < 0.01:
        rho_c2 = 0.01
    args.C0Init,args.rho_c2Init,args.pol1_max,args.pol1_min,args.pol2_max,args.pol2_min,args.ENL = C0,rho_c2,pol1_max,pol1_min,pol2_max,pol2_min,ENL
    args.I1u, args.I1d = pol1_max,pol1_min
    args.I2u, args.I2d = pol2_max,pol2_min
    ds.isSample = isSample
    args.thSeed = thProbSeeds
    return args, ds