import os
import dask
import dask.array as da
import numpy as np
from utils import *
from scipy.stats.distributions import chi2
from scipy.special import gamma, iv
import rasterio
import csv

def writeProg(logFile, thProb, NSCE, C, Rho_c, ThPD, it, preENL, granule, ENL = None):
    '''Write the progress of the algorithm to a log file.
    freqency: after water sample initialzation or after each optimization of thProb.
    '''
    if not os.path.isfile(logFile):
        writeHeader = True
    else:
        writeHeader = False
    with open(logFile, 'a+', newline='') as f:
        writer = csv.writer(f)
        if isinstance(NSCE, da.Array):
            NSCE = NSCE.compute()
        if isinstance(C, da.Array):
            C = C.compute()
        if isinstance(Rho_c, da.Array):
            Rho_c = Rho_c.compute()
        if isinstance(ThPD, da.Array):
            ThPD = ThPD.compute()
        if isinstance(preENL, da.Array):
            preENL = preENL.compute()
        if ENL is not None and isinstance(ENL, da.Array):
            ENL = ENL.compute()
        if len(C) == 2:
            header = ['Prob', 'NSCE', 'C11', 'C22', 'Rho_c', 'ThPD', 'it', 'preENL', 'ENL', 'granule']
            row = [thProb, NSCE, C[0], C[1], Rho_c, ThPD, it, preENL, ENL, granule]
        else:
            header = ['Prob', 'NSCE', 'C', 'Rho_c', 'ThPD', 'it', 'preENL', 'ENL', 'granule']
            row = [thProb, NSCE, C, Rho_c, ThPD, it, preENL, ENL, granule]
        if writeHeader:
            writer.writerow(header)       
        writer.writerow(row)

@tim
def findRange(I,ENL = 4.9, minOut = 1e-9, maxPrc = 90, maxP = 0.99, minP = 0.0001):
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
    #Valid_WOmask = da.where((ds.water_occurrence >= thProbSeeds) & (ds.water_occurrence <= 100), True, False)
    Valid_WOmask = da.logical_and((ds.water_occurrence >= thProbSeeds) , (ds.water_occurrence <= 100))
    Valid_sigma = da.logical_and((ds.img_lp > 0) , (ds.img_cp > 0))
    Valid_WOmask = da.logical_and(Valid_WOmask, Valid_sigma)
    pol1_water = ds.img_lp[Valid_WOmask]
    pol2_water = ds.img_cp[Valid_WOmask]
    pol1_max,pol1_min,n1 = findRange(pol1_water, ENL)
    pol2_max,pol2_min,n2 = findRange(pol2_water, ENL)
    nNew=((n1+n2)/2)
    print(f'New:{nNew}, Ori:{ENL}, for {args.filepath[0]}')
    if abs(nNew-ENL)/ENL>0.2:
        #Wether to use the new ENL or not.
        print(f'Too much deviation between estimated and theoretical ENLs.')
        ENL = nNew
    basinMask = da.logical_and(da.logical_and(ds.img_lp>pol1_min, ds.img_cp>pol2_min), da.logical_and(ds.img_lp<pol1_max, ds.img_cp<pol2_max))
    isSample = da.logical_and(basinMask,Valid_WOmask)
    #if not os.path.exists(fileSampleMap):
        #isSample.astype(np.uint8).rio.to_raster(fileSampleMap)
    ds.isSample = isSample
    ini_pol1_water = ds.img_lp[isSample]
    ini_pol2_water = ds.img_cp[isSample]
    C0 = da.array([da.mean(ini_pol1_water), da.mean(ini_pol2_water)]).compute()
    rho_c2 = ((da.mean(ini_pol1_water * ini_pol2_water) - C0[0] * C0[1]) / (da.nanstd(ini_pol1_water) * da.nanstd(ini_pol2_water))).compute()
    return C0,rho_c2,pol1_max,pol1_min,pol2_max,pol2_min,ENL,ds

def importDtSrc(args,ds):
    if not args.imported:
        ds.I_pol1 = da.where(ds.img_lp>0,ds.img_lp,np.nan)
        ds.I_pol2 = da.where(ds.img_cp>0,ds.img_cp,np.nan)
        ds.isWater = da.where(ds.isSample, True, False)
        #ds['I_pol1'] = ds.img_lp.where(ds.img_lp>0)
        #ds['I_pol2'] = ds.img_cp.where(ds.img_cp>0)
        #ds['isWater'] = ds.isSample
        args.imported = True
    return args,ds

def getgrids(args):
    # create meshgrids based on max, min, and sampling interval.
    if len(args.minI)==1:
        dual_pol=False
        minHH=args.minI
        maxHH=args.maxI
        deltaHH=args.delta
        hvgrid = None
    else:
        dual_pol=True
        minHH=args.minI[0]
        maxHH=args.maxI[0]
        minHV=args.minI[1]
        maxHV=args.maxI[1]
        deltaHH=args.delta[0]
        deltaHV=args.delta[1]
        hvgrid = da.linspace(minHV, maxHV, args.nInter)
    hhgrid = da.linspace(minHH, maxHH, args.nInter)
    return dual_pol, hhgrid, hvgrid, deltaHH, deltaHV

def dualpol_dist(args,HH,HV):
    ''' Compute the joint probability of HH and HV
    Using the dual-pol distribution function
    ENL: Equivalent number of looks, control the shape of the distribution.
    key modules: scipy.special.iv, scipy.special.gamma
    '''
    C, rho_c2, ENL = args.C,args.rho_c2,args.ENL
    var = da.sqrt((HH*HV)/(C[0]*C[1]))
    nom = ENL**(ENL+1) * da.power(var,(ENL-1)) / (C[0]*C[1]) * da.exp(-ENL*(HH/C[0]+HV/C[1]) / (1-rho_c2))
    denom = gamma(ENL) * (1-rho_c2) * rho_c2**((ENL-1)/2)
    factor = nom/denom
    varJ = 2*ENL*var*da.sqrt(rho_c2)/(1-rho_c2)
    I = iv(ENL-1,varJ)
    jointProb = factor*I
    return jointProb

@dask.delayed
def loop_PD(PD, dA, prob, thPD, minPD, maxPD, thProb):
    while abs(prob - thProb) > 1e-3 and abs(thPD - minPD) / minPD > 1e-2:
        if prob > thProb:
            thPD1 = (thPD + maxPD) / 2
            minPD = thPD
        else:
            thPD1 = (thPD + minPD) / 2
            maxPD = thPD
        thPD = thPD1
        PDIn = da.where(PD >= thPD, PD, 0)
        prob = (dA * PDIn).sum()
    return thPD

def FindMinPD(args, PD, deltaHH, deltaHV=None):
    ''' Find the minimum probality density threshold (thPD) that can be used to get the desired probability of detection.
    Using the bisection method.
    With the probability of detection (PD) and the area of each grid (dA), the total area of the grid with PD>=thPD is calculated.
    '''
    maxPD = da.max(PD)
    minPD = da.min(PD)
    thPD = (minPD + maxPD) / 2
    PDIn = da.where(PD >= thPD, PD, 0)
    if deltaHV is not None: 
        dA = deltaHH * deltaHV
    else:
        dA = deltaHH
    prob = (dA * PDIn).sum()
    thPD = loop_PD(PD, dA, prob, thPD, minPD, maxPD, args.thProb).compute()
    return thPD

def getHist(args,ds):
    ''' Get the histogram of polarizations'''
    dual_pol, hhgrid, hvgrid, deltaHH, deltaHV = getgrids(args)
    if dual_pol:
        inter_basinMask = da.logical_and(da.logical_and(ds.I_pol1>args.pol1_min, ds.I_pol2>args.pol2_min), da.logical_and(ds.I_pol1<args.pol1_max, ds.I_pol2<args.pol2_max))
        basinMask = da.logical_and(inter_basinMask,~ds.isWater)
        jointProbP=dualpol_dist(args,da.where(basinMask,ds.I_pol1,np.nan),da.where(basinMask,ds.I_pol2,np.nan))
        maskP = jointProbP>args.thPD
        maskall = da.logical_or(maskP,ds.isWater)
        t_pol1_water = ds.I_pol1[maskall]
        t_pol2_water = ds.I_pol2[maskall]
        #t_histImage,_,_ = da.histogram2d(t_pol1_water.flatten(), t_pol2_water.flatten(), bins=(args.nInter,args.nInter), range=((hhgrid[0]-0.5*deltaHH,hhgrid[-1]+0.5*deltaHH),(hvgrid[0]-0.5*deltaHV,hvgrid[-1]+0.5*deltaHV)))
        t_histImage,_,_ = np.histogram2d(t_pol1_water, t_pol2_water, bins=(args.nInter,args.nInter), range=(((hhgrid[0]-0.5*deltaHH).compute(),(hhgrid[-1]+0.5*deltaHH).compute()),((hvgrid[0]-0.5*deltaHV).compute(),(hvgrid[-1]+0.5*deltaHV).compute())))
        nSamples = da.count_nonzero(maskall==True)
    else:
        thI = chi2.ppf(args.thProb, df=2*args.ENL) * args.C / (2*args.ENL)
    return t_histImage,nSamples

def getNSCE(args, ds, thPD=None):
    ''' Get the normalized sample covariance error (NSCE) of two distribution.'''
    dual_pol, hhgrid, hvgrid, deltaHH, deltaHV = getgrids(args)
    if dual_pol:
        HV,HH = da.meshgrid(hvgrid, hhgrid)
        jointProb = dualpol_dist(args,HH,HV)
        if thPD==None:
            args.thPD = FindMinPD(args,jointProb,deltaHH,deltaHV)
            if args.ini_thPD is None:
                args.ini_thPD = args.thPD
        else:
            args.thPD = thPD
        histImage,nSamples = getHist(args,ds)
    if dual_pol:
        probReal=histImage/(deltaHH*deltaHV)/nSamples
    probmask = da.where(jointProb>args.thPD,True,False)
    probReal_np = probReal[probmask]
    jointProb_np = jointProb[probmask]
    args.NSCE=(1-da.sum((probReal_np-jointProb_np)**2)/da.sum((jointProb_np-da.mean(jointProb_np))**2)).compute()
    if args.NSCE != None:
        rWishart = jointProb/da.max(jointProb)
        rReal = probReal/da.max(probReal)
        isErr= rReal - da.where(rWishart > args.th_grid_ratio, rWishart,np.nan)
        if da.nansum(isErr)/jointProb[jointProb>300].sum()>0.1:
            args.PDFContaminated=True
        else:
            args.PDFContaminated=False
    return args

def wishartStats(args,I1,I2,nargout=3):
    ''' Calculate the Wishart statistics of two polarizations.'''
    ''' C: mean of two polarizations (water pixels)'''
    ''' rho_c2: correlation coefficient of two polarizations'''
    ''' The range of I1 and I2 are derived optimization steps.'''
    args.C = da.array([da.mean(I1), da.mean(I2)]).compute()
    args.rho_c2= ((da.mean(I1*I2)-(args.C[0]*args.C[1]))/(da.nanstd(I1)*da.nanstd(I2))).compute()
    if args.rho_c2<0:
        args.rho_c2=1e-3
    if nargout==3:
        minI = da.array([da.nanmin(I1),da.nanmin(I2)])
        maxI = da.array([da.nanmax(I1),da.nanmax(I2)])
        wishartinter = 500
        deltaHH, deltaHV = (maxI[0]-minI[0])/wishartinter, (maxI[1]-minI[1])/wishartinter
        hhgrid, hvgrid = da.linspace(minI[0], maxI[0], wishartinter), da.linspace(minI[1], maxI[1], wishartinter)
        HV,HH = da.meshgrid(hvgrid, hhgrid)
        jointProb = dualpol_dist(args,HH,HV)
        t_histImage,_,_ = da.histogram2d(I1, I2, bins=wishartinter, range=((minI[0], maxI[0]),(minI[1], maxI[1])))
        probReal=t_histImage/(deltaHH*deltaHV)/da.count_nonzero(~da.isnan(I2))
        probReal_np = probReal
        jointProb_np = jointProb
        args.NSCE=(1-da.nansum((probReal_np-jointProb_np)**2)/da.nansum((jointProb_np-da.nanmean(jointProb_np))**2))
    return args

def polsar_func3(args,ds):
    '''Calculate the Wishart statistics for dual-polarization SAR image/swath.
    Two distributions are compared: the theoretical and the empirical distribution of the input image.
    input: args, ds
    function prameters: args.C, args.rho_C2, args.maxI, args.minI. Which are the mean, the correlation coefficient, the maximum and the minimum backscattering intensity of water bodies from initial guess.
    control prameter: args.thProb. The threshold probability for the Wishart distribution.
    key functions: dualpol_dist FindMinPD.
    '''
    dual_pol, hhgrid, hvgrid, deltaHH, deltaHV = getgrids(args)
    if dual_pol:
        HV,HH = da.meshgrid(hvgrid, hhgrid)
        jointProb = dualpol_dist(args,HH,HV)
        args.thPD = FindMinPD(args,jointProb,deltaHH,deltaHV)
    #mask = da.where(jointProb>=args.thPD, True, False)
    mask = jointProb>=args.thPD
    I1, I2, jointProb = HH[mask], HV[mask], jointProb[mask]
    args.tC = da.array([(I1*jointProb).sum()*deltaHH*deltaHV,(I2*jointProb).sum()*deltaHH*deltaHV]).compute()
    I1, I2, jointProb=None,None,None
    if args.bExport:
        args,ds = importDtSrc(args,ds)
        args.bExport = False
    jointProbP=dualpol_dist(args,ds.jointProb_I_pol1,ds.jointProb_I_pol2)
    #have to include the persistent water pixels
    maskP_water = da.logical_or(jointProbP>args.thPD, ds.isWater)
    t_pol1_water = ds.I_pol1[maskP_water]
    #t_pol2_water = da.where(da.logical_or(maskP,ds.isWater), ds.I_pol2, np.nan)
    t_pol2_water = ds.I_pol2[maskP_water]
    args=wishartStats(args,t_pol1_water,t_pol2_water,2)
    args.f=args.tC-args.C
    args.f0=args.tC
    return args

@tim
def findCenter3(args,ds):
    args.it = 0
    args.optimizePDF = False
    if not args.bInitialized:
        args.C0Init,args.rho_c2Init,args.pol1_max,args.pol1_min,args.pol2_max,args.pol2_min,args.ENL, ds = \
            GetStatisticsFrmSample(args, ds)
        args.C,args.rho_c2 = args.C0Init,args.rho_c2Init
        args.bInitialized=True
        args.minI = da.array([args.pol1_min,args.pol2_min])
        args.maxI = da.array([args.pol1_max,args.pol2_max])
        args.delta = (args.maxI-args.minI)/args.nInter
        if args.bExport:
            args,ds = importDtSrc(args,ds)
            basinMask = da.where(ds.isWater == False, True, False)
            setattr(ds, 'jointProb_I_pol1', da.where(basinMask,ds.I_pol1,np.nan))
            setattr(ds, 'jointProb_I_pol2', da.where(basinMask,ds.I_pol2,np.nan))
            args.bExport = False
        with Timer('getNSCE 1'):
            args = getNSCE(args, ds)
        if args.PDFContaminated or not args.optimizePDF:
            if args.PDFContaminated:
                print('PDF contaminated')
                args.it = 0
                args.maxHH,args.minHH,args.maxHV,args.minHV = args.pol1_max,args.pol1_min,args.pol2_max,args.pol2_min
                #return args
        writeProg(args.logFile,args.thProb,args.NSCE,args.C,args.rho_c2,args.thPD,args.it,args.preENL,args.granules, args.ENL)
        # wether to use real ENL
        #args.ENL = args.preENL
    args.C0,args.rho_c2 = args.C0Init,args.rho_c2Init
    args.minI = da.array([args.pol1_min,args.pol2_min])
    args.maxI = da.array([args.pol1_max,args.pol2_max])
    if args.rho_c2<0:
        args.rho_c2=1e-3
    args.rho_c20 = args.rho_c2
    #print(args.maxI)
    args.maxI = args.maxI * args.PDFScaling
    #print('new maxI')
    #print(args.maxI)
    args.delta = (args.maxI-args.minI)/args.nInter
    if args.optIterAlg:
        args.COrg, args.C = args.C0, args.C0
        args.C0 = args.C * float('inf')
        print('Th=%.4f: iteration-' % args.thProb)
        err=abs((args.C-args.C0)/args.C)
        tol_dual, tol_single = 1e-2, 2e-2
        #if not os.path.exists(file2):
        #tol_raw = tol_dual
        tol=tol_dual
        while da.max(err)>tol:
            args.C0 = args.C.copy()
            #print(args.C0)
            #args.rho_c2 = args.rho_c20
            #if not os.path.exists(file2):
            args = polsar_func3(args,ds)
            #args.C = args.f0 - args.f
            args.it = args.it + 1
            err=da.abs(args.C-args.C0)/args.C0
            # compute the relative from euclidean distance
            #err = np.sqrt(np.sum((args.C-args.C0)**2))/np.sqrt(np.sum(args.C0**2))
            print('iter=%d, error=%.8f' % (args.it, da.max(err)))
            #print(np.abs(args.COrg-args.C)/args.COrg)
            if args.it>50:
                tol = 1e6
            #tol = (args.it // 3 + 1)*tol_raw
            #print(f'tol {tol}')
            if any(da.abs(args.COrg-args.C)/args.COrg > 2):
                print('wrong deviation during the optimization, try to decrease the probability threshold')
                #if dual
                args.C = da.array([np.nan,np.nan])
                args.rho_c2=np.NaN
                args.thPD=np.NaN
                args.NSCE=np.NaN
                break
            elif any(args.C/args.COrg <0.5):
                print('decreased too much from initial value, try to increase the probability threshold')
                #if dual
                args.C = da.array([np.nan,np.nan])
                args.rho_c2=np.NaN
                args.thPD=np.NaN
                args.NSCE=np.NaN
                break
        if any(da.isnan(args.C)):
            print('C all nan after this optimization')
            return args, ds
        with Timer('getNSCE 2'):
            args = getNSCE(args, ds, thPD = args.thPD)
    return args,ds

def initialstates(args,nDim):
    args.nTh=int((args.thProbMax-args.thProbMin)/args.dThProb)+1
    args.NSCE_arr = da.zeros((args.nTh,1))
    args.C_arr = da.zeros((args.nTh,nDim))
    args.rho_c2_arr = da.zeros((args.nTh,1))
    args.thPD_arr = da.zeros((args.nTh,1))
    args.it_arr = da.zeros((args.nTh,1))
    return args

def updatestates(args,i):
    args.C_arr[i,:] = args.C
    args.rho_c2_arr[i] = args.rho_c2
    args.NSCE_arr[i,:] = args.NSCE
    args.it_arr[i,:] = args.it
    args.thPD_arr [i] = args.thPD
    return args

@tim
def optimizeDist4(args,ds):
    nDim = 2
    if not os.path.exists(args.log_file_err):
        args = initialstates(args,nDim)
    finished=False
    args.preENL = args.ENL
    args.failed=False
    newStartMin=False
    args.ini_thPD = None
    while not finished:
        if args.thProbMin < 0.6: 
            print("Warning: minimal ThProb is lower than 0.6, yet an optimal is not found. The optimization is terminated")
            finished=True
            args.oC = args.C0Init
            args.oRho_c2 = args.rho_c2Init
            args.oThPD = args.ini_thPD
            return args,ds
        for i in range(args.nTh):
            args.PDFScaling = 1
            if i==0:
                newStartMin=True
            args.thProb = args.thProbMin+i*args.dThProb
            args.thProb = np.round(args.thProb, decimals=2)
            if args.NSCE_arr[i] != 0:
                #print('Computed probability, Skipped')
                pass
            else:
                args,ds = findCenter3(args,ds)
                args = updatestates(args,i)
                while args.thPD_arr[i]<0.1:
                    args.PDFScaling = args.PDFScaling + 1
                    print('The segment threshold of probability is too small. Please expand the PDF domain.\n Enlarge it and recompute!')
                    args,ds = findCenter3(args,ds)
                    args = updatestates(args,i)
                if args.PDFContaminated:
                    args.oC=args.C_arr[i,:];args.oRho_c2=args.rho_c2_arr[i];args.n_est=args.ENL;args.oThPD=args.thPD_arr[i];args.mNSCE=args.NSCE_arr[i];args.oThProb=np.nan
                    args.failed = True
                    return args, ds
                if ~da.isnan(args.NSCE_arr[i]):
                    print('probability= %.8f, has been optimized by %d iterations' % (args.thProb ,args.it_arr[i]))
                    print('NSCE = %.8f' % args.NSCE_arr[i])
                    #if nDim==2:
                    #    print('C=[%.8f,%.8f], Rho=%.8f' % (args.C[0],args.C[1], args.rho_c2))
                #need to write prog
                writeProg(args.logFile,args.thProb,args.NSCE,args.C,args.rho_c2,args.thPD,args.it,args.preENL,args.granules,args.ENL)
            if da.isnan(args.NSCE_arr[i]):
                if i==0:
                    print('The segment threshold of probability is too large. Reduce the lower limit to seek a potentially optimal thProb')
                    args.thProbMin=args.thProbMin-3*args.dThProb
                    args.thProbMax=args.thProbMax-3*args.dThProb
                    args = initialstates(args,nDim)
                elif i==1:
                    print('The segment threshold of probability is too large. Reduce the lower limit to seek a potentially optimal thProb')
                    args.thProbMax=args.thProbMin
                    args.thProbMin=args.thProbMin-2*args.dThProb
                    args.C_arr[-1,:] = args.C_arr[0,:]; args.C_arr[0:-1,:] = 0
                    args.rho_c2_arr[-1] = args.rho_c2_arr[0]; args.rho_c2_arr[0:-1] = 0
                    args.thPD_arr[-1] = args.thPD_arr[0]; args.thPD_arr[0:-1] = 0
                    args.it_arr[-1] = args.it_arr[0]; args.it_arr[0:-1] = 0
                    args.NSCE_arr[-1] = args.NSCE_arr[0]; args.NSCE_arr[0:-1] = 0
                else:
                    print('the current probability threshold is too large yet an optimal one is not found.')
                    finished=True
                break
            elif da.isposinf(args.NSCE_arr[i]) or da.isneginf(args.NSCE_arr[i]):
                print('too small lower limit')
                args.thProbMin=args.thProb+args.dThProb
                args.thProbMax=max(args.thProbMax,args.thProbMin+2*args.dThProb)
                args = initialstates(args,nDim)
                break
            if i>=1:
                if args.NSCE_arr[i-1]>args.NSCE_arr[i]:
                    if i==1:
                        print('Maximal NSCE found at the lower limit, reduce the lower limit to seek a potentially optimal thProb')
                        #finished=True
                        args.thProbMin = args.thProbMin - args.dThProb
                        a = da.empty((1,2), dtype=float, chunks=(1,2))
                        a[:] = np.nan
                        args.C_arr = da.vstack((a, args.C_arr))
                        args.rho_c2_arr = da.vstack((np.nan, args.rho_c2_arr))
                        args.thPD_arr = da.vstack((0, args.thPD_arr))
                        args.NSCE_arr = da.vstack((0, args.NSCE_arr))
                        args.it_arr = da.vstack((0, args.it_arr))
                        args.nTh = args.nTh + 1
                    else:
                        print('optimal probability found, save further iterations.')
                        finished=True
                    break
                elif args.NSCE_arr[i]>args.NSCE_arr[i-1] and args.NSCE_arr[i]!=0 and \
                    not da.isposinf(args.NSCE_arr[i]) and not da.isneginf(args.NSCE_arr[i])\
                        and i==args.nTh-1:
                    if args.thProbMax<0.99:
                        print('NSCE can be impoved. increase the maximal probability automatically')
                        args.thProbMax = args.thProbMax + args.dThProb
                        args.nTh = int((args.thProbMax - args.thProbMin) / args.dThProb) + 1
                        a = da.empty((1, 2), dtype=float, chunks=(1, 2))
                        a[:] = np.nan
                        args.C_arr = da.vstack((args.C_arr, a))
                        args.rho_c2_arr = da.vstack((args.rho_c2_arr, np.nan))
                        args.thPD_arr = da.vstack((args.thPD_arr, 0))
                        args.NSCE_arr = da.vstack((args.NSCE_arr, 0))
                        args.it_arr = da.vstack((args.it_arr, 0))
                    else:
                        print('Reached maximum ThProb (0.99) yet the NSCE is increasing. Iteration stops')
                        finished=True
                    break
    args.n_est=args.ENL
    #indexFailed = da.logical_or(args.NSCE_arr==0,da.isnan(args.NSCE_arr))
    #args.NSCE_arr[indexFailed]=-np.Inf
    iOpt = da.nanargmax(args.NSCE_arr)
    args.oC=args.C_arr[iOpt,:]
    args.oRho_c2=args.rho_c2_arr[iOpt]
    args.oThPD=args.thPD_arr[iOpt]
    args.mNSCE=args.NSCE_arr[iOpt]
    args.oThProb=args.thProbMin+int(iOpt-1)*args.dThProb
    return args, ds

@tim
def thSeg4(args,ds):
    args.fileOutHigh=os.path.join(args.dirOut,'waterMask_highProb.tif')
    args.fileOutMid=os.path.join(args.dirOut,'waterMask_modProb.tif')
    args.fileOutLow=os.path.join(args.dirOut,'waterMask_lowProb.tif')
    args.bcoutname = ['High_mask', 'Mod_mask','Low_mask']
    args.C, args.rho_c2, args.ThPD = args.oC, args.oRho_c2, args.oThPD
    if isinstance(args.C, da.Array):
        args.C = args.C.compute()
    if isinstance(args.rho_c2, da.Array):
        args.rho_c2 = args.rho_c2.compute()
    if isinstance(args.ThPD, da.Array):
        args.ThPD = args.ThPD.compute()
    dual_pol, hhgrid, hvgrid, deltaHH, deltaHV = getgrids(args)
    if dual_pol:
        args.ThPDL=args.ThPD/da.array([30,300])
    else:
        args.ThPDL=args.ThPD/da.array([5,17])
    probW=dualpol_dist(args,ds.I_pol1,ds.I_pol2)
    waterMask = da.where(probW>args.ThPD,1,0)
    High_mask = da.where(ds.img_lp>0, waterMask, 0).astype(np.uint8)    
    setattr(ds, args.bcoutname[0], High_mask)
    mask_profile = args.refmeta
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    High_mask_write = da.where(ds.I_pol1>0, waterMask, mask_profile['nodata']).astype(mask_profile['dtype'])
    with rasterio.open(args.fileOutHigh, 'w', **mask_profile, compress='deflate') as dst:
        dst.write(High_mask_write.compute(),indexes=1)
    #pol1_mask = da.where(waterMask==1,ds.I_pol1,np.nan)
    #pol2_mask = da.where(waterMask==1,ds.I_pol2,np.nan)
    pol1_mask = ds.I_pol1[waterMask==1]
    pol2_mask = ds.I_pol2[waterMask==1]
    pol1_max,pol1_min = da.max(pol1_mask),da.min(pol1_mask)
    pol2_max,pol2_min = da.max(pol2_mask),da.min(pol2_mask)
    savemodlowlist = [args.fileOutMid,args.fileOutLow]
    for idx,iThPD in enumerate(args.ThPDL):
        probmask = probW>iThPD
        if idx == 0:
            tmask = da.logical_and(da.logical_and(ds.I_pol1<pol1_max, ds.I_pol2<pol2_max), probmask)
        else:
            tmask = probmask
        waterMask= da.logical_or(tmask,da.logical_and(ds.I_pol1<pol1_min, ds.I_pol2<pol2_min))
        waterMask = da.where(ds.img_lp>0, waterMask, 0).astype(np.uint8)  
        setattr(ds, args.bcoutname[idx+1], waterMask)
        mask_profile = args.refmeta
        mask_profile.update(dtype=rasterio.uint8, nodata=255)
        waterMask_write = da.where(ds.I_pol1>0, waterMask, mask_profile['nodata']).astype(mask_profile['dtype'])
        with rasterio.open(savemodlowlist[idx], 'w', **mask_profile, compress='deflate') as dst:
            dst.write(waterMask_write.compute(),indexes=1)
    return args,ds

def Binary_classification(args,ds):
    #try:
    with Timer('BC TIME '+ args.filepath[0]):
        args, ds = optimizeDist4(args,ds)
        if not args.failed:
            print('finished optimizing')
        else:
            print('Cannot optimize..')
        args, ds = thSeg4(args,ds)
        #except Exception as e:
        #    print(e)
        if bool(args.tempfiles):
            for itemp in args.tempfiles:
                print('closing ' + itemp.name)
                itemp.close()
    return args,ds