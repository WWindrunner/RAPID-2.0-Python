import os,sys,joblib
from datetime import datetime

import pandas as pd
import numpy as np
import dask.dataframe as dd

#from dask.distributed import Client

#from dask_ml.linear_model import LogisticRegression
from sklearn import linear_model

#client = Client()
start_time = datetime.now()

fullname = sys.argv[1]
fname = os.path.basename(fullname)
if 'River' in fname:
    cols = ['tIndTable','tDEMDiff','tDEMDiffMax','tDEMRatio',
            'tProb','tProbMin','tFAC','aveDistLimUpper',
            'aveDistLimLower','tDist','tW']
    type = {'tIndTable': 'int64',
                'tDEMDiff': 'float64',
                'tDEMDiffMax': 'float64',
                'tDEMRatio': 'float64',
                'tProb': 'float64',
                'tProbMin': 'float64',
                'tFAC': 'float64',
                'aveDistLimUpper': 'float64',
                'aveDistLimLower': 'float64',
                'tDist': 'float64',
                'tW': 'float64'}
elif 'Lake' in fname:
    cols = ['tIndTable', 
            'tDEMDiff', 'tDEM99pDiff', 'tDEM97pDiff', 'tDEM95pDiff', 'tDEM90pDiff', 'tDEMDiffMax', 
            'tProb','tProb1p', 'tProb2p', 'tProb5p', 'tProb10p', 'tProb20p',
            'tW']
    type = {'tIndTable': 'int64',
            'tDEMDiff': 'float64',
            'tDEM99pDiff': 'float64',
            'tDEM97pDiff': 'float64',
            'tDEM95pDiff': 'float64',
            'tDEM90pDiff': 'float64',
            'tDEMDiffMax': 'float64',
            'tProb': 'float64',
            'tProb1p': 'float64',
            'tProb2p': 'float64',
            'tProb5p': 'float64',
            'tProb10p': 'float64',
            'tProb20p': 'float64',
            'tW': 'float64'}
else:
    print(f"Check File name convention: {sys.argv[1]} . Either 'River' or 'Lake'")
    quit()

print(cols)

#ddf = dd.read_hdf(fullname,'data',columns=cols)
ddf = dd.read_csv(fullname,blocksize=25e8,dtype=type)

if 'tFAC' in cols:
    ddf['tFAC'] = np.log(ddf['tFAC']) # replace if necessary

#ddf = ddf[np.isfinite(ddf).all(1)]

#df = ddf.compute()

X = ddf[cols[1:-1]].compute()  #independent variables
y = ddf[cols[-1]].compute()

time_elapsed = datetime.now() - start_time
print('Phase 1(Construct Data Set[X,y]): Time elapsed (hh:mm:ss.ms){}'.format(time_elapsed))

#sklearn logit
logr = linear_model.LogisticRegression(solver='newton-cg') #,max_iter=500)  choose the solver
logr.fit(X.values,y.values)
tw = logr.predict_proba(X.values)[:,1]#get the probability of water class
df_out = pd.DataFrame(tw, columns = ['tW_Pred'])
df_out.to_csv(sys.argv[2],index=False)#temp output fot MATLAB correction
# dask ML Logit
#logr = LogisticRegression(solver_kwargs={"normalize":False})  # dont converge for large data set
#logr.fit(X.values,y.values)

#save model 
#modelfile = sys.argv[2] #'logr_model.sav'
#joblib.dump(logr,modelfile)

time_elapsed = datetime.now() - start_time
print('Phase 2(Logistic Regression on the data set): Time elapsed (hh:mm:ss.ms){}'.format(time_elapsed))

#client.shutdown()

score = logr.score(X.values,y.values)
print(f'logit model score is: {score:.4f}')