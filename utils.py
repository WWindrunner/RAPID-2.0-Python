import os
import time
import psutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform

class Timer(object):
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def tim(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
        
# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
def profile(func):
    # decorator function
    # to estimate memory usage for each function
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}, before:{:,}; After:{:,};".format(
            func.__name__, mem_after - mem_before,mem_before,mem_after))
        return result
    return wrapper

def reproject_clip_readsrc(in_rasfile,ref):
    with rasterio.open(in_rasfile) as src:
        in_ras = src.read()
        src_meta = src.meta.copy()
        out_arr = reproject_clip_direct(in_ras,src_meta,ref)
    return out_arr

def reproject_clip_direct(in_ras,src,ref):
    dest_shape = (src['count'], ref['height'], ref['width'])
    out_arr = np.zeros(dest_shape, dtype = src['dtype'])
    rasterio.warp.reproject(
        source=in_ras,
        destination=out_arr,
        src_transform=src['transform'],
        src_crs=src['crs'],
        dst_transform=ref['transform'],
        dst_crs=ref['crs'],
        resampling=rasterio.warp.Resampling.nearest,
        src_nodata=src['nodata'],
        dst_nodata=src['nodata'],
        )
    return out_arr