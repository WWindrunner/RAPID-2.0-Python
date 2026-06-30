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
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def tim(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
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
            func.__name__, mem_after - mem_before, mem_before, mem_after))
        return result

    return wrapper


def reproject_clip_readsrc(in_rasfile, ref, resampling=rasterio.warp.Resampling.nearest):
    with rasterio.open(in_rasfile) as src:
        in_ras = src.read()
        src_meta = src.meta.copy()
        out_arr = reproject_clip_direct(
            in_ras,
            src_meta,
            ref,
            resampling=resampling
        )
    return out_arr


def reproject_clip_direct(in_ras, src, ref, resampling=rasterio.warp.Resampling.nearest):
    dest_shape = (src['count'], ref['height'], ref['width'])

    dst_nodata = src.get('nodata', None)
    if dst_nodata is None:
        dst_nodata = 0

    out_arr = np.full(dest_shape, dst_nodata, dtype=src['dtype'])

    rasterio.warp.reproject(
        source=in_ras,
        destination=out_arr,
        src_transform=src['transform'],
        src_crs=src['crs'],
        dst_transform=ref['transform'],
        dst_crs=ref['crs'],
        resampling=resampling,
        src_nodata=src.get('nodata', None),
        dst_nodata=dst_nodata,
    )

    return out_arr


def read_window_from_ds(ds, name: str, r0: int, c0: int, nRows: int, nCols: int):
    """
    Read a window from ds.<name> (dask array) and return numpy.
    Equivalent to MATLAB ReadRaster(file, r0,c0,nRows,nCols) + SetNull (if ds already has NaNs).
    """
    if not hasattr(ds, name):
        raise AttributeError(f"ds has no attribute '{name}'. Available: {sorted(ds.__dict__.keys())}")

    arr = getattr(ds, name)
    patch = arr[r0:r0 + nRows, c0:c0 + nCols]
    if hasattr(patch, "compute"):
        patch = patch.compute()
    return np.asarray(patch)
