import os
import sys
from utils import *
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
import dask
import dask.array as da
from dask import delayed, compute
from osgeo import ogr

class AncillaryData():    
    """
    Get ancillary data from the fly, based on the geometry and size of input xarray.

    Parameters
    ----------
    ref: input geotif file or the src of the ref file
    blockSize: blockSize for open_rasterio
    datamode: 'mem' direct input to memory (default) or 'tem' save to a tempfile. Use 'tmp' for machine with small memory.

    Returns
    -------
        Either a list of xarray objects or a stacked xarray object
    """
    def __init__(self, ref=None, chunk_size = 'auto', datamode='mem'):
        self.ref = ref
        self.chunk_size = chunk_size
        self.datamode = datamode
        self.bounds = self.getbounds()
        
    def getbounds(self):
        """
        Get the geo bounds of reference file
        
        Parameters
        ----------
        self
        
        Returns
        -------
        boundary in EPSG:4326
        """
        if isinstance(self.ref, str):
            # Open the TIFF file
            with rasterio.open(self.ref) as src:
                # Get the bounds of the image
                # Convert the bounds to latitude and longitude
                latlon_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        else:
            assert self.ref != None
            # Use src directly
            latlon_bounds = transform_bounds(self.ref.crs, "EPSG:4326", *self.ref.bounds)
        return latlon_bounds
    
    @tim
    def getWO(self):
        """
        Get water occurence data from ESA, based on the geobounds of reference tif or src.
        Resample and clip with rasterio.

        Returns
        -------
            Dask array or tempfile of water occurence
        """
        
        Hcrop_Ex = self.bounds
        maxLoESA = int(np.ceil(Hcrop_Ex[2] / 10) * 10)
        minLoESA = int(np.floor(Hcrop_Ex[0] / 10) * 10)
        maxLaESA = int(np.ceil(Hcrop_Ex[3] / 10) * 10 + 10)
        minLaESA = int(np.ceil(Hcrop_Ex[1] / 10) * 10)
        
        if maxLoESA > 0 and minLoESA > 0:
            lons = [str(e) + 'E' for e in range(minLoESA, maxLoESA, 10)]
        elif maxLoESA < 0 and minLoESA < 0:
            lons = [str(w) + 'W' for w in np.absolute(range(minLoESA, maxLoESA, 10))]
        else:
            lons = [str(w) + 'W' for w in np.absolute(range(minLoESA, 0, 10))]
            lons.extend([str(e) + 'E' for e in range(0, maxLoESA, 10)])

        if maxLaESA > 0 and minLaESA > 0:
            lats = [str(n) + 'N' for n in range(minLaESA, maxLaESA, 10)]
        elif maxLaESA < 0 and minLaESA < 0:
            lats = [str(s) + 'S' for s in np.absolute(range(minLaESA, maxLaESA, 10))]
        else:
            lats = [str(s) + 'S' for s in np.absolute(range(minLaESA, 0, 10))]
            lats.extend([str(n) + 'N' for n in range(0, maxLaESA, 10)])
        
        #ESArevision = '1_3_2020'
        ESArevision = '1_4_2021'
        ESAdatasets = ['occurrence']
        
        url_tmpl, file_tmpl, _ = templatesESAwater(ESArevision)
        elements=[]
        for ds_name in ESAdatasets:
            for lon in lons:
                for lat in lats:
                    filename = file_tmpl.format(ds=ds_name, lon=lon, lat=lat)
                    url = url_tmpl.format(ds=ds_name, file=filename)
                    print(url)
                    elements.append(url)
        merged_resandclip = self.merge_res(elements)
        return merged_resandclip

    @tim
    def getLCC(self):
        """
        Get land cover data from GongPeng LCC, based on the geobounds of reference tif or src.
        Resample and clip with rasterio.

        Returns
        -------
            Dask array or tempfile of land cover
        """
        
        Hcrop_Ex = self.bounds
        """
        Get the intersect copernicus glo30 data file list based on bounds and offical geojson
        """
        s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
        # Open the geojson file
        DEM_GEOJSON = f'{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson'
        dataSource = ogr.Open(DEM_GEOJSON)
        layer = dataSource.GetLayer()
        # Create an OGR geometry object from the Hcrop_Ex
        # Create an empty polygon
        extent_geom = ogr.Geometry(ogr.wkbPolygon)
        # Add the vertices of the bounding box
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(Hcrop_Ex[0], Hcrop_Ex[1])
        ring.AddPoint(Hcrop_Ex[2], Hcrop_Ex[1])
        ring.AddPoint(Hcrop_Ex[2], Hcrop_Ex[3])
        ring.AddPoint(Hcrop_Ex[0], Hcrop_Ex[3])
        ring.AddPoint(Hcrop_Ex[0], Hcrop_Ex[1])
        extent_geom.AddGeometry(ring)
        # Loop over features and find intersecting features
        elements = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            if extent_geom.Intersects(geom):
                tile = feature.GetField('ll_tile')
                file_path = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
                print(file_path)
                elements.append(file_path)
        merged_resandclip = self.merge_res(elements)
        return merged_resandclip
    
    def getLCC_ori(self):
        """
        Get land cover data from GongPeng LCC, based on the geobounds of reference tif or src.
        Resample and clip with rasterio.

        Returns
        -------
            Dask array or tempfile of land cover
        """
        
        Hcrop_Ex = self.bounds
        #LCCmainURL = "http://data.ess.tsinghua.edu.cn/data/fromglc10_2017v01/fromglc10v01" % expired on May/03/2023, refer to http://data.ess.tsinghua.edu.cn/
        LCCmainURL = '/gpfs/sharedfs1/manoslab/data/Land_Cover/GPfromglc10_2017v01/fromglc10v01'
        LCCfileregex = "_LA_LO.tif"
        maxLoLCC = np.ceil(Hcrop_Ex[2])
        minLoLCC = np.floor(Hcrop_Ex[0])
        maxLaLCC = np.ceil(Hcrop_Ex[3])
        minLaLCC = np.floor(Hcrop_Ex[1])
        Lo_arange = np.arange(minLoLCC - 1, maxLoLCC + 1, 1)
        La_arange = np.arange(minLaLCC - 1, maxLaLCC + 1, 1)
        Lo_arangeGPLCC = Lo_arange[Lo_arange % 2 == 0]
        La_arangeGPLCC = La_arange[La_arange % 2 == 0]
        elements = []
        for Lo in Lo_arangeGPLCC:
            for La in La_arangeGPLCC:
                LCCfileName = LCCfileregex.replace("LO", str(int(Lo)))
                LCCfileName = LCCfileName.replace("LA", str(int(La)))
                url = LCCmainURL+LCCfileName
                print(url)
                elements.append(url)
        merged_resandclip = self.merge_res(elements)
        return merged_resandclip
    
    def glo30list(bounds:tuple) -> list:
        """
        Get the intersect copernicus glo30 data file list based on bounds and offical geojson
        """
        # Open the geojson file
        DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30-2021.geojson'
        dataSource = ogr.Open(DEM_GEOJSON)
        layer = dataSource.GetLayer()
        # Create an OGR geometry object from the bounds
        # Create an empty polygon
        extent_geom = ogr.Geometry(ogr.wkbPolygon)
        # Add the vertices of the bounding box
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(bounds[0], bounds[1])
        ring.AddPoint(bounds[2], bounds[1])
        ring.AddPoint(bounds[2], bounds[3])
        ring.AddPoint(bounds[0], bounds[3])
        ring.AddPoint(bounds[0], bounds[1])
        extent_geom.AddGeometry(ring)
        # Loop over features and find intersecting features
        elements = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            if extent_geom.Intersects(geom):
                file_path = feature.GetField('file_path')
                print(file_path)
                elements.append(file_path)
        # Close the data source
        dataSource = None
        return elements

    def merge_res(self, rasterfiles):
        """
        merge, reproject and resample rasters to the geoextent of reference

        Parameters
        ----------
        reffile: reference file path or src of the reference file.
        rasterfiles: list of raster files to be processed
        Returns
        -------
            dask array
        """
        #@dask.delayed
        #def reproject_clip(in_path, refsrc):
        #    with reproject_raster(in_path, refsrc) as in_mem_ds:
        #        rc_data = da.from_array(in_mem_ds.read(1), chunks = self.chunk_size)
        #    return rc_data
        reffile = self.ref
        results = []
        if isinstance(reffile, str):
            #input with file path
            with rasterio.open(reffile) as ref:
                refsrc = ref
        else:
            #input with ref src directly
            refsrc = reffile
        refsrc = refsrc.meta.copy()
        with rasterio.open(rasterfiles[0]) as src:
            src_dtype = src.dtypes[0]
        data = [dask.delayed(reproject_clip_readsrc)(url,refsrc) for url in rasterfiles]
        results = dask.compute(*data)
        merged_array_raw = da.nanmax(da.stack(results, axis=0),axis=0).astype(src_dtype).squeeze()
        if self.datamode == 'mem':
            #save in memory
            temp = da.from_array(merged_array_raw.compute(), chunks = self.chunk_size)
        else:
            ##return a temp file instead
            import tempfile
            # Create a temporary file
            temp = tempfile.NamedTemporaryFile()
            da.to_hdf5(temp.name,'/merged_array_raw',merged_array_raw)
        del results,merged_array_raw
        return temp
    
def templatesESAwater(revision):
    """
    Return url templates for ESA data download.

    Parameters
    ----------
    revision: version of the data

    Returns
    -------
        url and filename templates as well as padding
    """
    REVISIONS = ['1_0', '1_1', '1_1_2019', '1_3_2020', '1_4_2021']
    v10, v11, v11_2019, v13_2020,v14_2021 = REVISIONS
    url_tmpl = 'http://storage.googleapis.com/global-surface-water/downloads'
    file_tmpl = '{ds}_{lon}_{lat}'
    if revision == v10:
        padding = 15
    elif revision == v11:
        url_tmpl += '2'
        file_tmpl += '_v' + v11
        padding = 20
    elif revision == v11_2019:
        url_tmpl += '2019v2'
        file_tmpl += 'v' + v11_2019
        padding = 24
    elif revision == v13_2020:
        url_tmpl += '2020'
        file_tmpl += 'v' + v13_2020
        padding = 24
    elif revision == v14_2021:
        url_tmpl += '2021'
        file_tmpl += 'v' + v14_2021
        padding = 24
    url_tmpl += '/{ds}/{file}'
    file_tmpl += '.tif'
    return (url_tmpl, file_tmpl, padding)
