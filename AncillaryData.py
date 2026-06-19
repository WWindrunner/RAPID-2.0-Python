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
from rasterio.transform import Affine
import tempfile
import subprocess


class AncillaryData:
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

    def __init__(self, ref=None, chunk_size='auto', datamode='mem'):
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

        # ESArevision = '1_3_2020'
        ESArevision = '1_4_2021'
        ESAdatasets = ['occurrence']

        url_tmpl, file_tmpl, _ = templatesESAwater(ESArevision)
        elements = []
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

    def getOceanMask(self):
        """
        Get global ocean mask from Natural Earth 10m ocean shapefile.
        Rasterize using rasterio.features (no GDAL needed),
        then reproject / resample / clip via merge_res.

        Returns
        -------
            dask array or tempfile
            1 = ocean, 0 = land
        """

        import zipfile
        import tempfile
        import urllib.request
        import json
        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        # --------------------------------------------------
        # 1. Download Natural Earth ocean shapefile
        # --------------------------------------------------
        url = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_ocean.zip"

        tmpdir = tempfile.mkdtemp()
        zip_path = os.path.join(tmpdir, "ne_10m_ocean.zip")

        print(url)
        urllib.request.urlretrieve(url, zip_path)

        # --------------------------------------------------
        # 2. Unzip
        # --------------------------------------------------
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        shp_path = os.path.join(tmpdir, "ne_10m_ocean.shp")
        if not os.path.exists(shp_path):
            raise FileNotFoundError("ne_10m_ocean.shp not found after unzip")

        # --------------------------------------------------
        # 3. Read geometries with OGR
        # --------------------------------------------------
        ds = ogr.Open(shp_path)
        layer = ds.GetLayer()

        shapes = []
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue

            # rasterio.rasterize needs GeoJSON-like dict, not JSON string
            geom_json = json.loads(geom.ExportToJson())
            shapes.append((geom_json, 1))

        if len(shapes) == 0:
            raise RuntimeError("No geometries found in ocean shapefile")

        # --------------------------------------------------
        # 4. Rasterize to a coarse global raster
        #    (resolution / extent DO NOT matter much,
        #     merge_res will fix everything)
        # --------------------------------------------------
        res = 0.1  # degrees, very coarse is fine
        xmin, ymin, xmax, ymax = -180.0, -90.0, 180.0, 90.0

        width = int((xmax - xmin) / res)
        height = int((ymax - ymin) / res)

        transform = from_origin(xmin, ymax, res, res)

        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # --------------------------------------------------
        # 5. Write temporary tif
        # --------------------------------------------------
        ocean_tif = os.path.join(tmpdir, "ne_10m_ocean.tif")

        with rasterio.open(
            ocean_tif,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=transform,
            nodata=None,
            compress="lzw",
        ) as dst:
            dst.write(mask, 1)

        # --------------------------------------------------
        # 6. Let merge_res do reprojection / resample / clip
        # --------------------------------------------------
        return self.merge_res([ocean_tif])

    @tim
    def getLocalRaster(self, rasterfile, resampling="nearest"):
        """
        Load user-provided local raster and align it to the reference SAR grid.

        Parameters
        ----------
        rasterfile : str
            Path to local raster.

        resampling : str
            "nearest" for masks/classes;
            "bilinear" for continuous variables.

        Returns
        -------
        Dask array or tempfile
            Raster aligned to the reference image.
        """

        print(rasterfile)

        if resampling == "nearest":
            resampling_method = rasterio.warp.Resampling.nearest
        elif resampling == "bilinear":
            resampling_method = rasterio.warp.Resampling.bilinear
        elif resampling == "cubic":
            resampling_method = rasterio.warp.Resampling.cubic
        else:
            raise ValueError(f"Unsupported resampling method: {resampling}")

        merged_resandclip = self.merge_res(
            [rasterfile],
            resampling=resampling_method
        )

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
        # LCCmainURL = "http://data.ess.tsinghua.edu.cn/data/fromglc10_2017v01/fromglc10v01" % expired on May/03/2023, refer to http://data.ess.tsinghua.edu.cn/
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
                url = LCCmainURL + LCCfileName
                print(url)
                elements.append(url)
        merged_resandclip = self.merge_res(elements)
        return merged_resandclip

    @staticmethod
    def glo30list(bounds: tuple) -> list:
        """
        Get intersecting Copernicus GLO-30 DEM files based on bounds in EPSG:4326.
        """
        DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30-2021.geojson'

        dataSource = ogr.Open(DEM_GEOJSON)
        if dataSource is None:
            raise RuntimeError(f"Failed to open DEM geojson: {DEM_GEOJSON}")

        layer = dataSource.GetLayer()

        extent_geom = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(bounds[0], bounds[1])
        ring.AddPoint(bounds[2], bounds[1])
        ring.AddPoint(bounds[2], bounds[3])
        ring.AddPoint(bounds[0], bounds[3])
        ring.AddPoint(bounds[0], bounds[1])
        extent_geom.AddGeometry(ring)

        elements = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is not None and extent_geom.Intersects(geom):
                file_path = feature.GetField('file_path')
                print(file_path)
                elements.append(file_path)

        dataSource = None

        if len(elements) == 0:
            raise RuntimeError(f"No Copernicus GLO-30 DEM tile found for bounds: {bounds}")

        return elements

    @tim
    def getTerrainMask(self, slope_th=20.0, dem_res=90.0):
        """
        Generate DEM-derived terrain exclusion mask.

        This function does NOT return DEM.
        It only returns a 0/1 terrain mask aligned to the reference SAR grid.

        1 = terrain area to exclude
        0 = keep

        Parameters
        ----------
        slope_th : float
            Slope threshold in degrees.
        dem_res : float
            Coarse DEM processing resolution in meters.

        Returns
        -------
        dask array or tempfile
            Terrain exclusion mask aligned with reference image.
        """

        dem_files = self.glo30list(self.bounds)

        tmpdir = tempfile.mkdtemp(prefix="rapid_dem_")

        tile_list_txt = os.path.join(tmpdir, "dem_tiles.txt")
        dem_vrt = os.path.join(tmpdir, "dem.vrt")
        dem_coarse_tif = os.path.join(tmpdir, "dem_coarse.tif")
        terrain_coarse_tif = os.path.join(tmpdir, "terrain_mask_coarse.tif")
        terrain_ref_tif = os.path.join(tmpdir, "terrain_mask_ref.tif")

        with open(tile_list_txt, "w") as f:
            for p in dem_files:
                f.write(p + "\n")

        # ---- reference metadata ----
        if isinstance(self.ref, str):
            with rasterio.open(self.ref) as ref:
                ref_crs = ref.crs
                ref_transform = ref.transform
                ref_bounds = ref.bounds
                ref_width = ref.width
                ref_height = ref.height
                ref_meta = ref.meta.copy()
        else:
            ref = self.ref
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_bounds = ref.bounds
            ref_width = ref.width
            ref_height = ref.height
            ref_meta = ref.meta.copy()

        # ---- build DEM VRT ----
        cmd_vrt = [
            "gdalbuildvrt",
            "-input_file_list", tile_list_txt,
            dem_vrt
        ]
        subprocess.run(cmd_vrt, check=True)

        # ---- choose coarse resolution in reference CRS units ----
        if ref_crs.is_geographic:
            # dem_res is meters; convert to approximate degree spacing.
            center_lat = (ref_bounds.bottom + ref_bounds.top) / 2.0

            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(center_lat))

            xres = dem_res / meters_per_deg_lon
            yres = dem_res / meters_per_deg_lat
        else:
            # projected CRS, usually meters
            xres = dem_res
            yres = dem_res

        # ---- warp DEM to coarse reference CRS / extent ----
        cmd_warp_dem = [
            "gdalwarp",
            "-overwrite",
            "-t_srs", ref_crs.to_string(),
            "-te",
            str(ref_bounds.left),
            str(ref_bounds.bottom),
            str(ref_bounds.right),
            str(ref_bounds.top),
            "-tr", str(xres), str(yres),
            "-r", "bilinear",
            "-dstnodata", "-9999",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            dem_vrt,
            dem_coarse_tif
        ]
        subprocess.run(cmd_warp_dem, check=True)

        # ---- calculate slope on coarse DEM ----
        with rasterio.open(dem_coarse_tif) as src:
            dem = src.read(1).astype(np.float32)
            dem_transform = src.transform
            dem_crs = src.crs
            dem_nodata = src.nodata
            dem_profile = src.profile.copy()

        if dem_nodata is not None:
            dem[dem == dem_nodata] = np.nan

        dx = abs(dem_transform.a)
        dy = abs(dem_transform.e)

        if dem_crs.is_geographic:
            nrows = dem.shape[0]
            center_lat = dem_transform.f + dem_transform.e * (nrows / 2.0)

            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(center_lat))

            dx = dx * meters_per_deg_lon
            dy = dy * meters_per_deg_lat

        dzdy, dzdx = np.gradient(dem, dy, dx)
        slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
        slope_deg = np.rad2deg(slope_rad)

        terrain_mask = (
                np.isfinite(slope_deg)
                & (slope_deg > slope_th)
        ).astype(np.uint8)

        # ---- write coarse terrain mask ----
        dem_profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
            tiled=True
        )

        with rasterio.open(terrain_coarse_tif, "w", **dem_profile) as dst:
            dst.write(terrain_mask, 1)

        # ---- warp terrain mask back to exact SAR grid ----
        cmd_warp_mask = [
            "gdalwarp",
            "-overwrite",
            "-t_srs", ref_crs.to_string(),
            "-te",
            str(ref_bounds.left),
            str(ref_bounds.bottom),
            str(ref_bounds.right),
            str(ref_bounds.top),
            "-ts", str(ref_width), str(ref_height),
            "-r", "near",
            "-dstnodata", "0",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            terrain_coarse_tif,
            terrain_ref_tif
        ]
        subprocess.run(cmd_warp_mask, check=True)

        # ---- return in the same style as other ancillary data ----
        if self.datamode == "mem":
            with rasterio.open(terrain_ref_tif) as src:
                arr = src.read(1).astype(np.uint8)

            temp = da.from_array(arr, chunks=self.chunk_size)

        else:
            temp = tempfile.NamedTemporaryFile()
            with rasterio.open(terrain_ref_tif) as src:
                arr = src.read(1).astype(np.uint8)

            da.to_hdf5(temp.name, "/merged_array_raw", da.from_array(arr, chunks=self.chunk_size))

        return temp

    def merge_res(self, rasterfiles, resampling=rasterio.warp.Resampling.nearest):
        """
        merge, reproject and resample rasters to the geoextent of reference
        """

        reffile = self.ref
        results = []

        if isinstance(reffile, str):
            with rasterio.open(reffile) as ref:
                refsrc = ref.meta.copy()
        else:
            refsrc = reffile.meta.copy()

        with rasterio.open(rasterfiles[0]) as src:
            src_dtype = src.dtypes[0]

        data = [
            dask.delayed(reproject_clip_readsrc)(
                url,
                refsrc,
                resampling=resampling
            )
            for url in rasterfiles
        ]

        results = dask.compute(*data)

        merged_array_raw = (
            da.nanmax(da.stack(results, axis=0), axis=0)
            .astype(src_dtype)
            .squeeze()
        )

        if self.datamode == 'mem':
            temp = da.from_array(
                merged_array_raw.compute(),
                chunks=self.chunk_size
            )
        else:
            import tempfile
            temp = tempfile.NamedTemporaryFile()
            da.to_hdf5(
                temp.name,
                '/merged_array_raw',
                merged_array_raw
            )

        del results, merged_array_raw
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
    v10, v11, v11_2019, v13_2020, v14_2021 = REVISIONS
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
