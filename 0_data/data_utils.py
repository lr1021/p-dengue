import geopandas as gpd
import rioxarray as rxr
from rasterstats import zonal_stats
import pandas as pd
import os
import rasterio
from rasterio.mask import mask
import numpy as np
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

def align_r1_to_r2(
    r1,
    r2,
    data_type="categorical"   # "categorical" or "continuous"
):
    """
    Align raster r1 to the grid of raster r2.

    Parameters
    ----------
    r1 : xarray.DataArray
        Source raster to be aligned.
    r2 : xarray.DataArray
        Target raster defining CRS, resolution, extent, and grid.
    data_type : str
        "categorical" or "continuous".
    clip : bool
        If True, crop r1 to r2 extent before resampling.

    Returns
    -------
    xarray.DataArray
        r1 aligned to r2 grid.
    """
    # 3. Choose resampling method
    if data_type == "nearest":
        resampling = Resampling.nearest
    elif data_type == "bilinear":
        resampling = Resampling.bilinear

    # 4. Resample onto r2 grid
    r1_aligned = r1.rio.reproject_match(
        r2,
        resampling=resampling
    )
    return r1_aligned, r2

def best_res_align(r1, r1catcon, r2, r2catcon,
                   region_bounds_buffered,
                   shape_crs, crop=True):
    """
    Align the raster with lower resolution to the raster with higher resolution.

    Parameters
    ----------
    r1 : xarray.DataArray
        First raster.
    r1catcon : str
        "categorical" or "continuous" for r1.
    r2 : xarray.DataArray
        Second raster.
    r2catcon : str
        "categorical" or "continuous" for r2.

    Returns
    -------
    aligned_r1 : xarray.DataArray
        r1 aligned to the higher-resolution raster grid (r2 or r1 swapped if needed).
    target_raster : xarray.DataArray
        The raster whose grid is being used as the reference.
    """

    # 1. CRS check / reproject if needed
    if r1.rio.crs != shape_crs:
        r1 = r1.rio.reproject(shape_crs)
    if r2.rio.crs != shape_crs:
        r2 = r2.rio.reproject(shape_crs)

    if crop:
        r1 = r1.squeeze().rio.clip_box(minx=region_bounds_buffered[0], miny=region_bounds_buffered[1],
                                                    maxx=region_bounds_buffered[2], maxy=region_bounds_buffered[3])
        r2 = r2.squeeze().rio.clip_box(minx=region_bounds_buffered[0], miny=region_bounds_buffered[1],
                                                    maxx=region_bounds_buffered[2], maxy=region_bounds_buffered[3])

    # 2. Compare resolutions
    res_r1_x, res_r1_y = r1.rio.resolution()
    # print('r1 resolution:', res_r1_x, res_r1_y)
    res_r2_x, res_r2_y = r2.rio.resolution()
    # print('r2 resolution:', res_r2_x, res_r2_y)

    # Use the smaller cell size as the target (higher-resolution raster)
    r1_avg_res = (abs(res_r1_x) + abs(res_r1_y)) / 2
    r2_avg_res = (abs(res_r2_x) + abs(res_r2_y)) / 2

    if r1_avg_res > r2_avg_res:
        # print('r1 is lower resolution, downsampling r1 to r2')
        # r1 is coarser -> align r1 to r2
        return align_r1_to_r2(r1, r2, data_type=r1catcon)
    else:
        # print('r2 is lower resolution, downsampling r2 to r1')
        # r2 is coarser -> align r2 to r1
        return align_r1_to_r2(r2, r1, data_type=r2catcon)[::-1]
