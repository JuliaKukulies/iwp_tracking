"""
iwp_tracking
============

This Python package provides functionality for the IWP tracking study
by Kukschuhlies et al.
"""
import numpy as np
from pyresample import create_area_def
import xarray as xr

CPCIR_GRID = create_area_def(
    "cpcir_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution=(0.03637833468067906, 0.036385688295936934),
    units="degrees",
    description="CPCIR grid",
)

#
# Determine index offsets for conus grid.
#

mrms_lon_min = -129.995
mrms_lon_max = -60.00500199999999
mrms_lat_min = 20.005001
mrms_lat_max = 54.995

lons_cpcir, lats_cpcir = CPCIR_GRID.get_lonlats()
lons_cpcir = lons_cpcir[0]
lats_cpcir = lats_cpcir[:, 0]

col_start = np.where(lons_cpcir > mrms_lon_min)[0][0]
col_end = np.where(lons_cpcir < mrms_lon_max)[0][-1]
row_start = np.where(lats_cpcir < mrms_lat_max)[0][0]
row_end = np.where(lats_cpcir > mrms_lat_min)[0][-1]

CONUS_GRID = CPCIR_GRID[row_start:row_end, col_start:col_end]


def subset_cpcir_data(dataset: xr.Dataset) -> xr.Dataset:
    """
    Subset CPCIR data to CONUS.

    Args:
        dataset: An xarray.Dataset to subset.

    Return:
        The dataset restricted to CONUS.
    """
    return dataset[
        {"latitude": slice(row_start, row_end), "longitude": slice(col_start, col_end)}
    ]
