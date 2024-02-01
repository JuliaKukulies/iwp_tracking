"""
iwp_tracking.collocations
=========================

Functionality for extracting collocations between GPM DPR and TOBAC tracks.
"""
import logging
from pyresample.geometry import AreaDefinition
from math import ceil

import numpy as np
import pandas as pd
import xarray as xr
from pansat import TimeRange
from pansat.geometry import LonLatRect
from pansat.time import to_datetime
from pansat.environment import get_index
from pansat.granule import merge_granules
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree


LOGGER = logging.getLogger(__name__)


def get_equal_area_grid(
    longitude: float, latitude: float, resolution: float = 4e3, extent: float = 400e3
) -> AreaDefinition:
    """
    Get equal area grid center on a point.

    Args:
        longitude: The longitude coordinate of the point.
        latitude: The latitude coordinate of the point.
        resolution: The resolution of the grid.
        extent: The extent of the full grid.

    Return:
        A AreaDefinition object defining an equal area grid center on the
        given point.
    """
    width = ceil(extent / resolution)
    return AreaDefinition(
        "collocation_grid",
        "Equal-area grid for collocation",
        "colocation_grid",
        projection={
            "proj": "laea",
            "lon_0": longitude,
            "lat_0": latitude,
            "units": "m",
        },
        width=width,
        height=width,
        area_extent=(-extent / 2.0, -extent / 2.0, extent / 2.0, extent / 2.0),
    )


def resample_data(
    dataset, target_grid, radius_of_influence=5e3, new_dims=("latitude", "longitude")
) -> xr.Dataset:
    """
    Resample xarray.Dataset data to global grid.

    Args:
        dataset: xr.Dataset containing data to resample to global grid.
        target_grid: A pyresample.AreaDefinition defining the global grid
            to which to resample the data.

    Return:
        An xarray.Dataset containing the give dataset resampled to
        the global grid.
    """
    lons = dataset.longitude.data
    lats = dataset.latitude.data
    if isinstance(target_grid, tuple):
        lons_t, lats_t = target_grid
        shape = lons_t.shape
    else:
        lons_t, lats_t = target_grid.get_lonlats()
        shape = target_grid.shape

    valid_pixels = (
        (lons_t >= lons.min())
        * (lons_t <= lons.max())
        * (lats_t >= lats.min())
        * (lats_t <= lats.max())
    )

    swath = SwathDefinition(lons=lons, lats=lats)
    target = SwathDefinition(lons=lons_t[valid_pixels], lats=lats_t[valid_pixels])

    info = kd_tree.get_neighbour_info(
        swath, target, radius_of_influence=radius_of_influence, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    resampled = {}
    resampled["latitude"] = (("latitude",), lats_t[:, 0])
    resampled["longitude"] = (("longitude",), lons_t[0, :])

    for var in dataset:
        if var in ["latitude", "longitude"]:
            continue
        data = dataset[var].data
        if data.ndim == 1 and lons.ndim > 1:
            data = np.broadcast_to(data[:, None], lons.shape)

        dtype = data.dtype
        if np.issubdtype(dtype, np.datetime64):
            fill_value = np.datetime64("NaT")
        elif np.issubdtype(dtype, np.integer):
            fill_value = -9999
        elif dtype == np.int8:
            fill_value = -1
        else:
            fill_value = np.nan

        print(var)
        data_r = kd_tree.get_sample_from_neighbour_info(
            "nn", target.shape, data, ind_in, ind_out, inds, fill_value=fill_value
        )

        data_full = np.zeros(shape + data.shape[lons.ndim :], dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            data_full = np.nan * data_full
        elif np.issubdtype(dtype, np.datetime64):
            data_full[:] = np.datetime64("NaT")
        elif dtype == np.int8:
            data_full[:] = -1
        else:
            data_full[:] = -9999

        data_full[valid_pixels] = data_r
        resampled[var] = (new_dims + dataset[var].dims[lons.ndim :], data_full)

    return xr.Dataset(resampled)


def extract_collocations(
    track_data: pd.DataFrame,
    pansat_product,
    max_time_diff: np.timedelta64 = np.timedelta64(15, "m"),
):
    """
    Extract collocations for all features in a data frame containing storm
    tracks.

    Args:
        track_data: A pandas data frame contraining TOBAC tracks.
        pansat_product: The pansat product to collocate the tracks with.
        max_time_diff: Half width of the time interval center on the track
            time step in which collocations will be considered.
    """
    for feature, row in track_data.iterrows():
        lon = row.longitude
        lat = row.latitude
        time = row.time

        lon_min = max(lon - 1.0, -180.0)
        lon_max = min(lon + 1.0, 180.0)
        lat_min = max(lat - 1.0, -90.0)
        lat_max = min(lat + 1.0, 90.0)
        geom = LonLatRect(lon_min, lat_min, lon_max, lat_max)

        start_time = row.time - max_time_diff
        end_time = row.time + max_time_diff

        recs = pansat_product.get(TimeRange(start_time, end_time))
        index = get_index(pansat_product)
        granules = index.find(TimeRange(start_time, end_time), roi=geom)

        granules = merge_granules(granules)

        for granule in granules:
            if len(granules) > 0:
                LOGGER.info(
                    "Found match for feature %s at lon = %s, lat = %s, t = %s.",
                    feature,
                    lon,
                    lat,
                    time,
                )

                product_data = granule.open().rename(
                    {"latitude_ns": "latitude", "longitude_ns": "longitude"}
                )[{"frequencies": -1}]
                target_area = get_equal_area_grid(
                    lon, lat, resolution=5e3, extent=1000e3
                )
                resampled = resample_data(
                    product_data,
                    target_area,
                    new_dims=("y", "x"),
                    radius_of_influence=5e3,
                )

                time_str = to_datetime(row.time).strftime("%Y%m%d%H%M")
                filename = f"{row.cell}_{feature}_{time_str}.nc"
                resampled.to_netcdf(filename)
