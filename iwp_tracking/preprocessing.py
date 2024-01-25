"""
iwp_tracking.preprocessing
==========================

Functionality for the preprocessing of the data used in the IWP study.
"""
import warnings

import numpy as np
from scipy.signal import convolve
import xarray as xr

PRECIP_CLASSES = {
    0: "No precipitation",
    1: "Warm stratiform rain",
    3: "Snow",
    6: "Convection",
    7: "Hail",
    10: "Cool stratiform rain",
    91: "Tropical stratiform rain",
    96: "Tropical convective rain",
}


def get_smoothing_kernel(fwhm: float, grid_resolution: float) -> np.ndarray:
    """
    Calculate Gaussian smoothing kernel with a given full width at half
    maximum (FWHM).

    Args:
        fwhm: The full width at half maximum of the kernel.
        grid_resolution: The resolution of the underlying grid.

    Return:
        A numpy.ndarray containing the convolution kernel.
    """
    fwhm_n = fwhm / grid_resolution

    # Make sure width is uneven.
    width = int(fwhm_n * 3)
    x = np.arange(-(width // 2), width // 2 + 0.1)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x**2 + y**2)
    k = np.exp(np.log(0.5) * (2.0 * r / fwhm_n) ** 2)
    k /= k.sum()

    return k


def smooth_field(data: np.ndarray) -> np.ndarray:
    """
    Smooth field to 0.036 degree resolution.

    Args:
        data: The input MRMS array as 0.01 resolution.

    Return:
        A numpy.ndarray containing the smoothed arrary.
    """
    k = get_smoothing_kernel(0.036, 0.01)
    invalid = np.isnan(data)
    data_r = np.nan_to_num(data, nan=0.0, copy=True)

    # FFT-based convolution can produce negative values so remove
    # them here.
    data_s = np.maximum(convolve(data_r, k, mode="same"), 0.0)
    cts = np.maximum(convolve(~invalid, np.ones_like(k), mode="same"), 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_s = data_s / (cts / k.size)
    data_s[invalid] = np.nan
    return data_s


def resample_scalar(data: xr.DataArray, grid) -> xr.DataArray:
    """
    Resamples MRMS scalar data (such as surface precip or RQI) using the
    bucket resampler.

    Args:
        data: A xarray.DataArray containing the MRMS data
            to resample.
        grid: A pyresample.geometry.AreaDefinition defining the area to which
            to resample the data.

    Return:
        An xr.DataArray containing the resampled data.
    """
    lons, lats = grid.get_lonlats()
    lons = lons[0]
    lats = lats[:, 0]
    data_s = xr.DataArray(
        data=smooth_field(data.data),
        coords={"latitude": data.latitude, "longitude": data.longitude},
    )
    data_g = data_s.interp(latitude=lats, longitude=lons)
    return data_g


def resample_categorical(data: xr.DataArray, grid) -> xr.DataArray:
    """
    Resamples categorical data (such as the MRMS precip flag) using neares
    neighbor interpolation.

    Args:
        data: A xarray.DataArray containing the data
            to resample.
        grid: A pyresample.geometry.AreaDefinition defining the area to which
            to resample the data.

    Return:
        A xr.DataArray containing the resampled data.
    """
    lons, lats = grid.get_lonlats()
    lons = lons[0]
    lats = lats[:, 0]
    data_r = data.interp(
        latitude=lats,
        longitude=lons,
        method="nearest",
        kwargs={"fill_value": -3},
    )

    precip_class_repr = ""
    for value, name in PRECIP_CLASSES.items():
        precip_class_repr += f"{value} = {name}"
    data_r.attrs["classes"] = precip_class_repr

    return data_r
