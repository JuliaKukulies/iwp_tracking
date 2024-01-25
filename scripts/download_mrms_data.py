import logging
from pathlib import Path

import click
import numpy as np
import xarray as xr

from pansat import TimeRange
from pansat.time import to_datetime
from pansat.products.ground_based import mrms

from iwp_tracking import CONUS_GRID
from iwp_tracking.preprocessing import resample_scalar, resample_categorical


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


@click.command()
@click.argument("start_time")
@click.argument("end_time")
@click.argument("output_folder")
def download_mrms_data(start_time: str, end_time: str, output_folder: str) -> None:
    """
    Download MRMS data for given time period.

    Args:
        start_time: The start time of the time period.
        end_time: The end time of the time period.
        output_folder: The folder to which to write the extracted data.
    """
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    output_folder = Path(output_folder)

    time = start_time

    while time <= end_time:
        LOGGER.info("Processing '%s'", time)

        mrms_precip_rate_rec = mrms.precip_rate.find_files(TimeRange(time, time))
        mrms_rqi_rec = mrms.radar_quality_index.find_files(TimeRange(time, time))
        mrms_precip_flag_rec = mrms.precip_flag.find_files(TimeRange(time, time))

        if (
            len(mrms_precip_rate_rec) == 0
            or len(mrms_precip_flag_rec) == 0
            or len(mrms_rqi_rec) == 0
        ):
            LOGGER.warning("Missing MRMS files for %s.", time)
        mrms_precip_rate_rec = mrms_precip_rate_rec[0].get()
        mrms_precip_rate = mrms.precip_rate.open(mrms_precip_rate_rec)
        mrms_precip_rate = resample_scalar(mrms_precip_rate.precip_rate, CONUS_GRID)

        mrms_rqi_rec = mrms_rqi_rec[0].get()
        mrms_rqi = mrms.radar_quality_index.open(mrms_rqi_rec)
        mrms_rqi = resample_scalar(mrms_rqi.radar_quality_index, CONUS_GRID)

        mrms_precip_flag_rec = mrms_precip_flag_rec[0].get()
        mrms_precip_flag = mrms.precip_flag.open(mrms_precip_flag_rec)
        mrms_precip_flag = resample_categorical(
            mrms_precip_flag.precip_flag, CONUS_GRID
        )

        date = to_datetime(time)
        date_str = date.strftime("%Y%m%d_%H_%M")

        filename = f"mrms_{date_str}.nc"
        results = xr.Dataset(
            {
                "precip_rate": mrms_precip_rate,
                "precip_flag": mrms_precip_flag,
                "rqi": mrms_rqi,
            }
        )

        encoding = {
            "precip_rate": {"dtype": "float32", "zlib": True},
            "precip_flag": {"dtype": "int8", "zlib": True},
        }
        results.to_netcdf(output_folder / filename, encoding=encoding)

        time += np.timedelta64(30, "m")


if __name__ == "__main__":
    download_mrms_data()
