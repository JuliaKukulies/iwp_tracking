"""

This script derives MRMS precipitation rate and type distributions for each tracked feature.

"""
import calendar
import numpy as np 
from pathlib import Path
import xarray as xr 
import pandas as pd
from pathlib import Path 
path = Path('/glade/derecho/scratch/kukulies/ccic/tracking/')
from tobac.utils import get_statistics_from_mask
############################################# Function to subset CCIC and CPCIR to MRMS extent #######################################

from pyresample import create_area_def

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

def subset_cpcir_data(dataset: xr.Dataset, latname, lonname) -> xr.Dataset:
    """
            Subset CPCIR data to CONUS.

            Args:
                dataset: An xarray.Dataset to subset.

            Return:
                The dataset restricted to CONUS.
    """
    lons_cpcir, lats_cpcir = CPCIR_GRID.get_lonlats()
    lons_cpcir = lons_cpcir[0]
    lats_cpcir = -lats_cpcir[:, 0]

    col_start = np.where(lons_cpcir > mrms_lon_min)[0][0]
    col_end = np.where(lons_cpcir < mrms_lon_max)[0][-1]
    row_start = np.where(lats_cpcir > mrms_lat_min)[0][0]
    row_end = np.where(lats_cpcir < mrms_lat_max)[0][-1]

    return dataset[{latname: slice(row_start, row_end), lonname: slice(col_start, col_end)}]


def subset_ccic_data(dataset: xr.Dataset, latname, lonname) -> xr.Dataset:
    """
            Subset CPCIR data to CONUS.

            Args:
                dataset: An xarray.Dataset to subset.

            Return:
                The dataset restricted to CONUS.
    """
    lons_cpcir, lats_cpcir = CPCIR_GRID.get_lonlats()
    lons_cpcir = lons_cpcir[0]
    lats_cpcir = lats_cpcir[:, 0]

    col_start = np.where(lons_cpcir > mrms_lon_min)[0][0]
    col_end = np.where(lons_cpcir < mrms_lon_max)[0][-1]
    row_start = np.where(lats_cpcir < mrms_lat_max)[0][0]
    row_end = np.where(lats_cpcir > mrms_lat_min)[0][-1]
    return dataset[{latname: slice(row_start, row_end), lonname: slice(col_start, col_end)}
                                                ]


def hist_wrapper(data, bins ):
    '''
    Wrapper function to only return the counts and not the bins for histogram method.
    '''
    hist, *_ = np.histogram(data, bins )
    return hist

#################################################################################################################

# MRMS precipitation classes
precip_classes = np.array([-3, 0,1,3,6,7,10,91,96])

import sys
year = 2020
month =  str(sys.argv[1]).zfill(2)
nr_days = calendar.monthrange(int(year), int(month))[1]
days = np.arange(1, nr_days + 1)
subdir = Path('2020' + month)

#### Processing per day #####
for day in days:
    day = str(day).zfill(2)
    print('Start processing day ', day, flush = True )

    ### Read in MRMS files for day ###
    daily_files = list( (path / 'mrms' / subdir ).glob('mrms_2020' + month+ day + '*.nc'))
    daily_files.sort()
    mrms = xr.open_mfdataset(daily_files, combine = 'nested', concat_dim = 'time') 
    precip_rate = mrms.precip_rate.where(mrms.rqi >= 0.8 )
    precip_type = mrms.precip_flag.where(mrms.rqi >= 0.8 )

    ### Read in track and mask file ###
    mask_tb = xr.open_dataset(path / 'tb' / subdir /  ('mask_tb_2020' + month + day +'_241K.nc'       ))
    mask_iwp = xr.open_dataset(path / 'iwp' / subdir / ('mask_iwp_2020' + month + day + '_0.2kgm2.nc' ))
    tracks_tb = xr.open_dataset(path / 'tb' / subdir / ('tracks_tb_2020' + month + day + '_241K.nc'   )).to_dataframe()
    tracks_iwp = xr.open_dataset(path / 'iwp' / subdir / ('tracks_iwp_2020' +month+ day + '_0.2kgm2.nc'  )).to_dataframe()
    tracks_iwp['feature'] = tracks_iwp.index
    tracks_tb['feature'] = tracks_tb.index

    ### Crop CCIC/CPC IR file to MRMS ###
    mask_tb_conus = subset_cpcir_data(mask_tb, 'lat', 'lon')
    mask_iwp_conus = subset_ccic_data(mask_iwp, 'latitude', 'longitude')

    mask_iwp_conus['time'] = np.unique(precip_type.time)
    mask_tb_conus['time'] = np.unique(precip_type.time) 
    
    ### Calculate precip statistics ###
    tracks_iwp = get_statistics_from_mask(tracks_iwp, mask_iwp_conus.segmentation_mask, precip_type, statistic= {'hist_precip_type': (hist_wrapper, {'bins': precip_classes}) }) 
    tracks_iwp = get_statistics_from_mask(tracks_iwp, mask_iwp_conus.segmentation_mask, precip_rate, statistic= {'hist_precip_rate': (hist_wrapper, {'bins': np.logspace(-1,2)} ) })

    tracks_tb = get_statistics_from_mask(tracks_tb, mask_tb_conus.segmentation_mask, precip_type, statistic= {'hist_precip_type': (hist_wrapper, {'bins': precip_classes}) }) 
    tracks_tb = get_statistics_from_mask(tracks_tb, mask_tb_conus.segmentation_mask, precip_rate, statistic= {'hist_precip_rate': (hist_wrapper, {'bins': np.logspace(-1,2)} ) })

    # filter out only tracks over CONUS and in mask
    for feature in tracks_iwp.index:
        selection = tracks_iwp[tracks_iwp.index == feature].hist_precip_rate.values
        tracks_iwp.loc[feature,'conus_flag']  = selection[0][0][0] is not None
    tracks_iwp_conus = tracks_iwp[tracks_iwp.conus_flag == True]

    for feature in tracks_tb.index:
        selection = tracks_tb[tracks_tb.index == feature].hist_precip_rate.values
        tracks_tb.loc[feature,'conus_flag']  = selection[0][0][0] is not None
    tracks_tb_conus = tracks_tb[tracks_tb.conus_flag == True]

    print('Features over CONUS with precip stats for IWP:', tracks_iwp_conus.shape[0], flush = True)
    print('Features over CONUS with precip stats for Tb:', tracks_tb_conus.shape[0], flush = True)

    ### Save new feature output ###
    tracks_iwp_conus_xr = tracks_iwp_conus.to_xarray()
    tracks_tb_conus_xr = tracks_tb_conus.to_xarray()
    
    # fixes to get into NETCDF4 compatible format 
    hist = [arr.item()[0].astype(int) for arr in tracks_iwp_conus_xr.hist_precip_rate]
    tracks_iwp_conus_xr['hist_precip_rate'] = ( ('features', 'bins'),  np.stack(hist))
    hist2 = [arr.item()[0].astype(int) for arr in tracks_iwp_conus_xr.hist_precip_type]
    tracks_iwp_conus_xr['hist_precip_type'] = ( ('features', 'type_bins'),  np.stack(hist2)  )
    tracks_iwp_conus_xr['conus_flag'] = tracks_iwp_conus_xr.conus_flag.astype(int)
    tracks_iwp_conus_xr.to_netcdf( path / ('tracks_iwp_mrms_precip_stats_2020' + month  + day + '_0.2kgm2.nc')) 

    hist = [arr.item()[0].astype(int) for arr in tracks_tb_conus_xr.hist_precip_rate]
    tracks_tb_conus_xr['hist_precip_rate'] = ( ('features', 'bins'),  np.stack(hist))
    hist2 = [arr.item()[0].astype(int) for arr in tracks_tb_conus_xr.hist_precip_type]
    tracks_tb_conus_xr['hist_precip_type'] = ( ('features', 'type_bins'),  np.stack(hist2)  )
    tracks_tb_conus_xr['conus_flag'] = tracks_tb_conus_xr.conus_flag.astype(int)
    tracks_tb_conus_xr.to_netcdf( path / ('tracks_tb_mrms_precip_stats_2020' +  month +  day + '_241K.nc'))  
