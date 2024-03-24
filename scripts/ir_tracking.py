"""

Tracking convection based on IR brightness temperatures in the CPC-IR dataset.

---------------------------------
Email contact: kukulies@ucar.edu
---------------------------------

"""

import xarray as xr 
import numpy as np
import sys
import pandas as pd 
from pathlib import Path 
import tobac
from datetime import datetime
import calendar
import ccic 
import warnings
warnings.filterwarnings("ignore")

print('Using tobac version', tobac.__version__)


# data paths                                                                                                    
data_path = Path('/glade/campaign/mmm/c3we/prein/observations/GPM_MERGIR/data/2020/')
savedir = Path('/glade/derecho/scratch/kukulies/ccic/tracking/tb/')

import sys
months =  sys.argv[1:]

# 4km horizontal grid spacing, half-hourly data                                                                 
dxy, dt  = 4000, 1800

# parameters for linking 
parameters_linking={}
parameters_linking['d_max']=10*dxy 
parameters_linking['stubs']= 4  # minimum number of timesteps a storm has to persist 
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['method_linking']= 'predict'

# parameters for segmentation                                                                                                         
parameters_segmentation = {}
parameters_segmentation['threshold']= conventional_threshold + 0.1
parameters_segmentation['target'] = "minimum"
parameters_segmentation['statistic'] = {"object_min_tb": np.nanmin, 'object_max_tb': np.nanmax, 'object_mean_tb': np.nanmean}

# optimal thresholds                                                                                                                  
parameters_features_optimal = parameters_features.copy()
parameters_features_optimal['threshold'] = [optimal_threshold]
parameters_segmentation_optimal = parameters_segmentation.copy()
parameters_segmentation_optimal['threshold'] = optimal_threshold + 0.1

parameters_features_dc = parameters_features.copy()
parameters_features_dc['threshold'] = [dc_threshold]
parameters_segmentation_dc = parameters_segmentation.copy()
parameters_segmentation_dc['threshold'] = dc_threshold + 0.1

############################# main tracking program ##################################
year = 2020

for month in months:
    print('Starting tracking for', str(year), flush = True)
    # check first if month has already been processed
    track_file = Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) '.nc'))
    if track_file.is_file() is False:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)

        # read in all features within month
        files     = list(savedir.glob(('features_*' +  str(year) + str(month).zfill(2) + '.nc' )))
        files_opt = list(savedir.glob(('features_*' +  str(year) + str(month).zfill(2)  + '_opt.nc' )))
        files_dc  = list(savedir.glob(('features_*' +  str(year) + str(month).zfill(2)  + '_dc.nc' )))
        files.sort()
        files_opt.sort()
        files_dc.sort()

        features_df_list     = []
        features_df_list_opt = []
        features_df_list_dc  = []

        for idx in np.arange(len(files)):
            features_df_list.append( xr.open_dataset(files[idx]).to_dataframe() ) 
            features_df_list_opt.append(xr.open_dataset(files_opt[idx]).to_dataframe() )
            features_df_list_dc.append(xr.open_dataset(files_dc[idx]).to_dataframe() )
            
        # Combine all feature dataframe for one month 
        features = tobac.utils.combine_tobac_feats(feature_df_list)
        features_opt = tobac.utils.combine_tobac_feats(feature_df_list_opt)
        features_dc = tobac.utils.combine_tobac_feats(feature_df_list_dc)
        
        ### Perform tracking on monthly basis ###
        print(datetime.now(), f"Commencing tracking  ", flush=True)
        tracks = tobac.linking_trackpy(features, tb_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks = tracks[tracks.cell != -1]
        tracks.to_xarray().to_netcdf(track_file)

        ### Perform tracking on monthly basis for optimal thresholds ###
        print(datetime.now(), f"Commencing tracking  ", flush=True)
        tracks_opt = tobac.linking_trackpy(features_opt, tb_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks_opt = tracks[tracks_opt.cell != -1]
        tracks_opt.to_xarray().to_netcdf(Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'_opt.nc')))

        ### Perform tracking on monthly basis for DC thresholds ###
        tracks_dc = tobac.linking_trackpy(features_dc, tb_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks_dc = tracks[tracks_dc.cell != -1]
        tracks_dc.to_xarray().to_netcdf(Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'_dc.nc')))

    # rerun the segmentations, but this time on the storm tracks instead of all features 
    if track_file.is_file() is True:
        # read in track file
        tracks     = xr.open_dataset(track_file).to_dataframe()
        tracks_opt = xr.open_dataset(Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'_opt.nc'))).to_dataframe()
        tracks_dc  = xr.open_dataset(Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'_dc.nc'))).to_dataframe()
        
        ### segmentation on daily basis only for tracked storm objects ###
        for day in days:
            # read in data and relevant variables for one day                                                   
            fnames = list(data_path.glob( ('merg_2020'+ str(month).zfill(2) + str(day).zfill(2) + '*nc4')) )
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path                                                     
            tb = ds.Tb
            # interpolate nan values
            rechunked = ds.Tb.chunk(dict(lat=-1))
            tb  = rechunked.interpolate_na(dim='lat', method='linear') 
            # convert tracking field to iris                                                                    
            tb_iris = tb.to_iris()
            
            # segmentation 
            print(datetime.now(), f"Commencing segmentation", flush=True)
            mask, tracks_day = tobac.segmentation_2D(tracks, tb_iris, dxy, **parameters_segmentation)    
            tracks_day = tracks_day.set_index(tracks_day.feature).to_xarray() 
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))
            tracks_day.to_netcdf(savedir /  ('features_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # segmentation with optimal threshold
            mask_opt, tracks_opt_day = tobac.segmentation_2D(tracks_opt, tb_iris, dxy, **parameters_segmentation_optimal)    
            tracks_opt_day = tracks_opt_day.set_index(tracks_opt_day.feature).to_xarray() 
            xr.DataArray.from_iris(mask_opt).to_netcdf(savedir / ('mask_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))
            tracks_opt_day.to_netcdf(savedir /  ('features_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))

            # segmentation threshold for DC threshold
            mask_dc, tracks_dc_day = tobac.segmentation_2D(tracks_dc, tb_iris, dxy, **parameters_segmentation_dc)    
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            tracks_dc_day = tracks_dc_day.set_index(tracks_dc_day.feature).to_xarray() 
            # save daily mask and track files 
            xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))
            tracks_opt_day.to_netcdf(savedir /  ('features_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))

            print(datetime.now(), str("Processing finished for " + str(year)+ str(month).zfill(2)) + str(day).zfill(2) , flush=True)
            ds.close()
            tracks_day.close()
            tracks_opt_day.close()
            tracks_dc_day.close()                                     
    else:
        print(str(track_file) , '  already processed.', flush = True)
        continue










