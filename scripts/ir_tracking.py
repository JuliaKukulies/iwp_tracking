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

# define thresholds 
conventional_threshold = 241 
optimal_threshold = 256
dc_threshold = 220

# parameters for feature detection                                                           
parameters_features = {}
parameters_features['threshold']=[conventional_threshold] #thresholds for ice water path 
parameters_features['target']='minimum'
parameters_features['n_min_threshold']= 100 # minimum number of grid cells that need to be above specified thresholds 
parameters_features['statistic'] = {"feature_min_tb": np.nanmin, 'feature_max_tb': np.nanmax, 'feature_mean_tb': np.nanmean}

# parameters for linking 
parameters_linking={}
parameters_linking['d_max']=10*dxy 
parameters_linking['stubs']= 4  # minimum number of timesteps a storm has to persist 
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['method_linking']= 'predict'

# parameters for segmentation 
parameters_segmentation = {}
parameters_segmentation['threshold']= conventional_threshold + 1                  
parameters_segmentation['target'] = "minimum"
parameters_segmentation['statistic'] = {"object_min_tb": np.nanmin, 'object_max_tb': np.nanmax, 'object_mean_tb': np.nanmean}

# optimal thresholds 
parameters_features_optimal = parameters_features.copy()
parameters_features_optimal['threshold'] = [optimal_threshold]
parameters_segmentation_optimal = parameters_segmentation.copy()
parameters_segmentation_optimal['threshold'] = optimal_threshold + 1

parameters_features_dc = parameters_features.copy()
parameters_features_dc['threshold'] = [dc_threshold]
parameters_segmentation_dc = parameters_segmentation.copy()
parameters_segmentation_dc['threshold'] = dc_threshold + 1 

############################# main tracking program ########################### #################################################
year = 2020
segmentation_flag = True 

for month in months:
    print('Starting tracking procedure for', str(year), str(month), flush = True)
    # check first if day has already been processed
    track_file = Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'.nc'))
    # if no track file for this month exist yet, start the tracking procedure 

    if track_file.is_file() is False:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)
        feature_df_list = [] # list to store daily feature data frames
        feature_df_list_opt = []
        feature_df_list_dc = []
        
        for day in days:
            # read in global data and relevant variables for one day
            fnames = list(data_path.glob( ('merg_2020'+ str(month).zfill(2) + str(day).zfill(2) + '*nc4')) )
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path                                                     
            tb = ds.Tb
            # interpolate nan values in Tb data
            rechunked = ds.Tb.chunk(dict(lat=-1))
            tb = rechunked.interpolate_na(dim='lat', method='linear')
            
            # convert tracking field to iris                                                                    
            tb_iris = tb.to_iris()
            print(datetime.now(), f"Commencing feature detection Tb for day ", str(day),  flush=True)
            
            # run the feature detection on daily file 
            features_d = tobac.feature_detection_multithreshold(tb_iris ,dxy, **parameters_features)
            # save daily features in list to use for a monthly tracking 
            feature_df_list.append(features_d)
    
            # run the feature detection on daily file with optimal threshold 
            features_opt = tobac.feature_detection_multithreshold(tb_iris ,dxy, **parameters_features_optimal)
            # save daily features in list to use for a monthly tracking 
            feature_df_list_opt.append(features_opt)

            # run the feature detection on daily file with optimal threshold 
            features_dc = tobac.feature_detection_multithreshold(tb_iris ,dxy, **parameters_features_dc)
            # save daily features in list to use for a monthly tracking 
            feature_df_list_dc.append(features_dc)
            
            # run the segmentation on daily file  
            print(datetime.now(), f"Commencing segmentation  ", flush=True)
            mask, features = tobac.segmentation_2D(features_d, tb_iris, dxy, **parameters_segmentation)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features = features.set_index(features.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '.nc' ))
            features.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # segmentation for optimal threshold
            mask_opt, features_opt = tobac.segmentation_2D(features_opt, tb_iris, dxy, **parameters_segmentation_optimal)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features_opt = features_opt.set_index(features_opt.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask_opt).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_opt.nc' ))
            features_opt.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))


            # segmentation for optimal threshold
            mask_dc, features_dc = tobac.segmentation_2D(features_dc, tb_iris, dxy, **parameters_segmentation_dc)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features_dc = features_dc.set_index(features_dc.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_dc.nc' ))
            features_dc.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))

        # Combine all feature dataframes for one month
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
        print(datetime.now(), f"Commencing tracking  ", flush=True)
        tracks_dc = tobac.linking_trackpy(features_dc, tb_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks_dc = tracks[tracks_dc.cell != -1]
        tracks_dc.to_xarray().to_netcdf(Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'_dc.nc')))

    # rerun the segmentations, but this time on the storm tracks instead of all features 
    if track_file.is_file() is True and segmentation_flag is True:
        
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
            
            # segmentation threshold  
            print(datetime.now(), f"Commencing segmentation", flush=True)
            mask, tracks_day = tobac.segmentation_2D(tracks, tb_iris, dxy, **parameters_segmentation)    
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            tracks_day = tracks_day.set_index(tracks_day.feature).to_xarray() 
            # save daily mask and track files 
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))
            tracks_day.to_netcdf(savedir /  ('features_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # segmentation threshold for optimal threshold
            mask_opt, tracks_opt_day = tobac.segmentation_2D(tracks_opt, tb_iris, dxy, **parameters_segmentation_optimal)    
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            tracks_opt_day = tracks_opt_day.set_index(tracks_opt_day.feature).to_xarray() 
            # save daily mask and track files 
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










