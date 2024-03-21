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

# data paths                                                                                                    
data_path = Path('/glade/campaign/mmm/c3we/prein/observations/GPM_MERGIR/data/2020/')
savedir = Path('/glade/derecho/scratch/kukulies/ccic/tracking/tb/')

import sys
months =  sys.argv[1:]

# 4km horizontal grid spacing, half-hourly data                                                                 
dxy, dt  = 4000, 1800

# define thresholds 
conventional_threshold = 241 
optimal_threshold = 230
dc_threshold = 220

# parameters for feature detection                                                           
parameters_features = {}
parameters_features['threshold']=[conventional_threshold, dc_threshold] #thresholds for ice water path 
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
parameters_featues_optimal['threshold'] = [optimal_threshold, dc_threshold]
parameters_segmentation_optimal = parameters_segmentation_optimal.copy()
parameters_segmentation_optimal['threshold'] = optimal_threshold + 1 

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
        
        for day in days:
            # read in global data and relevant variables for one day
            fnames = list(data_path.glob( ('merg_2020'+ str(month).zfill(2) + str(day).zfill(2) + '*nc4')) )
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path                                                     
            tb = ds.Tb
            # interpolate nan values in Tb data 
            
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
            
            # run the segmentation on daily file  
            print(datetime.now(), f"Commencing segmentation  ", flush=True)
            mask, features = tobac.segmentation_2D(features_d, tb_iris, dxy, **parameters_segmentation)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features_threshold = tracks.set_index(features.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '.nc' ))
            features.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # segmentation for optimal threshold
            mask_opt, features_opt = tobac.segmentation_2D(features_opt, tb_iris, dxy, **parameters_segmentation_optimal)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features_threshold_opt = tracks.set_index(features_opt.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask_opt).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_opt.nc' ))
            features_opt.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))
            
        # Combine all feature dataframes for one month
        features = tobac.utils.combine_tobac_feats(feature_df_list)
        features_opt = tobac.utils.combine_tobac_feats(feature_df_list_opt)

        ### Perform tracking on monthly basis ###
        print(datetime.now(), f"Commencing tracking  ", flush=True)
        tracks = tobac.linking_trackpy(features, tb_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks = tracks[tracks.cell != -1]
        tracks_convective = tracks.groupby("cell").feature_max_tb.max()
        valid_cells = tracks_convective.index[tracks_convective >= dc_threshold]
        tracks = tracks[np.isin(tracks.cell, valid_cells)]
        tracks.to_xarray().to_netcdf(track_file)


        ### Perform tracking on monthly basis for optimal thresholds ###
        print(datetime.now(), f"Commencing tracking  ", flush=True)
        tracks_opt = tobac.linking_trackpy(features_opt, tb_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks_opt = tracks[tracks_opt.cell != -1]
        tracks_convective = tracks_opt.groupby("cell").feature_max_tb.max()
        valid_cells = tracks_convective.index[tracks_convective >= dc_threshold]
        tracks_opt = tracks[np.isin(tracks_opt.cell, valid_cells)]
        tracks_opt.to_xarray().to_netcdf(Path(savedir /  ('tracks_tb_'+ str(year) + str(month).zfill(2) +'_opt.nc')))


        
    # rerun the segmentation, but this time on the storm tracks instead of all features 
    if track_file.is_file() is True and segmentation_flag is True:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)
        tracks = xr.open_dataset(track_file).to_dataframe()
        tracks_threshold = []
        
        ### segmentation on daily basis only for tracked storm objects ###
        for day in days:
            # read in data and relevant variables for one day                                                   
            fnames = list(data_path.glob( ('merg_2020'+ str(month).zfill(2) + str(day).zfill(2) + '*nc4')) )
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path                                                     
            tb = ds.Tb
            # convert tracking field to iris                                                                    
            tb_iris = tb.to_iris()
            
            # segmentation threshold  
            print(datetime.now(), f"Commencing segmentation", flush=True)
            mask, tracks = tobac.segmentation_2D(tracks, tb_iris, dxy, **parameters_segmentation)    
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            tracks_threshold = tracks.set_index(tracks.feature).to_xarray() 

            # save daily mask and track files 
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))
            tracks_threshold.to_netcdf(savedir /  ('features_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # segmentation threshold for optimal threshold
            print(datetime.now(), f"Commencing segmentation", flush=True)
            mask_opt, tracks_opt = tobac.segmentation_2D(tracks_opt, tb_iris, dxy, **parameters_segmentation_optimal)    
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            tracks_threshold_opt = tracks_opt.set_index(tracks_opt.feature).to_xarray() 
            # save daily mask and track files 
            xr.DataArray.from_iris(mask_opt).to_netcdf(savedir / ('mask_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))
            tracks_threshold_opt.to_netcdf(savedir /  ('features_storm_tracks_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))
            print(datetime.now(), str("Processing finished for " + str(year)+ str(month).zfill(2)) + str(day).zfill(2) , flush=True)
            ds.close()
            tracks_threshold.close()
            tracks_threshold_opt.close()
    else:
        print(str(track_file) , '  already processed.', flush = True )
        continue










