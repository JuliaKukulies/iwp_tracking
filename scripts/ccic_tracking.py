"""

Tracking convection based on total ice water path in CCIC database

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
data_path = Path('/glade/derecho/scratch/kukulies/ccic/ccic_2020/')
savedir = Path('/glade/derecho/scratch/kukulies/ccic/tracking/iwp/')

# perform feature detection, segmentation and tracking for a specific month(s) 
months =  sys.argv[1:] 

# 4km horizontal grid spacing, half-hourly data 
dxy, dt  = 4000, 1800

################################ parameters for feature detection ####################################################
optimal_threshold = 0.18 
dc_threshold = 1.0

# parameters for feature detection                                                           
parameters_features = {}
parameters_features['threshold']=[optimal_threshold] #thresholds for ice water path 
parameters_features['target']='maximum'
parameters_features['n_min_threshold']= 100 # minimum number of grid cells that need to be above specified thresholds 
parameters_features['statistic'] = {"feature_min_iwp": np.nanmin, 'feature_max_iwp': np.nanmax, 'feature_mean_iwp': np.nanmean}

# parameters for linking 
parameters_linking={}
parameters_linking['d_max']=10*dxy 
parameters_linking['stubs']= 4  # minimum number of timesteps a storm has to persist 
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['method_linking']= 'predict'

# parameters for segmentation 
parameters_segmentation = {}
parameters_segmentation['threshold']= optimal_threshold + 0.1 # kg/m2 used to define the extent of the cloud objects                  
parameters_segmentation['target'] = "maximum"
parameters_segmentation['statistic'] = {"object_min_iwp": np.nanmin, 'object_max_iwp': np.nanmax, 'object_mean_iwp': np.nanmean}

# for DC thresholds
parameters_segmentation_dc = parameters_segmentation.copy()
parameters_features_dc = parameters_features.copy()
parameters_segmentation_dc['threshold']= dc_threshold + 0.1
parameters_features_dc['threshold']= dc_threshold

############################# main tracking program ########################### #################################################
year = 2020
segmentation_flag = True 

for month in months:
    print('Starting tracking procedure for', str(year), str(month), flush = True)
    # check first if day has already been processed
    track_file = Path(savedir /  ('tracks_iwp_'+ str(year) + str(month).zfill(2) +'.nc'))
    # if no track file for this month exist yet, start the tracking procedure 
    if track_file.is_file() is False:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)
        feature_df_list = [] # list to store daily feature data frames
        features_df_list_dc = []
        
        for day in days:
            # read in global data and relevant variables for one day  
            fnames = list(data_path.glob( ('*2020'+ str(month).zfill(2) + str(day).zfill(2) + '*zarr')) )  
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path 
            tiwp = ds.tiwp           
            # convert tracking field to iris 
            iwp_iris = tiwp.to_iris()
            print(datetime.now(), f"Commencing feature detection IWP for day ", str(day),  flush=True)

            # run the feature detection on daily file 
            features_d=tobac.feature_detection_multithreshold(iwp_iris ,dxy, **parameters_features)
            # save daily features in list to use for a monthly tracking 
            feature_df_list.append(features_d)

            # run the segmentation on daily file  
            print(datetime.now(), f"Commencing segmentation IWP", flush=True)
            mask, features = tobac.segmentation_2D(features_d, iwp_iris, dxy, **parameters_segmentation)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features = features.set_index(features.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '.nc' ))
            features.to_netcdf(savedir /  ('features_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))


            # same for DC threshold
            features_dc=tobac.feature_detection_multithreshold(iwp_iris ,dxy, **parameters_features_dc)
            # save daily features in list to use for a monthly tracking 
            feature_df_list_dc.append(features_dc)
            mask_dc, features_dc = tobac.segmentation_2D(features_dc, iwp_iris, dxy, **parameters_segmentation_dc)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features_dc = features_dc.set_index(features_dc.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_dc.nc' ))
            features_dc.to_netcdf(savedir /  ('features_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))

        # Combine all feature dataframes for one month
        features = tobac.utils.combine_tobac_feats(feature_df_list)
        features_dc = tobac.utils.combine_tobac_feats(feature_df_list_dc)
        
        ### Perform tracking on monthly basis ###
        print(datetime.now(), f"Commencing tracking IWP", flush=True)
        tracks = tobac.linking_trackpy(features, iwp_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks = tracks[tracks.cell != -1]
        tracks.to_xarray().to_netcdf(track_file)

        tracks_dc = tobac.linking_trackpy(features_dc, iwp_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks_dc = tracks[tracks.cell != -1]
        tracks_dc.to_xarray().to_netcdf( Path(savedir / ('tracks_iwp_'+ str(year) + str(month).zfill(2) +'_dc.nc')))

    # rerun the segmentation, but this time on the storm tracks instead of all features 
    if track_file.is_file() is True and segmentation_flag is True:
        
        ### segmentation on daily basis only for tracked storm objects ###
        for day in days:
            # read in data and relevant variables for one day  
            fnames = list(data_path.glob( ('*2020'+ str(month).zfill(2) + str(day).zfill(2) + '*zarr')) )  
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path 
            tiwp = ds.tiwp           
            # convert tracking field to iris 
            iwp_iris = tiwp.to_iris()
            
            # segmentation IWP threshold  
            print(datetime.now(), f"Commencing segmentation IWP", flush=True)
            mask, tracks_day = tobac.segmentation_2D(tracks, iwp_iris, dxy, **parameters_segmentation)            
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            tracks_day= tracks_d.set_index(tracks.feature).to_xarray() 

            # save daily mask and track files 
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))
            tracks_day.to_netcdf(savedir /  ('features_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # same for DC threshold   
            mask_dc, tracks_dc_day = tobac.segmentation_2D(tracks_dc, iwp_iris, dxy, **parameters_segmentation_dc)
            tracks_dc_day = tracks_dc_day.set_index(tracks.feature).to_xarray() 
            # save daily mask and track files 
            xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))
            tracks_threshold_dc.to_netcdf(savedir /  ('features_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))
            print(datetime.now(), str("Processing finished for " + str(year)+ str(month).zfill(2)) + str(day).zfill(2) , flush=True)
            
            ds.close()
            tracks_day.close()
            tracks_dc_day.close()
    else:
        print(str(track_file) , '  already processed.', flush = True )
        continue










