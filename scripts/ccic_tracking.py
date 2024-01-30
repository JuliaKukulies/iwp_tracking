"""
Tracking convection based on total ice water path in CCIC database

---------------------------------
Email contact: kukulies@ucar.edu
---------------------------------

"""
import xarray as xr 
import numpy as np 
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
savedir = Path('/glade/derecho/scratch/kukulies/ccic/')

import sys
months =  sys.argv[1:] 

# 4km horizontal grid spacing, half-hourly data 
dxy, dt  = 4000, 1800

################################ parameters for feature detection ####################################################

# parameters for feature detection                                                           
parameters_features = {}
parameters_features['threshold']=[0.1, 0.2, 0.5, 1, 10 ] # multiple thresholds for ice water path 
parameters_features['target']='maximum'
parameters_features['n_min_threshold']= 100 # minimum number of grid cells that need to be above specified thresholds 
parameters_features['statistic'] = {"feature_min_iwp": np.nanmin, 'feature_max_iwp': np.nanmax, 'feature_mean_iwp': np.nanmean}

# parameters for linking 
parameters_linking={}
parameters_linking['v_max']=1e2
parameters_linking['stubs']= 6  # minimum number of timesteps a storm has to persist 
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['method_linking']= 'predict'

# parameters for segmentation 
parameters_segmentation = {}
parameters_segmentation['threshold']= 0.2 # kg/m2 used to define the extent of the cloud objects                  
parameters_segmentation['target'] = "maximum"
parameters_segmentation['statistic'] = {"object_min_iwp": np.nanmin, 'object_max_iwp': np.nanmax, 'object_mean_iwp': np.nanmean}

# segmentation with second threshold
parameters_segmentation_convective = parameters_segmentation.copy()
parameters_segmentation_convective['threshold']= 1 # kg/m2 used to define the extent of the cloud objects                     

############################# main tracking program ########################### #################################################
year = 2020

for month in months:
    print('Starting tracking procedure for', str(year), str(month), flush = True)
    # check first if day has already been processed
    track_file = Path(savedir /  ('monthly_tracks_iwp_'+ str(year) + str(month).zfill(2) +'.nc'))
    if track_file.is_file() is False:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)
        feature_df_list = [] # list to store daily feature data frames 
        for day in days:
            # read in data and relevant variables for one day  
            fnames = list(data_path.glob( ('*2020'+ str(month).zfill(2) + str(day).zfill(2) + '*zarr')) )  
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path 
            tiwp = ds.tiwp           
            # convert tracking field to iris 
            iwp_iris = tiwp.to_iris()
            print(datetime.now(), f"Commencing feature detection IWP for day ", str(day),  flush=True)
            
            # feature detection IWP on daily file 
            features_d=tobac.feature_detection_multithreshold(iwp_iris ,dxy, **parameters_features)
            feature_df_list.append(features_d)

        # Combine all feature dataframes for one month
        features = tobac.utils.combine_tobac_feats(feature_df_list)

        ### Perform tracking on monthly basis  
        print(datetime.now(), f"Commencing tracking IWP", flush=True)
        tracks = tobac.linking_trackpy(features, iwp_iris, dt, dxy, **parameters_linking)
        tracks = tracks[tracks.cell != -1]

        # reduce tracks to valid cells and those cells that contain some convection
        tracks = tracks[tracks.cell != -1]
        tracks_convective = tracks.groupby("cell").feature_max_iwp.max()
        valid_cells = tracks_convective.index[tracks_convective >= 1]
        tracks = tracks[np.isin(tracks.cell, valid_cells)]

        tracks.to_xarray().to_netcdf(track_file)

        
    if track_file.is_file() is True:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)
        
        tracks = xr.open_dataset(track_file).to_dataframe()
        tracks_threshold1 = []
        tracks_threshold2 = []

        ### segmentation of the tracked features on daily basis ###
        for day in days:
            # read in data and relevant variables for one day  
            fnames = list(data_path.glob( ('*2020'+ str(month).zfill(2) + str(day).zfill(2) + '*zarr')) )  
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path 
            tiwp = ds.tiwp           
            # convert tracking field to iris 
            iwp_iris = tiwp.to_iris()
        
            # segmentation IWP threshold 1 
            print(datetime.now(), f"Commencing segmentation IWP", flush=True)
            mask1, tracks1 = tobac.segmentation_2D(tracks, iwp_iris, dxy, **parameters_segmentation)
            
            # segmentation IWP threshold 2 
            print(datetime.now(), f"Commencing segmentation IWP", flush=True)
            mask2, tracks2 = tobac.segmentation_2D(tracks, iwp_iris, dxy, **parameters_segmentation_convective)

            # convert output tracks to xarray and save them to daily tracks for specific segmentation 
            tracks_threshold1 = tracks1.set_index(tracks1.feature).to_xarray() 
            tracks_threshold2 = tracks2.set_index(tracks2.feature).to_xarray()
            
            # save daily mask file 
            xr.DataArray.from_iris(mask1).to_netcdf(savedir / ('mask_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_0.2kgm2.nc' ))
            xr.DataArray.from_iris(mask2).to_netcdf(savedir / ('mask_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_1kgm2.nc' ))

            # save monthly track files
            #out_ds1 = tobac.utils.combine_tobac_feats(tracks_threshold1, old_feature_column_name='featureID_mask')
            #out_ds2 = tobac.utils.combine_tobac_feats(tracks_threshold2, old_feature_column_name= 'featureID_mask')

            tracks_threshold1.to_netcdf(savedir /  ('tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_0.2kgm2.nc'))
            tracks_threshold2.to_netcdf(savedir /  ('tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_1kgm2.nc'))       
            print(datetime.now(), str("Processing finished for " + str(year)+ str(month).zfill(2)) + str(day).zfill(2) , flush=True)

            ds.close()
            tracks_threshold1.close()
            tracks_threshold2.close()
            
    else:
        print(str(track_file) , '  already processed.', flush = True )
        continue










