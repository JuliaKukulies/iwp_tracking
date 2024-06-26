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
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")

# data paths
data_path = Path('/glade/derecho/scratch/kukulies/ccic/ccic_2020/')
savedir = Path('/glade/derecho/scratch/kukulies/ccic/tracking/iwp/tracks/')

# perform feature detection, segmentation and tracking for a specific month(s) 
months =  sys.argv[1:] 

# 4km horizontal grid spacing, half-hourly data 
dxy, dt  = 4000, 1800

################################ parameters for feature detection ####################################################
optimal_threshold = 0.24 
dc_threshold = 2.6

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

################################################# main tracking program #####################################################################
year = 2020
# compression options 
encoding = {"segmentation_mask": {'zlib': True,'dtype':'int32' }}

for month in months:
    track_file = Path(savedir /  ('tracks_iwp_'+ str(year) + str(month).zfill(2) +'.nc'))
    if track_file.is_file() is False:
        fnames = list(data_path.glob( ('*2020'+ str(month).zfill(2) +'*zarr')) )
        fnames.sort()
        ds = xr.open_dataset(fnames[0])
        for filename in fnames:
            ds = xr.concat([ds, xr.open_dataset(filename)], dim = 'time')
        #ds = xr.open_mfdataset(fnames[0]) 
        tiwp = ds.tiwp            
        iwp_iris = tiwp.to_iris()

        print('Starting tracking for', str(year), flush = True)
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1)
        # read in all features within month                                                                                                                  
        files     = list(savedir.glob(('features_iwp*' +  str(year) + str(month).zfill(2) + '??.nc' )))
        files_dc  = list(savedir.glob(('features_iwp*' +  str(year) + str(month).zfill(2)  + '*_dc.nc' )))
        files.sort()
        files_dc.sort()

        features_df_list     = []
        features_df_list_dc  = []
        print(len(files), len(files_dc), flush = True)

        for idx in np.arange(len(files)):
            df = xr.open_dataset(files[idx]).to_dataframe()
            df['feature'] = df.index.values
            features_df_list.append(df)
            df = xr.open_dataset(files_dc[idx]).to_dataframe()
            df['feature'] = df.index.values            
            features_df_list_dc.append(df)

        # Combine all feature dataframe for one month  
        features = tobac.utils.combine_tobac_feats(features_df_list)
        features_dc = tobac.utils.combine_tobac_feats(features_df_list_dc)

        ### Perform tracking on monthly basis ###
        print(datetime.now(), f"Commencing tracking IWP", flush=True)
        tracks = tobac.linking_trackpy(features, iwp_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks = tracks[tracks.cell != -1]
        tracks.to_xarray().to_netcdf(track_file)

        tracks_dc = tobac.linking_trackpy(features_dc, iwp_iris, dt, dxy, **parameters_linking)
        # reduce tracks to valid cells and those cells that contain deep convection
        tracks_dc = tracks_dc[tracks_dc.cell != -1]
        tracks_dc.to_xarray().to_netcdf( Path(savedir / ('tracks_iwp_'+ str(year) + str(month).zfill(2) +'_dc.nc')))

    # rerun the segmentation, but this time on the storm tracks instead of all feature 
    if track_file.is_file() is True:
        nr_days = calendar.monthrange(int(year), int(month))[1]
        days = np.arange(1, nr_days + 1 )
        
        # read in track file                                 
        tracks     = xr.open_dataset(track_file).to_dataframe()
        tracks_dc  = xr.open_dataset(Path(savedir /  ('tracks_iwp_'+ str(year) + str(month).zfill(2) +'_dc.nc'))).to_dataframe()
        
        ### segmentation on daily basis only for tracked storm objects ###
        for day in days:
            print('getting CCIC for segmentation', str(day), flush = True)
            # check if this has alrady been done for the day                                                                           
            fname = Path(savedir / ('mask_storm_tracks_iwp_' +  str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))
            if fname.is_file() is False:
                # read in data and relevant variables for one day  
                fnames = list(data_path.glob( ('*2020'+ str(month).zfill(2) + str(day).zfill(2) + '*zarr')) )
                fnames.sort()
                ds = xr.open_dataset(fnames[0])
                for filename in tqdm(fnames[1:]):
                    ds = xr.concat([ds, xr.open_dataset(filename)], dim = 'time')
                #ds = xr.open_mfdataset(fnames)
                # field used for tracking: total ice water path 
                tiwp = ds.tiwp           
                # convert tracking field to iris 
                iwp_iris = tiwp.to_iris()

                # segmentation IWP threshold  
                print(datetime.now(), f"Commencing segmentation IWP", flush=True)
                input_tracks = tracks[tracks.time.dt.day == day]
                input_tracks['feature'] = input_tracks.index.values
                mask, tracks_day = tobac.segmentation_2D(input_tracks, iwp_iris, dxy, **parameters_segmentation)            
                tracks_day= tracks_day.set_index(tracks_day.feature).to_xarray() 
                xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'), encoding = encoding)
                tracks_day.to_netcdf(savedir /  ('features_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc' ))

                # same for DC threshold
                input_tracks_dc = tracks_dc[tracks_dc.time.dt.day == day]
                input_tracks_dc['feature'] = input_tracks_dc.index.values
                mask_dc, tracks_dc_day = tobac.segmentation_2D(input_tracks_dc, iwp_iris, dxy, **parameters_segmentation_dc)
                tracks_dc_day = tracks_dc_day.set_index(tracks_dc_day.feature).to_xarray() 
                xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'), encoding = encoding)
                tracks_dc_day.to_netcdf(savedir /  ('features_storm_tracks_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))
                print(datetime.now(), str("Processing finished for " + str(year)+ str(month).zfill(2)) + str(day).zfill(2) , flush=True)

                ds.close()
                tracks_day.close()
                tracks_dc_day.close()

            else:
                print(str(day), 'already processed.', flush = True)
                continue                
    else:
        print(str(track_file) , '  already processed.', flush = True )
        continue










