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
optimal_threshold = 0.2 
dc_threshold = 2.0

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

for month in months:
    print('Starting tracking procedure for', str(year), str(month), flush = True)
    nr_days = calendar.monthrange(int(year), int(month))[1]
    days = np.arange(1, nr_days + 1)

    # if no track file for this month exist yet, start the tracking procedure 
    for day in days:
        #check first if day has already been processed
        output_file = Path(savedir / ('tracks_iwp_'+ str(year) + str(month).zfill(2) +'.nc'))
        if output_file.is_file() is False:
            # read in global data and relevant variables for one day  
            fnames = list(data_path.glob(('*2020'+ str(month).zfill(2) + str(day).zfill(2) + '*zarr')))  
            ds = xr.open_mfdataset(fnames)
            # field used for tracking: total ice water path 
            tiwp = ds.tiwp           
            # convert tracking field to iris 
            iwp_iris = tiwp.to_iris()
            print(datetime.now(), f"Commencing feature detection IWP for day ", str(day),  flush=True)

            # run the feature detection on daily file 
            features_day=tobac.feature_detection_multithreshold(iwp_iris ,dxy, **parameters_features)
            # run the segmentation on daily file  
            print(datetime.now(), f"Commencing segmentation IWP", flush=True)
            mask, features = tobac.segmentation_2D(features_day, iwp_iris, dxy, **parameters_segmentation) 
            features = features.set_index(features.feature).to_xarray()
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '.nc' ))
            features.to_netcdf(savedir /  ('features_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # same for DC threshold 
            features_dc=tobac.feature_detection_multithreshold(iwp_iris ,dxy, **parameters_features_dc) 
            mask_dc, features_dc = tobac.segmentation_2D(features_dc, iwp_iris, dxy, **parameters_segmentation_dc)
            features_dc = features_dc.set_index(features_dc.feature).to_xarray()
            xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_dc.nc' ))
            features_dc.to_netcdf(savedir /  ('features_iwp_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))
    else:
        print(str(output_file) , '  already processed.', flush = True)
        continue










