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

############################# main feature detection program #####################################################

year = 2020

for month in months:
    print('Starting feature detection and segmentation procedure for ', str(year), str(month), flush = True)
    nr_days = calendar.monthrange(int(year), int(month))[1]
    days = np.arange(1, nr_days + 1)
    # run feature detection for each day
    for day in days:
        # check first if day has already been processed
        output_file = Path(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))
        if output_file.is_file() is False:
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
            
            print(datetime.now(), f"Commencing feature detection Tb for day", str(day),  flush=True)
            # run the feature detection on daily file 
            features_day = tobac.feature_detection_multithreshold(tb_iris, dxy, **parameters_features)
            features_opt = tobac.feature_detection_multithreshold(tb_iris, dxy, **parameters_features_optimal)
            features_dc = tobac.feature_detection_multithreshold(tb_iris, dxy, **parameters_features_dc)
            
            # run the segmentation on daily file  
            print(datetime.now(), f"Commencing segmentation", flush=True)
            mask, features = tobac.segmentation_2D(features_day, tb_iris, dxy, **parameters_segmentation)
            # convert output tracks to xarray and save them to daily tracks for segmentation 
            features = features.set_index(features.feature).to_xarray()
            # save daily mask and feature files
            xr.DataArray.from_iris(mask).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '.nc' ))
            features.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc'))

            # segmentation for optimal threshold
            mask_opt, features_opt = tobac.segmentation_2D(features_opt, tb_iris, dxy, **parameters_segmentation_optimal)
            features_opt = features_opt.set_index(features_opt.feature).to_xarray()
            xr.DataArray.from_iris(mask_opt).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_opt.nc' ))
            features_opt.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_opt.nc'))

            # segmentation for deep convection threshold
            mask_dc, features_dc = tobac.segmentation_2D(features_dc, tb_iris, dxy, **parameters_segmentation_dc)
            features_dc = features_dc.set_index(features_dc.feature).to_xarray()
            xr.DataArray.from_iris(mask_dc).to_netcdf(savedir / ('mask_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) +  '_dc.nc' ))
            features_dc.to_netcdf(savedir /  ('features_tb_'+ str(year) + str(month).zfill(2) + str(day).zfill(2) + '_dc.nc'))

    else:
        print(str(output_file) , '  already processed.', flush = True)
        continue










