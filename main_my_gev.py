"""Main file for GEV fitting of ERA5 data.

Adam Michael Bauer
UChicago
11.10.2025

To run: main_my_gev.py GRID
"""

import sys

import xarray as xr

from src.mle import ds_mle_fit

# import command line grid
GRID = sys.argv[1]

print("Importing land-masked data...")
# define variables and open datasets
vars = ['t2m_annual_max', 't2m_annual_min']
dss = [xr.open_dataset('data/ERA5/landonly/era5_' + VAR
                       + '_' + GRID + '_landonly.nc') for VAR in vars]

print("Doing GEV fits...")
# carry out GEV fitting for each dataset
dss_with_fit = [ds_mle_fit(ds, var_name='t2m') for ds in dss]
dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom') for ds in dss_with_fit]
    
print("GEV fitting complete.")

print("Saving datasets...")
# save datasets
for VAR, ds_masked in zip(vars, dss_with_fit_on_both):
    ds_masked.to_netcdf('data/ERA5/landonly/era5_' + VAR + '_' + GRID + '_landonly_gevfitted_mle.nc')

print("Datasets saved to data/ERA5/landonly/")