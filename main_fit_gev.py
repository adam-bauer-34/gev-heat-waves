"""Main file for GEV fitting of ERA5 data.

Adam Michael Bauer
UChicago
11.10.2025

To run: main_fit_gev.py GRID STAT
"""

import sys

import xarray as xr

from src.mle import ds_mle_fit

# import command line grid
GRID = sys.argv[1]
STAT = sys.argv[2]

print("Importing land-masked data...")
# define variables and open datasets
vars = ['t2m_annual_max']
dss = [xr.open_dataset('data/ERA5/landonly/era5_' + VAR
                       + '_' + GRID + '_landonly.nc') for VAR in vars]

# carry out GEV fitting for each dataset
if STAT == 'stat':
    print("Doing stationary GEV fits...")
    dss_with_fit = [ds_mle_fit(ds, var_name='t2m') for ds in dss]
    dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom') for ds in dss_with_fit]

elif STAT == 'nonstat':
    print("Doing nonstationary GEV fits...")
    dss_with_fit = [ds_mle_fit(ds, var_name='t2m',
                               fit_dim='year', non_stat=True) for ds in dss]
    dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom',
                                       fit_dim='year', non_stat=True) for ds in dss_with_fit]

else:
    raise ValueError("Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
    
print("GEV fitting complete.")

print("Saving datasets...")
# save datasets
for VAR, ds_masked in zip(vars, dss_with_fit_on_both):
    ds_masked.to_netcdf('data/ERA5/landonly/era5_' + VAR + '_' + GRID + '_landonly_gev_'
                        + STAT + '.nc')

print("Datasets saved to data/ERA5/landonly/")