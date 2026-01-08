"""Main file for GEV fitting of CMIP data.

Adam Michael Bauer
UChicago
Jan 2026

To run: main_gev_fit.py GRID STAT
"""

import os
import sys
import shutil

import xarray as xr

from config import DATA_ROOT
from src.mle import ds_mle_fit

# import command line grid
STAT, = sys.argv[1:]
width = shutil.get_terminal_size(fallback=(80, 20)).columns

# define variables and open datasets
vars = ['tas_annual_max', 'tas_annual_min']

for var in vars:
    print('='*width)
    print("🏋🏼‍♀️ Carrying out GEV fits for: ", var)
    print('='*width)
    
    # make data directory if they don't exist
    print("🧐 Making a {} directory if it doesn't exist...".format(DATA_ROOT / 'CMIP6' / var / 'gev'))
    os.makedirs(DATA_ROOT / 'CMIP6' / var / 'gev', exist_ok=True)
    data_path = DATA_ROOT / 'CMIP6' / var

    # make all landonly file names
    fnames = [f for f in data_path.glob("*_landonly.nc")]

    # open dataset
    print('-'*width)
    print("🪏 Importing land-masked data...")
    # dss = [xr.open_dataset(DATA_ROOT / 'ERA5' /'landonly' / fname) for fname in fnames]

    # carry out GEV fitting for each dataset
    if STAT == 'stat':
        print('-'*width)
        print("🧮 Doing stationary GEV fits...")
        #dss_with_fit = [ds_mle_fit(ds, var_name='t2m') for ds in dss]
        #dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom') for ds in dss_with_fit]

    elif STAT == 'nonstat':
        print('-'*width)
        print("🧮 Doing nonstationary GEV fits...")
        #dss_with_fit = [ds_mle_fit(ds, var_name='t2m',
        #                        fit_dim='year', non_stat=True) for ds in dss]
        #dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom',
        #                                fit_dim='year', non_stat=True) for ds in dss_with_fit]

    else:
        raise ValueError("Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
    
    print('-'*width)
    print("✅ GEV fitting complete.")

    print('-'*width)
    print("✍️ Saving datasets...")
    # save datasets
    #for VAR, ds_masked, fname in zip(vars, dss_with_fit_on_both, fnames):
    #    ds_masked.to_netcdf(DATA_ROOT / 'CMIP6' / VAR / 'gev' / (fname[:-3] + '_gev_' + STAT + '.nc'))
    
    print('-'*width)
    print("✅ Datasets saved to f{DATA_ROOT}/ERA5/landonly/")