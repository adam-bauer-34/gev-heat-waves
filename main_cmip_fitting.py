"""Main file for GEV fitting of CMIP data.

Adam Michael Bauer
UChicago
Jan 2026

To run: main_gev_fit.py STAT
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

    for f in fnames:
        fparts = f.stem.split('_')
        model_name = '_'.join(fparts[2:3])

        print('-'*width)
        print("🪛 Working on ", model_name)
        ds = xr.open_dataset(f)

        # carry out GEV fits
        print('-'*width)
        print("🧮 Doing GEV fits on raw data...")
        ds_fit = ds_mle_fit(ds, var_name='tas', fit_dim='year')

    # open dataset
    print('-'*width)
    print("🪏 Importing land-masked data...")
    dss = [xr.open_dataset(DATA_ROOT / 'CMIP6'/ var / f) for f in fnames]

    # carry out GEV fitting for each dataset
    if STAT == 'stat':
        print('-'*width)
        print("🧮 Doing stationary GEV fits...")
        #dss_with_fit = [ds_mle_fit(ds, var_name='tas') for ds in dss]
        #dss_with_fit_on_both = [ds_mle_fit(ds, var_name='tas_anom') for ds in dss_with_fit]

    elif STAT == 'nonstat':
        print('-'*width)
        print("🧮 Doing nonstationary GEV fits...")
        # dss_with_fit = [ds_mle_fit(ds, var_name='tas',
        #                         fit_dim='year', non_stat=True) for ds in dss]
        dss_with_fit_on_both = [ds_mle_fit(ds, var_name='tas_anom',
                                        fit_dim='year', non_stat=True) for ds in dss_with_fit]

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