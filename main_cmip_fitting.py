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
    f = fnames[0]

    print(
        data_path / 'gev' / (f.stem + "_gev_{}".format(STAT) + f.suffix)
    )

    for f in fnames:
        fparts = f.stem.split('_')
        model_name = '_'.join(fparts[2:3])

        print('-'*width)
        print("🪛 Working on ", model_name)
        ds = xr.open_dataset(f)  # import land masked data

        # carry out GEV fits
        print('-'*width)
        print("🧮 Doing GEV fitting...")

        if STAT == 'stat':
            print("🥩 Doing GEV fits on raw temperature data...")
            ds_raw_fit = ds_mle_fit(ds, var_name='tas', fit_dim='year',
                                    non_stat=False)
            print("⚡️ Doing GEV fits on temperature anomalies...")
            ds_both_fit = ds_mle_fit(ds_raw_fit, var_name='tas_anom', fit_dim='year',
                                     non_stat=False)

        if STAT == 'nonstat':
            print("🥩 Doing GEV fits on raw temperature data...")
            ds_raw_fit = ds_mle_fit(ds, var_name='tas', fit_dim='year',
                                    non_stat=True)
            print("⚡️ Doing GEV fits on temperature anomalies...")
            ds_both_fit = ds_mle_fit(ds_raw_fit, var_name='tas_anom', fit_dim='year',
                                     non_stat=True)

        else:
            raise ValueError("⚠️ Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
    
        print("✅ GEV fitting complete.")

        # save datasets
        ds_both_fit.to_netcdf(
            f.with_name('gev' / 
                f.stem + "_gev_{}".format(STAT) + f.suffix
            )
        )

        print("✍️ Dataset successfully saved.")

print('='*width)
print("🥳 All done! 🥳")
print('='*width)