"""Preprocess CMIP6 data.

This script will mask out the oceans in CMIP6, leaving only the land
surface for our analysis. Note we use the ERA5 land/sea mask for 
consistency with our ERA5-based analysis.

Adam Michael Bauer
UChicago
1.12.2026

To run: pproc_cmip6_landmasking.py VAR MEM MAKE_CHECK_PLOTS
"""

import sys
import shutil

import xarray as xr
import xesmf as xe
import numpy as np

from config import DATA_ROOT
from src.check_plots import plot_side_by_side
from src.preprocessing import make_regridded_land_mask
from src.utils import check_lat_lon_grids_consistent

# import command line stuff
VAR, MEM, MAKE_CHECK_PLOTS = sys.argv[1:]
MAKE_CHECK_PLOTS = bool(int(MAKE_CHECK_PLOTS))
width = shutil.get_terminal_size(fallback=(80, 20)).columns

# set mask threshold
MASK_THRES = 0.5  # "standard", according to Rahul Singh :)

# make annual mean data path for reference
data_path_mean = DATA_ROOT / 'CMIP6' / 'tas_annual_mean'

# define variables
vars = ['tas_annual_max', 'tas_annual_min']

# import land mask, or if it's not there, make it with CMIP grid (1 deg x 1 deg)
try:
    land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask_1deg.nc')

except FileNotFoundError:
    print("No land mask found, making a new one...")
    make_regridded_land_mask(GRID='1deg')
    land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask_1deg.nc')


for var in vars:
    print('='*width)
    print("◾️ ➡️ ⬛️ & 👺 Regridding and masking ", var, " data...")
    print('='*width)

    # make data path
    data_path = DATA_ROOT / 'CMIP6' / var

    # make list of all files
    fnames = [f for f in data_path.glob('*.nc')]
    fnames_mean = [f for f in data_path_mean.glob('*.nc')]

    # open datasets, regrid, and save
    for f, f_mean in zip(fnames, fnames_mean):
        fparts = f.replace(".nc", "").split('_')
        model_name = '_'.join(fparts[1:3])

        print('-'*width)
        print("💽 Working on ", model_name)
        ds = xr.open_dataset(f)
        ds_mean = xr.open_dataset(f_mean)

        # select member
        ds = ds.sel(member_id=MEM)
        ds_mean = ds_mean.sel(member_id=MEM)

        # compute anomalies relative to the annual mean
        da_anom = ds[var] - ds_mean[var]

        ds = ds.assign({'tas_anom': da_anom})

        # mask out ocean / non-land
        ds_masked = ds.where(land_mask['lsm'].data[0] > MASK_THRES, np.nan)

        ds_masked.to_netcdf(f + '_landonly.nc')
        print('-'*width)
        print("✅ ", model_name, " done.")

        ds.close()
        ds_mean.close()
        ds_masked.close()
        da_anom.close()

        if MAKE_CHECK_PLOTS:
            plot_side_by_side(
                ds_masked['tas'].sel(year=2000),
                ds['tas'].sel(year=2000),
                titles=("Masked tas", "Original tas"),
                val_plotted='tas',
                save_figs=True,
                filename_args=['tas_landmask_check_' + var + '_' + MEM + '_' + model_name, 'png', 'figs']
                )
            print('-'*width)
            print("📊 Check plot for ", model_name, " saved.")