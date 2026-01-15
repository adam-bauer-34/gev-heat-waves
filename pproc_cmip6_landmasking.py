"""Preprocess CMIP6 data.

This script will mask out the oceans in CMIP6, leaving only the land
surface for our analysis. Note we use the ERA5 land/sea mask for 
consistency with our ERA5-based analysis.

Adam Michael Bauer
UChicago
1.12.2026

To run: pproc_cmip6_landmasking.py MEM MAKE_CHECK_PLOTS

NOTE: MEM is the ensemble member string WITHOUT the f piece, e.g., it is
'r1i1p1' or some such. The code will try f=1, 2, 3, 4 in that order. I did
this mainly because i only care about f=1 (the default) or whatever the default
is for that model (sometimes f=2 or f=3 depending).
"""

import sys
import shutil

import xarray as xr
import numpy as np

from config import DATA_ROOT
from src.check_plots import plot_side_by_side
from src.preprocessing import make_regridded_land_mask

# import command line stuff
MEM, MAKE_CHECK_PLOTS = sys.argv[1:]
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
    print("🧑‍🍳 No regridded land mask found, making a new one...")
    make_regridded_land_mask(GRID='1deg')
    land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask_1deg.nc')

for var in vars:
    print('='*width)
    print("👺 Regridding and masking ", var, " data...")
    print('='*width)

    # make data path
    data_path = DATA_ROOT / 'CMIP6' / var

    # make list of all files
    files = [f for f in data_path.glob('*.nc')]

    # only keep non-landonly files
    fnames = [
        f
        for f in files
        if not f.stem.endswith("_landonly")
    ]

    # make list of all annual mean files (never land masked)
    fnames_mean = [f for f in data_path_mean.glob('*.nc')]

    # open datasets, regrid, and save
    for f, f_mean in zip(fnames, fnames_mean):
        fparts = f.stem.split('_')
        model_name = '_'.join(fparts[2:3])

        print('-'*width)
        print("🪛 Working on ", model_name)
        ds = xr.open_dataset(f)
        ds_mean = xr.open_dataset(f_mean)

        # select member
        candidate_fs = (1, 2, 3, 4)
        for c in candidate_fs:
            MEM_try = MEM + 'f{}'.format(c)

            try:
                ds = ds.sel(member_id=str(model_name) + '_' + MEM_try)
                ds_mean = ds_mean.sel(member_id=str(model_name) + '_' + MEM_try)
                break  # exit loop if successful

            except KeyError:
                if c == 4:
                    raise KeyError("⚠️ No matching member_id found for model ", model_name,
                                   " with MEMs ", MEM, "fX for f = (1, 2, 3, 4).")
                else:
                    continue

        # compute anomalies relative to the annual mean
        # if max, do data - mean, if min, do mean - data
        if var == 'tas_annual_max':
            da_anom = ds['tas'] - ds_mean['tas']
        else:
            da_anom = ds_mean['tas'] - ds['tas']

        # assign the anomaly data array to the dataset
        ds = ds.assign({'tas_anom': da_anom})

        # mask out ocean / non-land
        ds_masked = ds.where(land_mask['lsm'].data[0] > MASK_THRES, np.nan)
        ds_masked.attrs['selected_member'] = MEM_try  # store ensemble member

        # save to netCDF
        ds_masked.to_netcdf(
            f.with_name(
                f.stem + "_landonly" + f.suffix
            )
        )
        print('-'*width)
        print("✅ ", model_name, " land masking done and saved successfully.")

        # close datasets to save memory
        ds.close()
        ds_mean.close()
        ds_masked.close()
        da_anom.close()

        # make check plots if desired
        if MAKE_CHECK_PLOTS:
            plot_side_by_side(
                ds_masked['tas_anom'].sel(year=2000),
                ds['tas_anom'].sel(year=2000),
                titles=("Masked tas", "Original tas"),
                val_plotted='tas',
                save_figs=True,
                filename_args=['tas_landmask_check_' + var + '_' + MEM + '_' + model_name, 'png', 'figs']
                )
            print("📊 Check plot for ", model_name, " saved.")

print('='*width)
print("🥳 All done! 🥳")
print('='*width)