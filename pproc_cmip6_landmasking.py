"""Preprocess CMIP6 data.

This script will mask out the oceans in CMIP6, leaving only the land
surface for our analysis. Note we use the ERA5 land/sea mask for 
consistency with our ERA5-based analysis.

Adam Michael Bauer
UChicago
1.12.2026

To run: pproc_cmip6_landmasking.py MAKE_CHECK_PLOTS
"""

import sys
import shutil

import xarray as xr
import numpy as np

from config import DATA_ROOT
from src.check_plots import plot_side_by_side
from src.preprocessing import make_regridded_land_mask
from src.cmip_dataclass import CMIP6EnsembleConfig

# import command line stuff
MAKE_CHECK_PLOTS = bool(int(sys.argv[1]))
width = shutil.get_terminal_size(fallback=(80, 20)).columns

# make CMIP6 model config class (only used for plotting here)
CMIPConfig = CMIP6EnsembleConfig.from_yaml("config.meta.yaml", 
                                            "config/qc.yaml")

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

        # 1. compute anomalies relative to annual mean temperature
        da_anom_annmean = ds['tas'] - ds_mean['tas']

        # assign the anomaly data array to the dataset
        ds = ds.assign({'t2m_anom_annmean': da_anom_annmean})

        # 2. compute anomalies relative to *trend* in annual mean temperature
        # do linear regression on annual mean data
        annual_mean_trend = ds_mean.polyfit(dim='year', deg=1, skipna=True)

        # make time series of temperature values given by trendline 
        t2m_annual_mean_trend = annual_mean_trend.t2m_polyfit_coefficients.sel(degree=0)\
            + ds_mean.year * annual_mean_trend.t2m_polyfit_coefficients.sel(degree=1)

        # subtract trendline temperatures from annual max / min to get anomalies relative
        # to the trend
        da_anom_trend = ds['tas'] - t2m_annual_mean_trend

        # assign to datasets
        ds = ds.assign({'t2m_anom_trend': da_anom_trend})

        # mask out ocean / non-land
        ds_masked = ds.where(land_mask['lsm'].data[0] > MASK_THRES, np.nan)

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
        da_anom_annmean.close()
        da_anom_trend.close()

        # make check plots if desired
        if MAKE_CHECK_PLOTS:
            print("-"*width)
            print("📈 Making check plot...")

            plot_side_by_side(
                ds_masked['tas'].sel(year=2000,
                                     member_id=CMIPConfig.ensemble_config[model_name].primary_member),
                ds['tas'].sel(year=2000,
                              member_id=CMIPConfig.ensemble_config[model_name].primary_member),
                titles=("Masked tas", "Original tas"),
                save_figs=True,
                filename_args=['tas_landmask_check_' + var + '_' + model_name, 'png', 'figs/checks']
                )
            print("📊 Check plot for ", model_name, " saved.")

print('='*width)
print("🥳 All done! 🥳")
print('='*width)