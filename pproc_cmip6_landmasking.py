"""Preprocess CMIP6 data.

This script will mask out the oceans in CMIP6, leaving only the land
surface for our analysis. Note we use the ERA5 land/sea mask for 
consistency with our ERA5-based analysis.

Adam Michael Bauer
UChicago
1.12.2026

To run: pproc_cmip6_landmasking.py MAKE_CHECK_PLOTS

Last edited: 1/28/2026, 12:30 PM CST
"""

import sys
import shutil

import xarray as xr
import numpy as np

from config import DATA_ROOT
from src.check_plots import plot_side_by_side
from src.preprocessing import make_regridded_land_mask
from src.cmip_dataclass import CMIP6EnsembleConfig
from src.utils import extract_model_name

# import command line stuff
MAKE_CHECK_PLOTS = bool(int(sys.argv[1]))
width = shutil.get_terminal_size(fallback=(80, 20)).columns

# make CMIP6 model config class (only used for plotting here)
CMIPConfig = CMIP6EnsembleConfig.from_yaml("config/meta.yaml", 
                                            "config/qc.yaml")

# set mask threshold
MASK_THRES = 0.5  # "standard", according to Rahul Singh :)

# make annual mean data path for reference
data_path_mean = DATA_ROOT / 'CMIP6' / 'tas_annual_mean' / 'raw'

# define variables
vars = ['tas_annual_max', 'tas_annual_min']

# import land mask, or if it's not there, make it with CMIP grid (1 deg x 1 deg)
try:
    land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask_1deg.nc')

except FileNotFoundError:
    print("\n🧑‍🍳 No regridded land mask found, making a new one...")
    make_regridded_land_mask(GRID='1deg')
    land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask_1deg.nc')

for var in vars:
    print('='*width)
    print("👺 Regridding and masking ", var, " data...")
    print('='*width)

    # make data path
    data_path = DATA_ROOT / 'CMIP6' / var / 'raw'

    # make list of all files
    fnames = [f for f in data_path.glob('*.nc')]

    # make list of all annual mean files (never land masked)
    fnames_mean = [f for f in data_path_mean.glob('*.nc')]

    # extract model names
    var_models = {extract_model_name(f) for f in fnames}
    mean_models = {extract_model_name(f) for f in fnames_mean}

    # check if all models that are available for this variable have corresponding
    # annual mean data.
    if var_models != mean_models:
        missing_in_mean = var_models - mean_models
        missing_in_var = mean_models - var_models

        print("⚠️ Not all CMIP output have complete data records for analysis.")
        if missing_in_mean:
            print(f"Models that have {var} data but not annual mean data: {missing_in_mean}.")
        if missing_in_var:
            print(f"Models that have annual mean data but not {var} data: {missing_in_var}.")
        
        error_message = ("CMIP ensemble data is incomplete."
                         "Either archive models that are missing data,"
                         " or reprocess model output to complete records.")
        raise ValueError(error_message)

    # Sort both lists by model name to ensure they match up correctly
    fnames = sorted(fnames, key=extract_model_name)
    fnames_mean = sorted(fnames_mean, key=extract_model_name)

    # open datasets, regrid, and save
    for f, f_mean in zip(fnames, fnames_mean):
        fparts = f.stem.split('_')
        model_name = '_'.join(fparts[2:3])  # hard coded, would have to change if we had diff data

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
        t2m_annual_mean_trend = annual_mean_trend.tas_polyfit_coefficients.sel(degree=0)\
            + ds_mean.year * annual_mean_trend.tas_polyfit_coefficients.sel(degree=1)

        # subtract trendline temperatures from annual max / min to get anomalies relative
        # to the trend
        da_anom_trend = ds['tas'] - t2m_annual_mean_trend

        # assign to datasets
        ds = ds.assign({'t2m_anom_trend': da_anom_trend})

        # mask out ocean / non-land
        ds_masked = ds.where(land_mask['lsm'].data[0] > MASK_THRES, np.nan)

        # save to netCDF
        land_dir = f.parent.parent / 'landonly'  # get land only directory path
        land_name = f.with_name(
            f.stem + "_landonly" + f.suffix
        ).name  # make name of file with _landonly appended on end
        ds_masked.to_netcdf(land_dir / land_name)  # save to netCDF file

        print('-'*width)
        print("✅ ", model_name, " land masking done and saved successfully to:\n{}".format(land_dir / land_name))

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