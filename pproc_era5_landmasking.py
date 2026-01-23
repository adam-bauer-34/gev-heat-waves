"""Preprocess ERA5 data.

This script will mask out the oceans in ERA5, leaving only the land
surface for our analysis.

Adam Michael Bauer
UChicago
11.10.2025

To run: pproc_era5_landmasking.py GRID MAKE_CHECK_PLOTS
"""

import sys
import shutil
from pathlib import Path

import xarray as xr
import xesmf as xe
import numpy as np

from config import DATA_ROOT
from src.check_plots import plot_side_by_side

# import command line stuff
GRID, MAKE_CHECK_PLOTS = sys.argv[1:]
MAKE_CHECK_PLOTS = bool(int(MAKE_CHECK_PLOTS))
width = shutil.get_terminal_size(fallback=(80, 20)).columns

print('-'*width)
print("🔎 Importing data...")
# load in full ERA5 data
vars = ['t2m_annual_max', 't2m_annual_mean', 't2m_annual_min']
dss = [xr.open_dataset(DATA_ROOT / 'ERA5' / ('era5_' + var + '_' + GRID + '.nc')) for var in vars]

# COMPUTE ANOMALIES
print('-'*width)
print("🧮 Computing anomalies with respect to annual mean (removing climate change and interannual variability signal)...")
# 1. compute anomalies relative to annual mean for maximum and minimum datasets
da_t2m_max_anoms = dss[0]['t2m'] - dss[1]['t2m']  # annual max - annual mean
da_t2m_min_anoms = dss[2]['t2m'] - dss[1]['t2m']  # annual min - annual mean

# add anomalies to datasets
dss[0] = dss[0].assign({'t2m_anom_annmean': da_t2m_max_anoms})
dss[2] = dss[2].assign({'t2m_anom_annmean': da_t2m_min_anoms})

print('-'*width)
print("🧮 Computing anomalies with respect to trend in annual mean (removing climate change signal)...")
# 2. compute anomalies relative to *trend* in annual mean temperature
# do linear regression on annual mean data
annual_mean_trend = dss[1].polyfit(dim='year',
                                   deg=1, skipna=True)

# make time series of temperature values given by trendline 
t2m_annual_mean_trend = annual_mean_trend.t2m_polyfit_coefficients.sel(degree=0)\
    + dss[1].year * annual_mean_trend.t2m_polyfit_coefficients.sel(degree=1)

# subtract trendline temperatures from annual max / min to get anomalies relative
# to the trend
da_t2m_max_anoms_trend = dss[0]['t2m'] - t2m_annual_mean_trend
da_t2m_min_anoms_trend = dss[2]['t2m'] - t2m_annual_mean_trend

# assign to datasets
dss[0] = dss[0].assign({'t2m_anom_trend': da_t2m_max_anoms_trend})
dss[2] = dss[2].assign({'t2m_anom_trend': da_t2m_min_anoms_trend})

print('-'*width)
print("✅ Anomalies created successfully.")

# load in land/sea mask
land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask.nc')

# make masks for lat / lon ranges in ERA5 data to match land mask
lat_mask = (land_mask.latitude >= min(dss[0].lat)) & (land_mask.latitude <= max(dss[0].lat))
lon_mask = (land_mask.longitude >= min(dss[0].lon)) & (land_mask.longitude <= max(dss[0].lon))

# subselect land mask values that correspond to ERA5 values
land_mask = land_mask.sel(longitude=lon_mask, latitude=lat_mask).copy()

# make dataset for regridding the land mask
ds_output_grid = xr.Dataset(
    {
        'lat': (['lat'], dss[0]['lat'].values),
        'lon': (['lon'], dss[0]['lon'].values)
    }
)

print('-'*width)
print("👺 Making regridding object... (this could take a second)")
# initialize the regridder and regrid the land mask
regridder = xe.Regridder(land_mask, ds_output_grid, 'conservative')
land_mask_regridded = regridder(land_mask, keep_attrs=True)

print('-'*width)
print("⚒️ Applying land mask to dataset...")
# apply thea land mask to the dataset
MASK_THRES = 0.5  # threshold for me to consider something "land"
ds_maskeds = [ds.where(land_mask_regridded['lsm'].data[0] > MASK_THRES, np.nan) for ds in dss]

# create landonly directory if it doesn't exist
landonly_dir = DATA_ROOT / 'ERA5' / 'landonly'
landonly_dir.mkdir(parents=True, exist_ok=True)

# save datasets
for VAR, ds_masked in zip(vars, ds_maskeds):
    ds_masked.to_netcdf(landonly_dir / ('era5_' + VAR + '_' + GRID + '_landonly.nc'))

print('-'*width)
print("✅ Complete.")

print('-'*width)
print('✍️ Saved land-masked datasets to {}/ERA5/landonly/'.format(DATA_ROOT))

if MAKE_CHECK_PLOTS:
    print('-'*width)
    print("📊 Making check plots...")
    plot_side_by_side(
        land_mask_regridded['lsm'],
        land_mask['lsm'],
        titles=("Regridded Land/Sea Mask", "Original Land/Sea Mask"),
        val_plotted='Land/Sea Mask',
        save_figs=True,
        filename_args=['landmask_regrid_check_' + GRID, 'png', 'figs'])
    
    for VAR, ds_masked, ds in zip(vars, ds_maskeds, dss):
        plot_side_by_side(
            ds_masked['t2m'].sel(year=2000),
            ds['t2m'].sel(year=2000),
            titles=("Masked t2m", "Original t2m"),
            val_plotted='t2m',
            save_figs=True,
            filename_args=['t2m_landmask_check_' + VAR + '_' + GRID, 'png', 'figs'])
    
    print('-'*width)
    print("✅ Check plots are complete.")