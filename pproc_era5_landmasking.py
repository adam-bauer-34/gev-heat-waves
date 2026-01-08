"""Preprocess ERA5 data.

This script will mask out the oceans in ERA5, leaving only the land
surface for our analysis.

Adam Michael Bauer
UChicago
11.10.2025

To run: pproc_era5_landmasking.py VAR GRID MAKE_CHECK_PLOTS
"""

import sys

import xarray as xr
import xesmf as xe
import numpy as np

from config import DATA_ROOT
from src.check_plots import plot_side_by_side

# import command line stuff
GRID, MAKE_CHECK_PLOTS = sys.argv[1:]
MAKE_CHECK_PLOTS = bool(int(MAKE_CHECK_PLOTS))

print("Importing data...")
# load in full ERA5 data
vars = ['t2m_annual_max', 't2m_annual_mean', 't2m_annual_min']
dss = [xr.open_dataset(DATA_ROOT / 'ERA5' / ('era5_' + VAR + '_' + GRID + '.nc')) for VAR in vars]

# first compute anomalies relative to annual mean for maximum and minimum datasets
da_t2m_max_anoms = dss[0]['t2m'] - dss[1]['t2m']
da_t2m_min_anoms = dss[1]['t2m'] - dss[2]['t2m']

# add anomalies to datasets
dss[0] = dss[0].assign({'t2m_anom': da_t2m_max_anoms})
dss[2] = dss[2].assign({'t2m_anom': da_t2m_min_anoms})

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

print("Making regridding object... (this could take a second)")
# initialize the regridder and regrid the land mask
regridder = xe.Regridder(land_mask, ds_output_grid, 'conservative')
land_mask_regridded = regridder(land_mask, keep_attrs=True)

print("Applying land mask to dataset...")
# apply the land mask to the dataset
MASK_THRES = 0.5  # threshold for me to consider something "land"
ds_maskeds = [ds.where(land_mask_regridded['lsm'].data[0] > MASK_THRES, np.nan) for ds in dss]

# save datasets
for VAR, ds_masked in zip(vars, ds_maskeds):
    ds_masked.to_netcdf(DATA_ROOT / 'ERA5' / 'landonly' / ('era5_' + VAR + '_' + GRID + '_landonly.nc'))

print("Complete.")

print('Saved land-masked datasets to {}/ERA5/landonly/'.format(DATA_ROOT))

if MAKE_CHECK_PLOTS:
    print("Making check plots...")
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
        
    print("Check plots are complete.")