"""Preprocess ERA5 data.

Adam Michael Bauer
UChicago
Apr 8, 2026
"""

import shutil

import xarray as xr
import xesmf as xe
import numpy as np

from evt_heat_waves.config import ERA5_PATH, FIGS_PATH
from evt_heat_waves.check_plots.check_plots import plot_side_by_side

width = shutil.get_terminal_size(fallback=(80, 20)).columns

def pproc_era5(logger, args):
    """Preprocess ERA5 data to be analyzed in EVT pipeline.

    Parameters
    ----------
    logger: logging.Logger
        Logger for logging info and debugging

    args: argparse.Namespace
        CLI arguments for the experiment
    """

    logger.info("Preprocessing ERA5 data...")

    # parse CLI arguments
    GRID = args.grid
    MAKE_CHECK_PLOTS = args.make_check_plots

    logger.info('-' * width)
    logger.info("Importing data...")
    # load in full ERA5 data
    vars = ['t2m_annual_max', 't2m_annual_mean', 't2m_annual_min']
    dss = [xr.open_dataset(ERA5_PATH / ('era5_' + var + '_' + GRID + '.nc')) for var in vars]

    # COMPUTE ANOMALIES
    logger.info("Computing anomalies...")
    logger.info("  - Anoms with respect to annual mean (removing climate change and interannual variability signal)...")
    # 1. compute anomalies relative to annual mean for maximum and minimum datasets
    da_t2m_max_anoms = dss[0]['t2m'] - dss[1]['t2m']  # annual max - annual mean
    da_t2m_min_anoms = dss[2]['t2m'] - dss[1]['t2m']  # annual min - annual mean

    # add anomalies to datasets
    dss[0] = dss[0].assign({'t2m_anom_annmean': da_t2m_max_anoms})
    dss[2] = dss[2].assign({'t2m_anom_annmean': da_t2m_min_anoms})

    logger.info("  - Anoms with respect to trend in annual mean (removing climate change signal)...")
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

    # load in land/sea mask
    land_mask = xr.open_dataset(ERA5_PATH / 'era5_land_mask.nc')

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

    logger.info("Making regridding object... (this could take a second)")
    # initialize the regridder and regrid the land mask
    regridder = xe.Regridder(land_mask, ds_output_grid, 'conservative')
    land_mask_regridded = regridder(land_mask, keep_attrs=True)

    logger.info("Applying land mask to dataset...")
    # apply thea land mask to the dataset
    MASK_THRES = 0.5  # threshold for me to consider something "land" in the mask
    ds_maskeds = [ds.where(land_mask_regridded['lsm'].data[0] > MASK_THRES, np.nan) for ds in dss]

    # create landonly directory if it doesn't exist
    landonly_dir = ERA5_PATH / 'landonly'
    landonly_dir.mkdir(parents=True, exist_ok=True)

    # save datasets
    for VAR, ds_masked in zip(vars, ds_maskeds):
        ds_masked.to_netcdf(landonly_dir / ('era5_' + VAR + '_' + GRID + '_landonly.nc'))

    logger.info(f'Saved land-masked datasets to {ERA5_PATH}/landonly')

    if MAKE_CHECK_PLOTS:
        plot_side_by_side(
            land_mask_regridded['lsm'],
            land_mask['lsm'],
            titles=("Regridded Land/Sea Mask", "Original Land/Sea Mask"),
            save_figs=True,
            filename_args=['era5_landmask_regrid_check_' + GRID, 'png', f'{FIGS_PATH}/checks'])
        
        for VAR, ds_masked, ds in zip(vars, ds_maskeds, dss):
            plot_side_by_side(
                ds_masked['t2m'].sel(year=2000),
                ds['t2m'].sel(year=2000),
                titles=("Masked t2m", "Original t2m"),
                save_figs=True,
                filename_args=['era5_t2m_landmask_check_' + VAR + '_' + GRID, 'png', f'{FIGS_PATH}/checks'])
        
        logger.info("Check plots completed.")