"""Preprocess CMIP6 data.

This script will mask out the oceans in CMIP6, leaving only the land
surface for our analysis. Note we use the ERA5 land/sea mask for 
consistency with our ERA5-based analysis.

Adam Michael Bauer
UChicago
1.12.2026

Last edited: 1/28/2026, 12:30 PM CST
"""

import shutil
import yaml

import xarray as xr
import numpy as np

from datetime import datetime
from pathlib import Path
from evt_heat_waves.config import CMIP_PATH, ERA5_PATH, CONFIG_PATH, FIGS_PATH
from evt_heat_waves.plotting.check_plots import plot_side_by_side
from evt_heat_waves.pproc.preprocessing import make_regridded_land_mask
from evt_heat_waves.mip_fit.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.utils import extract_model_name, yaml_safe

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def pproc_cmip(logger, args):
    """Preprocess CMIP6 data to be analyzed in EVT pipeline.

    NOTE: This pipeline has two steps, unlike ERA5. The first is to create
    meta.yaml and qc.yaml files that contain information to construct the CMIP
    ensemble dataclass that is used throughout. The second is landmasking and anomaly
    computation, which mirrors the ERA5 script.

    Parameters
    ----------
    logger: logging.Logger
        Logger for logging info and debugging

    args: argparse.Namespace
        CLI arguments for the experiment
    """

    logger.info(f"Preprocessing {args.data} data...")
    logger.info("-" * width)

    # parse CLI arguments
    MAKE_CHECK_PLOTS = args.make_check_plots

    META_FILE = CONFIG_PATH.parent / 'meta.yaml'
    QC_FILE = CONFIG_PATH.parent / 'qc.yaml'

    if META_FILE.exists() and QC_FILE.exists():
        logger.info("meta.yaml and qc.yaml found. Skipping creation.")

    else:
        logger.info("metadata and quality control files not found, creating...")
        _make_meta_and_qc(logger, args)
    
        # stop here to allow for manual checking if desired
        if not args.bypass_checks:
            logger.info("! BREAK - Please check the files meta.generated.yaml and qc.generated.yaml before proceeding. Once happy, rename meta.generated.yaml -> meta.yaml and qc.generated.yaml -> qc.yaml and rerun.")
            return None  # stop here to allow for manual checking of meta.yaml and qc.yaml files before proceeding with land masking and regridding

    # make CMIP6 model config class (only used for plotting here)
    CMIPConfig = CMIP6EnsembleConfig.from_yaml(CONFIG_PATH.parent / "meta.yaml", 
                                               CONFIG_PATH.parent / "qc.yaml")

    # set mask threshold
    MASK_THRES = 0.5  # "standard", according to Rahul Singh :)

    # make annual mean data path for reference
    data_path_mean = CMIP_PATH / 'tas_annual_mean' / 'raw'

    # define variables
    vars = ['tas_annual_max', 'tas_annual_min']

    # import land mask, or if it's not there, make it with CMIP grid (1 deg x 1 deg)
    try:
        land_mask = xr.open_dataset(ERA5_PATH / 'era5_land_mask_1deg.nc')

    except FileNotFoundError:
        logger.info("No regridded land mask found, making a new one...")
        make_regridded_land_mask(GRID='1deg')
        land_mask = xr.open_dataset(ERA5_PATH / 'era5_land_mask_1deg.nc')

    for var in vars:
        logger.info(f"Regridding and masking {var} data...")

        # make data path
        data_path = CMIP_PATH / var / 'raw'

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

            logger.info("    ⚠️ Not all CMIP output have complete data records for analysis.")
            if missing_in_mean:
                logger.info(f"    Models that have {var} data but not annual mean data: {missing_in_mean}.")
            if missing_in_var:
                logger.info(f"    Models that have annual mean data but not {var} data: {missing_in_var}.")
            
            error_message = ("    CMIP ensemble data is incomplete."
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

            logger.info(f"    Working on {model_name}...")
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

            logger.info(f"    {model_name} land masking done and saved successfully to: {land_dir / land_name}")

            # close datasets to save memory
            ds.close()
            ds_mean.close()
            ds_masked.close()
            da_anom_annmean.close()
            da_anom_trend.close()

            # make check plots if desired
            if MAKE_CHECK_PLOTS:
                plot_side_by_side(
                    ds_masked['tas'].sel(year=2000,
                                        member_id=CMIPConfig.ensemble_config[model_name].primary_member),
                    ds['tas'].sel(year=2000,
                                member_id=CMIPConfig.ensemble_config[model_name].primary_member),
                    titles=("Masked tas", "Original tas"),
                    save_figs=True,
                    filename_args=['tas_landmask_check_' + var + '_' + model_name, 'png', FIGS_PATH / 'checks']
                    )
                logger.info(f"     Check plot for {model_name} saved.")


def _make_meta_and_qc(logger, args):
    """Make meta data and quality control files for CMIP ensemble.
    
    Parameters
    ----------
    logger: logging.Logger
        Logger for logging info and debugging

    args: argparse.Namespace
        CLI arguments for the experiment
    """

    # set variables to perform qc on, should be directory names in DATA_ROOT/data_type
    vars = ['tas_annual_max', 'tas_annual_mean', 'tas_annual_min']

    # metadata and quality control
    meta = {}
    qc = {}

    # required years
    required_years = np.arange(1979, 2025, 1)

    # loop through variables for QC
    for var in vars:
        logger.info("    " + "."*width)
        logger.info("    Performing QC on {} data...".format(var))
        logger.info("    " + "."*width)

        # set data path, making sure to QC on raw data
        data_path = CMIP_PATH / var / 'raw'

        # grab files
        files = [f for f in data_path.glob('*.nc')]
        var_qc =  {}

        for f in files:
            # define empty lists for each file / model
            model_qc = {}
            tmp_failure_mode = None
            tmp_active = True

            # get model name
            fparts = f.stem.split('_')
            model = '_'.join(fparts[2:3])

            logger.info("    Working on {}...".format(model))

            # open dataset
            ds = xr.open_dataset(f)

            # empty lists to populate later
            valid_yrs = []
            invalid_yrs = []

            # loop through the years i need. if i find any nans,
            # i add that year to the invalid years and i turn off the
            # model
            for yr in required_years:
                try:
                    data = ds.tas.sel(year=yr).values
                    if np.isnan(data).all():
                        invalid_yrs.append(yr)
                        tmp_active = False
                        tmp_failure_mode = 'nan_data'
                    else:
                        valid_yrs.append(yr)

                except Exception as e:
                    invalid_yrs.append(yr)
                    tmp_active = False
                    tmp_failure_mode = 'missing_data'

            # populate this model's qc data
            model_qc['valid_years'] = valid_yrs
            model_qc['invalid_years'] = invalid_yrs
            model_qc['failure_mode'] = tmp_failure_mode
            model_qc['active'] = tmp_active

            # add model to variable qc
            var_qc[model] = model_qc

            # close dataset to save memory
            ds.close()

        # add variable to qc
        qc[var] = var_qc

    # fill in metadata
    # NOTE: I'm assuming each quantity i consider has the same metadata, which should be fine,
    # but i didn't check it!
    meta_path = CMIP_PATH / 'tas_annual_max' / 'raw'
    meta_files = [f for f in meta_path.glob('*.nc')]

    for f in meta_files:
        # model-specific metadata dict
        model_meta = {}
    
        # get model name
        fparts = f.stem.split('_')
        model = '_'.join(fparts[2:3])

        # open dataset
        ds = xr.open_dataset(f)

        # extract relevant info
        model_meta['ensemble_members'] = [str(m) for m in ds.member_id.values]
        model_meta['N_members'] = int(len(model_meta['ensemble_members']))
        model_meta['primary_member'] = str(ds.member_id.values[0])  # for now

        # close dataset to save memory
        ds.close()

        # store model metadata in the dict
        meta[model] = model_meta

    # make final dictionaries
    metas = {}
    qcs = {}

    # adding this to remember last time i did all this
    metas['generated_on'] = datetime.now().isoformat()
    qcs['generated_on'] = datetime.now().isoformat()

    # populate
    metas['models'] = meta
    qcs['models'] = qc

    # save dictionaries as .yamls 
    outpath_meta = Path(CONFIG_PATH.parent / 'meta.yaml') if args.bypass_checks else Path(CONFIG_PATH.parent / 'meta.generated.yaml')
    with open(outpath_meta, 'w') as f:
        yaml.safe_dump(
            yaml_safe(metas),
            f,
            sort_keys=True,
            default_flow_style=False,
            indent=2
        )

    outpath_qc = Path(CONFIG_PATH.parent / 'qc.yaml') if args.bypass_checks else Path(CONFIG_PATH.parent / 'qc.generated.yaml')
    with open(outpath_qc, 'w') as f:
        yaml.safe_dump(
            yaml_safe(qcs),
            f,
            sort_keys=True,
            default_flow_style=False,
            indent=2
        )