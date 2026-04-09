"""Function bank for serial (non-MPI) GEV fitting to CMIP data.

Adam Michael Bauer
UChicago
Apr 8 2026
"""

import os
import shutil
import time

import xarray as xr

from evt_heat_waves.config import MIP_FIT_PATH_DICT, CONFIG_PATH
from evt_heat_waves.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.utils import extract_model_name
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def runner(logger, args):
    """Serial runner for GEV fitting of CMIP data.

    Parameters
    ----------
    logger: logging.Logger
        logger object
    
    args: argparse.Namespace
        CLI arguments for fit details
    """

    start_time = time.time()
    logger.info("Starting SERIAL processing (no MPI)")

    # Setup CMIP config object
    CMIPConfig = CMIP6EnsembleConfig.from_yaml(
        CONFIG_PATH.parent / "meta.yaml",
        CONFIG_PATH.parent / "qc.yaml"
    )

    # Define variables and fit types
    vars = ['tas_annual_max', 'tas_annual_min']
    anom_types = ['raw', 'annmean', 'trend']

    all_tasks = []

    # Build task list
    for var in vars:
        logger.info(f"Setting up tasks for: {var}")

        os.makedirs(MIP_FIT_PATH_DICT[args.data] / var / 'gev', exist_ok=True)
        data_path = MIP_FIT_PATH_DICT[args.data] / var / 'landonly'

        fnames = [f for f in data_path.glob("*_landonly.nc")]
        modelname_filepath_matcher = {
            extract_model_name(f): f for f in fnames
        }

        for m in CMIPConfig.iter_active_models(var):
            for anom_type in anom_types:
                all_tasks.append({
                    'var': var,
                    'anom_type': anom_type,
                    'model': m,
                    'filepath_matcher': modelname_filepath_matcher,
                })

    logger.info(f"Total tasks to process: {len(all_tasks)}")

    # Process tasks sequentially
    results = []

    for i, task in enumerate(all_tasks):
        logger.info(
            f"Processing task {i+1}/{len(all_tasks)}: "
            f"{task['var']}:{task['model'].name}:{task['anom_type']}"
        )

        result = process_single_fit(
            logger=logger,
            args=args,
            var=task['var'],
            anom_type=task['anom_type'],
            m=task['model'],
            modelname_filepath_matcher=task['filepath_matcher'],
        )

        results.append(result)

    # ---- Summary ----
    successes = sum(1 for r in results if r[0])
    failures = sum(1 for r in results if not r[0])

    fit_type_counts = {}
    for r in results:
        fit_type = r[1]
        if fit_type not in fit_type_counts:
            fit_type_counts[fit_type] = {'success': 0, 'failure': 0}
        if r[0]:
            fit_type_counts[fit_type]['success'] += 1
        else:
            fit_type_counts[fit_type]['failure'] += 1

    elapsed = time.time() - start_time

    logger.info("-" * width)
    logger.info("SUMMARY")
    logger.info("-" * width)
    logger.info(f"    Successful: {successes}/{len(results)}")
    logger.info(f"    Failed: {failures}/{len(results)}")
    logger.info("    Breakdown by fit type:")

    for fit_type, counts in sorted(fit_type_counts.items()):
        total = counts['success'] + counts['failure']
        logger.info(f"      - {fit_type:8s}: {counts['success']}/{total} successful")

    logger.info(f"    Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"    Average time per task: {elapsed/len(results):.2f} seconds")

    if failures > 0:
        logger.info("    Failed tasks:")
        for r in results:
            if not r[0]:
                logger.info(f"   - {r[3]}")


def process_single_fit(logger, args, var, anom_type, m, modelname_filepath_matcher):
    """Process a single fit for a single model-variable combination.
    
    Parameters
    ----------
    logger: logging.Logger
        logging object

    args: argparse.Namespace
        CLI args, needs to have
            - args.fit (fit type to pass to MLE)
            - args.data (cmip or amip)

    var : str
        Variable name

    anom_type: str
        the type of anomaly (trend, raw, annmean)

    m : Model object
        Model configuration object

    modelname_filepath_matcher : dict
        Dictionary mapping model names to file paths

    width : int
        Terminal width for formatting

    rank : int
        MPI rank of current process
        
    Returns
    -------
    tuple
        (success, anom_type, output_path, error_message)
    """

    try:
        logger.info(f"Working on {var}:{m.name} - {anom_type} fit")

        fpath = modelname_filepath_matcher[m.name]
        ds = xr.open_dataset(fpath)
        ds_selected = ds.sel(member_id=m.primary_member)

        # Select fit type
        if anom_type == 'raw':
            ds_fit = ds_mle_fit(args, ds_selected, var_name='tas', fit_dim='year')
            var_suffix = 'raw'

        elif anom_type == 'annmean':
            ds_fit = ds_mle_fit(args, ds_selected, var_name='t2m_anom_annmean', fit_dim='year')
            var_suffix = 'annmean'

        elif anom_type == 'trend':
            ds_fit = ds_mle_fit(args, ds_selected, var_name='t2m_anom_trend', fit_dim='year')
            var_suffix = 'trend'

        else:
            raise ValueError(f"Unknown fit_type: {anom_type}")

        logger.info(f"{anom_type} fit complete.")

        # MLE stats
        success_rate = get_mle_success_rate()
        reset_mle_stats()

        ds_fit.attrs['MLE_success_rate'] = success_rate

        # Save
        gev_dir = fpath.parent.parent / 'gev'
        gev_name = fpath.with_name(
            fpath.stem + f"_gev_{args.fit}_{var_suffix}" + fpath.suffix
        ).name

        output_path = gev_dir / gev_name
        ds_fit.to_netcdf(output_path)

        logger.info(f"Dataset saved to: {output_path}")

        ds_fit.close()
        ds.close()

        return (True, anom_type, str(output_path), None)

    except Exception as e:
        import traceback
        error_msg = (
            f"Error processing {var}:{m.name}:{anom_type} - {str(e)}\n"
            f"{traceback.format_exc()}"
        )
        logger.warning(error_msg)
        return (False, anom_type, None, error_msg)