"""Runner function for fitting GEV to ERA5 data - serialized.

Adam Bauer
UChicago
Apr 2026
"""

import shutil
import traceback
import time

import xarray as xr

from evt_heat_waves.config import ERA5_PATH, ANOM_TYPE_TO_VAR
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate

TERMINAL_WIDTH = shutil.get_terminal_size(fallback=(80, 20)).columns


def runner(logger, args):
    """Runner for GEV fitting of ERA5 data.

    Parameters
    ----------
    logger: logging.Logger
        logger object
    
    args: argparse.Namespace
        CLI arguments for fit details
    """
    start_time = time.time()

    vars = ['t2m_annual_max', 't2m_annual_min']
    anom_types = ['raw', 'annmean', 'trend']
    tmins = [1950, 1979]

    # Collect all tasks
    all_tasks = []

    for var in vars:
        logger.info(f"Setting up tasks for: {var}")

        for TMIN in tmins:
            for anom_type in anom_types:
                all_tasks.append({
                    'var': var,
                    'TMIN': TMIN,
                    'anom_type': anom_type
                })

    logger.info(f"Total tasks to process: {len(all_tasks)}")

    # Process all tasks serially
    all_results = []
    for task_idx, task in enumerate(all_tasks):
        logger.info(f"Processing task {task_idx + 1}/{len(all_tasks)}: "
              f"{task['var']}:{task['TMIN']}:{task['anom_type']}")

        result = process_single_fit(
            logger=logger,
            args=args,
            var=task['var'],
            TMIN=task['TMIN'],
            anom_type=task['anom_type'],
        )
        all_results.append(result)

    # Summarize results
    successes = sum(1 for r in all_results if r[0])
    failures = sum(1 for r in all_results if not r[0])

    fit_type_counts = {}
    for r in all_results:
        fit_type = r[1]
        if fit_type not in fit_type_counts:
            fit_type_counts[fit_type] = {'success': 0, 'failure': 0}
        if r[0]:
            fit_type_counts[fit_type]['success'] += 1
        else:
            fit_type_counts[fit_type]['failure'] += 1

    end_time = time.time()
    elapsed = end_time - start_time

    logger.info('-' * TERMINAL_WIDTH)
    logger.info("SUMMARY")
    logger.info('-' * TERMINAL_WIDTH)
    logger.info(f"    Successful: {successes}/{len(all_results)}")
    logger.info(f"    Failed: {failures}/{len(all_results)}")
    logger.info(f"    Breakdown by fit type:")
    for fit_type, counts in sorted(fit_type_counts.items()):
        total = counts['success'] + counts['failure']
        logger.info(f"      - {fit_type:8s}: {counts['success']}/{total} successful")
    logger.info(f"    Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"    Average time per task: {elapsed/len(all_results):.2f} seconds")

    if failures > 0:
        logger.info("    Failed tasks:")
        for r in all_results:
            if not r[0]:
                logger.info(f"   - {r[3]}")


def process_single_fit(logger, args, var, TMIN, anom_type):
    """Process a single fit for a single variable combination.
    
    Parameters
    ----------
    logger: logging.Logger
        logging object

    args: argparse.Namespace
        CLI args, needs to have
            - args.fit (fit type to pass to MLE)
            - args.no_se (whether to calc SEs)

    var : str
        Variable name

    TMIN: int
        starting year for MLE fit

    anom_type: str
        the type of anomaly (trend, raw, annmean)
        
    Returns
    -------
    tuple
        (success, anom_type, output_path, error_message)
    """
    try:
        data_path = ERA5_PATH / 'landonly'
        fpath = data_path / f"era5_{var}_{args.grid}_landonly.nc"

        ds = xr.open_dataset(fpath)
        ds = ds.sel(year=slice(TMIN, 2024))

        try:
            var_name = ANOM_TYPE_TO_VAR[anom_type]
            var_name = var_name if anom_type != 'raw' else 't2m'
        except KeyError:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        logger.debug(f"The anomaly type {anom_type} was converted to variable name {var_name}")

        ds_fit = ds_mle_fit(
            args,
            ds,
            var_name=var_name,
            fit_dim='year'
        )

        stat_success_rate = get_mle_success_rate()
        reset_mle_stats()

        logger.info(f"Completed fitting for {var}:{anom_type}:{TMIN}")

        ds_fit.attrs['MLE_success_rate'] = stat_success_rate

        # save dataset
        gev_dir = fpath.parent.parent / 'gev'
        gev_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists

        logger.debug(f"[Rank {rank}] Output directory for GEV fit: {gev_dir}")
        
        fit_fname = f"{fpath.stem}_gev_{args.fit}_TMIN{TMIN}_{anom_type}{fpath.suffix}"
        logger.debug(f"The output path is: {gev_dir / fit_fname}")

        ds_fit.to_netcdf(gev_dir / fit_fname)
        
        # close datasets to save memory
        ds.close()
        ds_fit.close()

        return (True, anom_type, fit_fname, None)

    except Exception as e:
        error_msg = f"Error processing {var}:{TMIN}:{anom_type} - {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return (False, anom_type, None, error_msg)