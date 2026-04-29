"""Runner function for fitting GEV to ERA5 data using MPI to parallelize across many fits.

Adam Bauer
UChicago
Apr 2026
"""

import shutil
import traceback

import xarray as xr

from evt_heat_waves.config import ERA5_PATH, ANOM_TYPE_TO_VAR
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate

TERMINAL_WIDTH = shutil.get_terminal_size(fallback=(80, 20)).columns

        
def print_summary(flat_results, logger):
    """Print a summary of MPI task results.

    Parameters
    ----------
    flat_results : list[tuple]
        List of (success, fit_type, output_path, error_message) tuples.
    all_results : list[list[tuple]]
        List of lists of (success, fit_type, output_path, error_message) tuples.
    logger : logging.Logger
        Logger used to report summary information.
    """
    # compute the number of successes and failures
    successes = sum(1 for r in flat_results if r[0])
    failures = sum(1 for r in flat_results if not r[0])
    
    # Count by fit type
    fit_type_counts = {}
    for r in flat_results:
        fit_type = r[1]
        if fit_type not in fit_type_counts:
            fit_type_counts[fit_type] = {'success': 0, 'failure': 0}
        if r[0]:
            fit_type_counts[fit_type]['success'] += 1
        else:
            fit_type_counts[fit_type]['failure'] += 1

    logger.info('-'*TERMINAL_WIDTH)
    logger.info("SUMMARY")
    logger.info('-'*TERMINAL_WIDTH)
    logger.info(f"Successful: {successes}/{len(flat_results)}")
    logger.info(f"Failed: {failures}/{len(flat_results)}")
    logger.info(f"Breakdown by fit type:")
    for fit_type, counts in sorted(fit_type_counts.items()):
        total = counts['success'] + counts['failure']
        logger.info(f"  - {fit_type:8s}: {counts['success']}/{total} successful")
    
    if failures > 0:
        logger.info("Failed tasks:")
        for r in flat_results:
            if not r[0]:
                logger.info(f"  - {r[3]}")
    
    logger.info('-' * TERMINAL_WIDTH)
        
def process_single_fit(logger, args, var, TMIN, anom_type, rank):
    """Process a single fit for a single model-variable combination.
    
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

    rank : int
        MPI rank of current process
        
    Returns
    -------
    tuple
        (success, anom_type, output_path, error_message)
    """

    try:
        # import data from ERA5/landonly
        data_path = ERA5_PATH / 'landonly'
        fpath = data_path / f"era5_{var}_{args.grid}_landonly.nc"

        ds = xr.open_dataset(fpath)
        ds = ds.sel(year=slice(TMIN, 2024))

        # mapping for data -> variable name in dataset
        try:
            var_name = ANOM_TYPE_TO_VAR[anom_type]
            var_name = var_name if anom_type != 'raw' else 't2m'  # convert to ERA5 naming convention if using raw data
        except KeyError:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        logger.debug(f"The anomaly type {anom_type} was converted to variable name {var_name}")

        # do fitting
        ds_fit = ds_mle_fit(
            args,
            ds,
            var_name=var_name,
            fit_dim='year'
        )

        # reset MLE success counter
        stat_success_rate = get_mle_success_rate()
        reset_mle_stats()

        logger.info(f"[RANK {rank}] Completed fitting for {var}:{anom_type}:{TMIN}")

        # set success rate
        ds_fit.attrs['MLE_success_rate'] = stat_success_rate
        
        # save dataset
        gev_dir = fpath.parent.parent / 'gev'
        gev_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists

        logger.debug(f"[Rank {rank}] Output directory for GEV fit: {gev_dir}")

        fit_fname = f"{fpath.stem}_gev_{args.fit}_TMIN{TMIN}_{anom_type}{fpath.suffix}"
        logger.debug(f"The output path is: {gev_dir / fit_fname}")

        ds_fit.to_netcdf(gev_dir / fit_fname)  # save kuiper results

        # close datasets to save memory
        ds.close()
        ds_fit.close()

        return (True, anom_type, fit_fname, None)

    except Exception as e:
        error_msg = f"Error processing {var}:{TMIN}:{anom_type} - {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[Rank: {rank}] {error_msg}")

        # return success, anomaly type, stationary fit output path, nonstationary fit output path, and error msg
        return (False, anom_type, None, error_msg)