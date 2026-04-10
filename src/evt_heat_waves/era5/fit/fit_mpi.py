"""Runner function for fitting GEV to ERA5 data using MPI to parallelize across many fits.

Adam Bauer
UChicago
Apr 2026
"""

import shutil
import traceback
import time

import xarray as xr
from mpi4py import MPI

from evt_heat_waves.config import ERA5_PATH, ANOM_TYPE_TO_VAR
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate
from evt_heat_waves.logging_utils import setup_logger

TERMINAL_WIDTH = shutil.get_terminal_size(fallback=(80, 20)).columns


def runner(logger, args):
    """Runner for GEV fitting of CMIP data - one "primary" ensemble member
    per model.

    Parameters
    ----------
    logger: logging.Logger
        logger object
    
    args: argparse.Namespace
        CLI arguments for fit details
    """

    # Initialize MPI
    comm = MPI.COMM_WORLD  # communicator object -- allows communication across tasks
    rank = comm.Get_rank()  # gets *this* process's unique ID
    size = comm.Get_size()  # total number of processes
    
    # Only rank 0 does initial setup and task distribution
    ## rank 0 does all the initial setup and distribution because I/O operations
    ## like reading .yaml files and so on don't parallelize well
    if rank == 0:
        start_time = time.time()
        logger.info(f"Starting MPI parallel processing with {size} processes")
        
        # Define variables, stationary/nonstationary, and anomaly types to parallelize over
        vars = ['t2m_annual_max', 't2m_annual_min']
        anom_types = ['raw', 'annmean']
        tmins = [1979]
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            logger.info(f"Setting up tasks for: {var}")
            
            # Collect tasks for this variable
            # Now we create 3 tasks per model (one for each fit type)
            for TMIN in tmins:
                for anom_type in anom_types:
                    all_tasks.append({
                        'args': args,
                        'var': var,
                        'TMIN': TMIN,
                        'anom_type': anom_type
                    })
        
        logger.info(f"Total tasks to process: {len(all_tasks)}")
        logger.info(f"Number of MPI processes: {size}")
        logger.info(f"Tasks per process: ~{len(all_tasks) / size:.1f}")
    
    # other workers are idle while all of this gets setup since I/O and task setup
    # is not easily parallelizable
    else:
        all_tasks = None
    
    # Broadcast tasks to all processes
    all_tasks = comm.bcast(all_tasks, root=0)  # set root rank to zero
    logger = setup_logger(args.debug)
    logger.debug(f"[Rank {rank}] logger initalized successfully")
    
    # Distribute tasks using round-robin distribution
    # this is good for tasks that take about as long to take as one another
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    logger.info(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    ## each task runs this independently, since it is embarrassingly parallelizable
    my_results = []  # "my" refers to the task that's running this -- it's different for each one
    
    # loop through tasks for this rank and perform operations...
    for task_idx, task in enumerate(my_tasks):
        logger.info(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['TMIN']}:{task['anom_type']}")
        
        result = process_single_fit(
            logger=logger,
            args=task['args'],
            var=task['var'],
            TMIN=task['TMIN'],
            anom_type=task['anom_type'],
            rank=rank
        )
        my_results.append(result)
    
    # Gather all results to rank 0
    all_results = comm.gather(my_results, root=0)
    
    # Rank 0 prints summary
    if rank == 0:
        # Flatten results
        flat_results = [item for sublist in all_results for item in sublist]
        
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
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        logger.info('-'*TERMINAL_WIDTH)
        logger.info("SUMMARY")
        logger.info('-'*TERMINAL_WIDTH)
        logger.info(f"    Successful: {successes}/{len(flat_results)}")
        logger.info(f"    Failed: {failures}/{len(flat_results)}")
        logger.info(f"    Breakdown by fit type:")
        for fit_type, counts in sorted(fit_type_counts.items()):
            total = counts['success'] + counts['failure']
            logger.info(f"      - {fit_type:8s}: {counts['success']}/{total} successful")
        logger.info(f"    Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"    Average time per task: {elapsed/len(flat_results):.2f} seconds")
        
        if failures > 0:
            logger.info("    Failed tasks:")
            for r in flat_results:
                if not r[0]:
                    logger.info(f"   - {r[3]}")
        
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