"""Runner function for fitting GEV to ERA5 data using MPI to parallelize across many fits.

Adam Bauer
UChicago
Apr 2026
"""

import sys
import shutil
import os
import traceback
import time

import xarray as xr
from mpi4py import MPI

from evt_heat_waves.config import ERA5_PATH, ANOM_TYPE_TO_VAR
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate
from evt_heat_waves.kuiper.kuiper_fitting import compute_kuiper_stats

width = shutil.get_terminal_size(fallback=(80, 20)).columns


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
        anom_types = ['raw', 'annmean', 'trend']
        tmins = [1950, 1979]
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            print(f"🏋🏼Setting up tasks for: {var}")
            
            # Make data directory if it doesn't exist
            os.makedirs(ERA5_PATH / 'gev', exist_ok=True)
            
            # Collect tasks for this variable
            # Now we create 3 tasks per model (one for each fit type)
            for TMIN in tmins:
                for anom_type in anom_types:
                    all_tasks.append({
                        'logger': logger,
                        'args': args,
                        'var': var,
                        'TMIN': TMIN,
                        'anom_type': anom_type
                    })
        
        print(f"Total tasks to process: {len(all_tasks)}")
        print(f"Number of MPI processes: {size}")
        print(f"Tasks per process: ~{len(all_tasks) / size:.1f}")
    
    # other workers are idle while all of this gets setup since I/O and task setup
    # is not easily parallelizable
    else:
        all_tasks = None
    
    # Broadcast tasks to all processes
    all_tasks = comm.bcast(all_tasks, root=0)  # set root rank to zero
    
    # Distribute tasks using round-robin distribution
    # this is good for tasks that take about as long to take as one another
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    print(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    ## each task runs this independently, since it is embarrassingly parallelizable
    my_results = []  # "my" refers to the task that's running this -- it's different for each one
    
    # loop through tasks for this rank and perform operations...
    for task_idx, task in enumerate(my_tasks):
        print(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['TMIN']}:{task['anom_type']}")
        
        result = process_single_fit_and_kuiper(
            logger=task['logger'],
            args=task['args'],
            var=task['var'],
            TMIN=task['TMIN'],
            anom_type=task['anom_type'],
            GRID=task['GRID'],
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
        
        print('-'*width)
        print("SUMMARY")
        print('-'*width)
        print(f"    Successful: {successes}/{len(flat_results)}")
        print(f"    Failed: {failures}/{len(flat_results)}")
        print(f"    Breakdown by fit type:")
        for fit_type, counts in sorted(fit_type_counts.items()):
            total = counts['success'] + counts['failure']
            print(f"      - {fit_type:8s}: {counts['success']}/{total} successful")
        print(f"    Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"    Average time per task: {elapsed/len(flat_results):.2f} seconds")
        
        if failures > 0:
            print("    Failed tasks:")
            for r in flat_results:
                if not r[0]:
                    print(f"   - {r[3]}")
        
def process_single_fit_and_kuiper(logger, args, var, TMIN, anom_type, GRID, rank):
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

    grid: str
        the grid type (e.g., "1deg" or "0.5deg")

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
        fpath = data_path / (
            'era5_' + var + '_' + GRID + '_landonly.nc'
        )
        ds = xr.open_dataset(fpath)
        ds = ds.sel(year=slice(TMIN, 2024))

        gev_dir = fpath.parent.parent / 'gev'

        logger.debug(f"[Rank {rank}] Output directory for GEV fit: {gev_dir}")

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

        print(f"[RANK {rank}] Completed fitting for {var}:{anom_type}:{TMIN}")

        # ==================================
        # STEP 2: COMPUTE KUIPER STATISTICS
        # ==================================     
        if anom_type == 'raw':
            ds_kuiper = compute_kuiper_stats(
                ds_stat_fit,
                var_name='t2m'
            )
            var_suffix = 'raw'

        elif anom_type == 'annmean':
            ds_kuiper = compute_kuiper_stats(
                ds_stat_fit,
                var_name='t2m_anom_annmean'
            )
            var_suffix = 'annmean'            

        elif anom_type == 'trend':
            ds_kuiper = compute_kuiper_stats(
                ds_stat_fit,
                var_name='t2m_anom_trend'
            )
            var_suffix = 'trend'

        else:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        
        # set success rate
        ds_kuiper.attrs['MLE_success_rate'] = stat_success_rate

        # reset success rate for mle
        reset_mle_stats()

        # check: print kuiper dataset
        print(f"[Rank {rank}]: Kuiper statistics-fitted dataset:\n {ds_kuiper}")
        
        # save joined dataset from stationary + kuiper stats
        gev_dir = fpath.parent.parent / 'gev'
        gev_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists

        kuiper_name = f"era5_{var}_{GRID}_landonly_gev_stat_TMIN{TMIN}_{var_suffix}_kuiper.nc"
        stat_output_path = gev_dir / kuiper_name

        print(f"The output path is: {stat_output_path}")

        ds_kuiper.to_netcdf(stat_output_path)  # save kuiper results

        # close kuiper and stationary datasets after saving to keep memory abundant
        ds_kuiper.close()
        ds_stat_fit.close()

        # ==============================
        # STEP 3: DO NONSTATIONARY FIT
        # ==============================
        if anom_type == 'raw':
            ds_nonstat_fit = ds_mle_fit(
                ds,
                var_name='t2m',
                fit_dim='year',
                non_stat=True
            )
            var_suffix = 'raw'

        elif anom_type == 'annmean':
            ds_nonstat_fit = ds_mle_fit(
                ds,
                var_name='t2m_anom_annmean',
                fit_dim='year',
                non_stat=True
            )
            var_suffix = 'annmean'            
        
        elif anom_type == 'trend':
            ds_nonstat_fit = ds_mle_fit(
                ds,
                var_name='t2m_anom_trend',
                fit_dim='year',
                non_stat=True
            )
            var_suffix = 'trend'
        
        else:
            raise ValueError(f"Unknown anom_type: {anom_type}")
        
        # get mle success rate
        nonstat_success_rate = get_mle_success_rate()
        ds_nonstat_fit.attrs['MLE_success_rate'] = nonstat_success_rate
        reset_mle_stats()

        # save nonstationary dataset
        nonstat_output_path = gev_dir / f"era5_{var}_{GRID}_landonly_gev_nonstat_TMIN{TMIN}_{var_suffix}.nc"
        print(nonstat_output_path)
        ds_nonstat_fit.to_netcdf(nonstat_output_path)  # save kuiper results

        # close dataset
        # return success, anomaly type, stationary fit output path, nonstationary fit output path, and error msg
        return (True, anom_type, stat_output_path, nonstat_output_path, None)

    except Exception as e:
        error_msg = f"Error processing {var}:{anom_type} function call with TMIN={TMIN} - str{e}\n{traceback.format_exc()}"
        print(f"Rank: {rank}] ❌ {error_msg}")

        # return success, anomaly type, stationary fit output path, nonstationary fit output path, and error msg
        return (False, anom_type, None, None, error_msg)