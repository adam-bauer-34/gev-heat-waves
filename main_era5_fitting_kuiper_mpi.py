"""Main file for fitting GEV distributions to ERA5 temperatures and computing
the corresponding Kuiper statistic at each gridcell.

This script uses `mpi4py` to parallelize operations across many tasks on UChicago's
Midway3 cluster.

Adam Bauer
UChicago

To run: mpirun -n [number of tasks in allocation] python main_era5_fitting_kuiper_mpi.py [GRID]
"""

import sys
import shutil
import os
import traceback
import time

import xarray as xr

from mpi4py import MPI
from config import DATA_ROOT
from src.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate
from src.kuiper import compute_kuiper_stats
from pathlib import Path


def process_single_fit_and_kuiper(var, TMIN, anom_type, GRID, width, rank):
    """
    Docstring for process_single_fit_and_kuiper
    
    Parameters
    ----------
    :param var: Description
    :param STAT: Description
    :param TMIN: Description
    :param fit_type: Description
    :param width: Description
    :param rank: Description

    Returns
    -------
    ds
    """

    try:
        # import data from ERA5/landonly
        data_path = DATA_ROOT / 'ERA5' / 'landonly'
        fpath = data_path / (
            'era5_' + var + '_' + GRID + '_landonly.nc'
        )
        ds = xr.open_dataset(fpath)
        ds = ds.sel(year=slice(TMIN, 2024))

        gev_dir = fpath.parent.parent / 'gev'
        kuiper_name = f"era5_{var}_{GRID}_landonly_gev_stat_TMIN{TMIN}_[var_suffix]_kuiper.nc"
        stat_output_path = gev_dir / kuiper_name

        print(f"Rank [{rank}]: output path is {stat_output_path}")

        # ==============================
        # STEP 1: DO STATIONARY FIT
        # ==============================
        if anom_type == 'raw':
            ds_stat_fit = ds_mle_fit(
                ds,
                var_name='t2m',
                fit_dim='year',
                non_stat=False
            )
            var_suffix = 'raw'

        elif anom_type == 'annmean':
            ds_stat_fit = ds_mle_fit(
                ds,
                var_name='t2m_anom_annmean',
                fit_dim='year',
                non_stat=False
            )
            var_suffix = 'annmean'
        
        elif anom_type == 'trend':
            ds_stat_fit = ds_mle_fit(
                ds,
                var_name='t2m_anom_trend',
                fit_dim='year',
                non_stat=False
            )
            var_suffix = 'trend'
        
        else:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        # reset MLE success counter
        stat_success_rate = get_mle_success_rate()
        reset_mle_stats()

        print(f"[RANK {rank}] Completed stationary fitting for {var}:{anom_type}:{TMIN}. Moving on to Kuiper statistics...")

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
    

def main():
    """The main function that calls mpi4py and executes across multiple tasks.
    """

    # Initialize MPI
    comm = MPI.COMM_WORLD  # communicator object -- allows communication across tasks
    rank = comm.Get_rank()  # gets *this* process's unique ID
    size = comm.Get_size()  # total number of processes
    
    # Get command line arguments
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: srun -n <nprocs> python main_cmip_fitting_mpi.py STAT")
            print("where STAT is 'stat' or 'nonstat'")
        sys.exit(1)
    
    # import command line arguments
    GRID = sys.argv[1]

    # terminal width
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    
    # Only rank 0 does initial setup and task distribution
    ## rank 0 does all the initial setup and distribution because I/O operations
    ## like reading .yaml files and so on don't parallelize well
    if rank == 0:
        start_time = time.time()
        print('='*width)
        print(f"🚀 Starting MPI parallel processing with {size} processes")
        print('='*width)
        
        # Define variables, stationary/nonstationary, and anomaly types to parallelize over
        vars = ['t2m_annual_max', 't2m_annual_min']
        anom_types = ['raw', 'annmean', 'trend']
        tmins = [1979]
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            print('='*width)
            print(f"🏋🏼‍♀️ Setting up tasks for: {var}")
            print('='*width)
            
            # Make data directory if it doesn't exist
            os.makedirs(DATA_ROOT / 'ERA5' / 'gev', exist_ok=True)
            
            # Collect tasks for this variable
            # Now we create 3 tasks per model (one for each fit type)
            for TMIN in tmins:
                for anom_type in anom_types:
                    all_tasks.append({
                        'var': var,
                        'TMIN': TMIN,
                        'anom_type': anom_type,
                        'GRID': GRID,
                        'width': width
                    })
        
        print(f"\n📋 Total tasks to process: {len(all_tasks)}")
        print(f"   ({len(vars)} variables × {len(anom_types)} amomaly types per variables × {len(tmins)} minimum times per fit)")
        print(f"🖥️  Number of MPI processes: {size}")
        print(f"📊 Tasks per process: ~{len(all_tasks) / size:.1f}")
        print('='*width)
    
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
            var=task['var'],
            TMIN=task['TMIN'],
            anom_type=task['anom_type'],
            GRID=task['GRID'],
            width=task['width'],
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
        
        print('='*width)
        print("📊 SUMMARY")
        print('='*width)
        print(f"✅ Successful: {successes}/{len(flat_results)}")
        print(f"❌ Failed: {failures}/{len(flat_results)}")
        print(f"\nBreakdown by fit type:")
        for fit_type, counts in sorted(fit_type_counts.items()):
            total = counts['success'] + counts['failure']
            print(f"  {fit_type:8s}: {counts['success']}/{total} successful")
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"⚡ Average time per task: {elapsed/len(flat_results):.2f} seconds")
        
        if failures > 0:
            print("\n❌ Failed tasks:")
            for r in flat_results:
                if not r[0]:
                    print(f"   - {r[3]}")
        
        print('='*width)
        print("🥳 All done! 🥳")
        print('='*width)

if __name__ == '__main__':
    main()  # R U N   I T