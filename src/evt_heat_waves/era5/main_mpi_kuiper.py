"""Main file for computing Kuiper statistics to analyze ERA5 GEV fit quality using
MPI parallelization.

Adam Michael Bauer
UChicago
Apr 2026

Last edited: 1/29/2026
"""

import shutil
import time

from mpi4py import MPI

from evt_heat_waves.logging_utils import setup_logger, get_git_hash
from evt_heat_waves.era5.cli import parse_args_era5_fit, check_kuiper_config_compatability
from evt_heat_waves.era5.kuiper.kuipers_mpi import process_single_kuiper, print_summary

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def main():    
    # Initialize MPI
    comm = MPI.COMM_WORLD  # communicator object -- allows communication across tasks
    rank = comm.Get_rank()  # gets *this* process's unique ID
    size = comm.Get_size()  # total number of processes
    
    # Only rank 0 does initial setup and task distribution
    ## rank 0 does all the initial setup and distribution because I/O operations
    ## like reading .yaml files and so on don't parallelize well
    if rank == 0:
        # only rank 0 does setup and distribution
        args = parse_args_era5_fit()
        check_kuiper_config_compatability(args)
        logger = setup_logger(args.debug)

        start_time = time.time()
        logger.info('-' * width)
        logger.info(f"Git hash: {get_git_hash()}")
        logger.info(f"Doing GEV fitting for config: {args.fit}|{args.grid}|NO_SE={args.no_se}")
        
        # Define variables, stationary/nonstationary, and anomaly types to parallelize over
        vars = ['t2m_annual_max', 't2m_annual_min']
        anom_types = ['raw', 'annmean', 'trend']
        tmins = [1950, 1979]
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            print(f"Setting up tasks for: {var}")
            
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
        
        print(f"Total tasks to process: {len(all_tasks)}")
        print(f"Number of MPI processes: {size}")
        print(f"Tasks per process: ~{len(all_tasks) / size:.1f}")
    
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
    
    print(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    ## each task runs this independently, since it is embarrassingly parallelizable
    my_results = []  # "my" refers to the task that's running this -- it's different for each one
    
    # loop through tasks for this rank and perform operations...
    for task_idx, task in enumerate(my_tasks):
        print(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['TMIN']}:{task['anom_type']}")
        
        result = process_single_kuiper(
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
        end_time = time.time()
        elapsed = end_time - start_time

        print_summary(flat_results, all_results, logger)

        logger.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"Average time per task: {elapsed/len(flat_results):.2f} seconds")

if __name__ == "__main__":
    main()