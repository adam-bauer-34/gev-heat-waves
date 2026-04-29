"""Main file for GEV fitting of CMIP data - MPI parallelized version with independent fits.

Adam Michael Bauer
UChicago
Jan 2026

Each of the 3 fits per model is treated as an independent task.
To run: 
    srun python main_cmip_fitting_mpi.py STAT

Last edited: 1/29/2026
"""

import os
import shutil
import time

from mpi4py import MPI

from evt_heat_waves.config import MIP_FIT_PATH_DICT
from evt_heat_waves.mip_fit.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.mip_fit.prim.prim_mpi import process_single_fit, print_summary
from evt_heat_waves.utils import extract_model_name
from evt_heat_waves.logging_utils import setup_logger, get_git_hash
from evt_heat_waves.mip_fit.cli import parse_args_mip_fit

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD  # communicator object -- allows communication across tasks
    rank = comm.Get_rank()  # gets *this* process's unique ID
    size = comm.Get_size()  # total number of processes

    if rank == 0:
        # only rank 0 does setup and distribution
        args = parse_args_mip_fit()
        logger = setup_logger(args.debug)

        start_time = time.time()

        logger.info('-' * width)
        logger.info(f"Git hash: {get_git_hash()}")
        logger.info(f"Doing GEV fitting for config: {args.data}|{args.member_config}|MPI=True")

        # extract path info
        try:
            head_data_path = MIP_FIT_PATH_DICT[args.data]['data']
            config_meta_path = MIP_FIT_PATH_DICT[args.data]['config']['meta']
            config_qc_path = MIP_FIT_PATH_DICT[args.data]['config']['qc']
        except Exception as e:
            raise KeyError(f"Error in data config for GEV fitting: {str(e)}")
        
        # Setup CMIP config object
        CMIPConfig = CMIP6EnsembleConfig.from_yaml(config_meta_path, 
                                                   config_qc_path)
        
        # Define variables
        vars = ['tas_annual_max', 'tas_annual_min']
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            logger.info(f"Setting up tasks for: {var}")

            # Make data directory if it doesn't exist
            os.makedirs(head_data_path / var / 'gev', exist_ok=True)
            data_path = head_data_path / var / 'landonly'
            
            # Make all landonly file names
            fnames = [f for f in data_path.glob("*_landonly.nc")]
            modelname_filepath_matcher = {
                extract_model_name(f): f for f in fnames
            }
            
            # Collect tasks for this variable
            # Now we create 3 tasks per model (one for each fit type)
            if args.debug:
                # Define truncated anom types and one model
                anom_types = ['annmean', 'trend']
                m = [m for m in CMIPConfig.iter_active_models(var)][0]  # just first model

                for anom_type in anom_types:
                    all_tasks.append({
                        'args': args,
                        'var': var,
                        'anom_type': anom_type,
                        'model': m,
                        'filepath_matcher': modelname_filepath_matcher
                    })

            else:
                # full anom types, loop through all models
                anom_types = ['raw', 'annmean', 'trend']

                for m in CMIPConfig.iter_active_models(var):
                    for anom_type in anom_types:
                        all_tasks.append({
                            'args': args,
                            'var': var,
                            'anom_type': anom_type,
                            'model': m,
                            'filepath_matcher': modelname_filepath_matcher
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
    logger = setup_logger()
    logger.debug(f"[Rank {rank}] logger initalized successfully")

    # Distribute tasks using round-robin distribution
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    logger.info(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    ## each task runs this independently, since it is embarrassingly parallelizable
    my_results = []  # "my" refers to the task that's running this -- it's different for each one
    
    # loop through tasks for this rank and perform operations...
    for task_idx, task in enumerate(my_tasks):
        logger.info(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}|{task['model'].name}|{task['anom_type']}")
        
        result = process_single_fit(
            logger=logger,
            args=task['args'],
            var=task['var'],
            anom_type=task['anom_type'],
            m=task['model'],
            modelname_filepath_matcher=task['filepath_matcher'],
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

        print_summary(flat_results, logger)
        logger.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"Average time per task: {elapsed/len(flat_results):.2f} seconds")

if __name__ == "__main__":
    main()