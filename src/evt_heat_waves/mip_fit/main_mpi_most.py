"""Main file for GEV fitting of CMIP model with most members - MPI parallelized.

Adam Michael Bauer
UChicago
Jan 2026

Each variable-fit combination runs as an independent MPI task.
To run: 
    srun -n <ntasks> python main_cmip_allmembers_mpi.py STAT

Last edited: 1/29/2026
"""

import os
import shutil
import time

from mpi4py import MPI

from evt_heat_waves.config import MIP_FIT_PATH_DICT
from evt_heat_waves.mip_fit.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.logging_utils import setup_logger, get_git_hash
from evt_heat_waves.mip_fit.cli import parse_args_mip_fit
from evt_heat_waves.mip_fit.most.most_mpi import process_single_fit, print_summary, combine_results_into_datasets, find_model_with_most_members

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 does initial setup
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
        anom_types = ['raw', 'trend', 'annmean']
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            logger.info(f"Setting up tasks for: {var}")
            
            # Make data directory if it doesn't exist
            os.makedirs(head_data_path / 'CMIP6' / var / 'gev', exist_ok=True)
            data_path = head_data_path / 'CMIP6' / var / 'landonly'
            
            # Find model with most members
            model_with_most, fpath, n_members, tied_models = find_model_with_most_members(
                var, CMIPConfig, data_path
            )
            
            message = (f"Identified {model_with_most} as model with most ensemble "
                      f"members (has {n_members} members).")
            logger.info(message)
            
            if tied_models:
                logger.warning(f"Note: This model was tied with {tied_models}!")

            # Create one task for each fit type and member for this variable
            for anom_type in anom_types:
               for mem in CMIPConfig.ensemble_config[model_with_most].ensemble_members:
                    all_tasks.append({
                        'args': args,
                        'var': var,
                        'anom_type': anom_type,
                        'model_with_most': model_with_most,
                        'mem': mem,
                        'fpath': fpath,
                        'rank': rank
                    })
        
        logger.info(f"Total tasks to process: {len(all_tasks)}")
        logger.info(f"Number of MPI processes: {size}")
        
        if len(all_tasks) <= size:
            logger.info(f"All {len(all_tasks)} tasks can run simultaneously!")
        else:
            logger.info(f"Tasks per process: ~{len(all_tasks) / size:.1f}")
        
    else:
        all_tasks = None
    
    # Broadcast tasks to all processes
    all_tasks = comm.bcast(all_tasks, root=0)
    logger = setup_logger(args.debug)
    logger.debug(f"[Rank {rank}] logger initalized successfully")
    
    # Distribute tasks using round-robin distribution
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    if len(my_tasks) > 0:
        logger.info(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    else:
        logger.info(f"[Rank {rank}] No tasks assigned (more processes than tasks)")
    
    # Process assigned tasks
    my_results = []
    for task_idx, task in enumerate(my_tasks):
        logger.info(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['anom_type']}:{task['mem']}")
        
        result = process_single_fit(
            logger=logger,
            args=task['args'],
            var=task['var'],
            anom_type=task['anom_type'],
            model_with_most=task['model_with_most'],
            mem=task['mem'],
            fpath=task['fpath'],
            rank=rank
        )
        my_results.append(result)
    
    # Gather all results to rank 0
    all_results = comm.gather(my_results, root=0)
    
    # Rank 0 combines results and prints summary
    if rank == 0:
        # Flatten results
        flat_results = [item for sublist in all_results for item in sublist]
        
        # Combine successful results into datasets
        successes = sum(1 for r in flat_results if r[0])
        if successes > 0:
            output_paths = combine_results_into_datasets(
                logger, args, flat_results, model_with_most, fpath
            )
        else:
            output_paths = {}
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print_summary(flat_results, output_paths, elapsed, logger)

if __name__ == '__main__':
    main()