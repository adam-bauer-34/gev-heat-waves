"""Function bank for MPI-enhanced GEV fitting to CMIP data.

Adam Michael Bauer
UChicago
Apr 8 2026
"""

import os
import shutil
import time

import xarray as xr
from mpi4py import MPI

from evt_heat_waves.config import MIP_FIT_PATH_DICT, CONFIG_PATH
from evt_heat_waves.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.utils import extract_model_name
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate

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
        
        # Setup CMIP config object
        CMIPConfig = CMIP6EnsembleConfig.from_yaml(CONFIG_PATH.parent / "meta.yaml", 
                                                   CONFIG_PATH.parent / "qc.yaml")
        
        # Define variables and fit types
        vars = ['tas_annual_max', 'tas_annual_min']
        anom_types = ['raw', 'annmean', 'trend']
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            logger.info(f"Setting up tasks for: {var}")
            
            # Make data directory if it doesn't exist
            os.makedirs(MIP_FIT_PATH_DICT[args.data] / var / 'gev', exist_ok=True)
            data_path = MIP_FIT_PATH_DICT[args.data] / var / 'landonly'
            
            # Make all landonly file names
            fnames = [f for f in data_path.glob("*_landonly.nc")]
            modelname_filepath_matcher = {
                extract_model_name(f): f for f in fnames
            }
            
            # Collect tasks for this variable
            # Now we create 3 tasks per model (one for each fit type)
            for m in CMIPConfig.iter_active_models(var):
                for anom_type in anom_types:
                    all_tasks.append({
                        'logger': logger,
                        'args': args,
                        'var': var,
                        'anom_type': anom_type,
                        'model': m,
                        'filepath_matcher': modelname_filepath_matcher,
                        'width': width
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
    
    # Distribute tasks using round-robin distribution
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    logger.info(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    ## each task runs this independently, since it is embarrassingly parallelizable
    my_results = []  # "my" refers to the task that's running this -- it's different for each one
    
    # loop through tasks for this rank and perform operations...
    for task_idx, task in enumerate(my_tasks):
        logger.info(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['model'].name}:{task['fit_type']}")
        
        result = process_single_fit(
            logger=task['logger'],
            args=task['args'],
            var=task['var'],
            anom_type=task['anom_type']
            m=task['model'],
            modelname_filepath_matcher=task['filepath_matcher'],
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
        
        logger.info("-" * width)
        logger.info("SUMMARY")
        logger.info('-' *width)
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


def process_single_fit(logger, args, var, anom_type, m, modelname_filepath_matcher, rank):
    """
    Process a single fit for a single model-variable combination.
    
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
        logger.info(f"[Rank {rank}] 🪛 Working on {var}:{m.name} - {anom_type} fit")
        
        fpath = modelname_filepath_matcher[m.name]
        ds = xr.open_dataset(fpath)
        ds_selected = ds.sel(member_id=m.primary_member)
        
        # Determine which fit to perform
        if anom_type == 'raw':
            ds_fit = ds_mle_fit(
                logger,
                args,
                ds_selected, 
                var_name='tas',
                fit_dim='year',
            )
            var_suffix = 'raw'
            
        elif anom_type == 'annmean':
            ds_fit = ds_mle_fit(
                logger,
                args,
                ds_selected, 
                var_name='t2m_anom_annmean', 
                fit_dim='year',
            )
            var_suffix = 'annmean'
            
        elif anom_type == 'trend':
            ds_fit = ds_mle_fit(
                logger,
                args,
                ds_selected, 
                var_name='t2m_anom_trend', 
                fit_dim='year',
            )
            var_suffix = 'trend'
        else:
            raise ValueError(f"Unknown fit_type: {anom_type}")
        
        logger.info(f"[Rank {rank}] {anom_type} fit complete.")

        # get MLE success rate; reset immediately after
        success_rate = get_mle_success_rate()        
        reset_mle_stats()

        # store MLE success rate as a dataset attribute
        ds_fit.attrs['MLE_success_rate'] = success_rate
        
        # Save dataset
        gev_dir = fpath.parent.parent / 'gev'
        gev_name = fpath.with_name(
            fpath.stem + f"_gev_{args.fit}_{var_suffix}" + fpath.suffix
        ).name
        
        output_path = gev_dir / gev_name
        ds_fit.to_netcdf(output_path)
        logger.info(f"[Rank {rank}] Dataset saved to: {output_path}")
        
        # Close datasets to save RAM
        ds_fit.close()
        ds.close()
        
        return (True, anom_type, str(output_path), None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {var}:{m.name}:{anom_type} - {str(e)}\n{traceback.format_exc()}"
        logger.warning(f"[Rank {rank}] ❌ {error_msg}")
        return (False, anom_type, None, error_msg)
