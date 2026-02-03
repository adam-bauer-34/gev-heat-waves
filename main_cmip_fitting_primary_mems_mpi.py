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
import sys
import shutil
import time

import xarray as xr
from mpi4py import MPI

from config import DATA_ROOT
from src.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate
from src.cmip_dataclass import CMIP6EnsembleConfig
from src.utils import extract_model_name


def process_single_fit(var, m, modelname_filepath_matcher, STAT, fit_type, width, rank):
    """
    Process a single fit for a single model-variable combination.
    
    Parameters
    ----------
    var : str
        Variable name
    m : Model object
        Model configuration object
    modelname_filepath_matcher : dict
        Dictionary mapping model names to file paths
    STAT : str
        'stat' or 'nonstat' for stationary/nonstationary fitting
    fit_type : str
        One of 'raw', 'annmean', or 'trend'
    width : int
        Terminal width for formatting
    rank : int
        MPI rank of current process
        
    Returns
    -------
    tuple
        (success, fit_type, output_path, error_message)
    """
    try:
        print(f"[Rank {rank}] 🪛 Working on {var}:{m.name} - {fit_type} fit")
        
        fpath = modelname_filepath_matcher[m.name]
        ds = xr.open_dataset(fpath)
        ds_selected = ds.sel(member_id=m.primary_member)
        
        non_stat = (STAT == 'nonstat')
        
        # Determine which fit to perform
        if fit_type == 'raw':
            # print(f"[Rank {rank}] 🥩 Doing {'non-stationary' if non_stat else 'stationary'} GEV fit on raw temperature data...")
            ds_fit = ds_mle_fit(
                ds_selected, 
                var_name='tas', 
                fit_dim='year',
                non_stat=non_stat
            )
            var_suffix = 'raw'
            
        elif fit_type == 'annmean':
            # print(f"[Rank {rank}] ⚡️ Doing {'non-stationary' if non_stat else 'stationary'} GEV fit on temp anomalies (annual mean)...")
            ds_fit = ds_mle_fit(
                ds_selected, 
                var_name='t2m_anom_annmean', 
                fit_dim='year',
                non_stat=non_stat
            )
            var_suffix = 'annmean'
            
        elif fit_type == 'trend':
            # print(f"[Rank {rank}] ⚡️ Doing {'non-stationary' if non_stat else 'stationary'} GEV fit on temp anomalies (trend)...")
            ds_fit = ds_mle_fit(
                ds_selected, 
                var_name='t2m_anom_trend', 
                fit_dim='year',
                non_stat=non_stat
            )
            var_suffix = 'trend'
        else:
            raise ValueError(f"Unknown fit_type: {fit_type}")
        
        print(f"[Rank {rank}] ✅ {fit_type} fit complete.")

        # get MLE success rate; reset immediately after
        success_rate = get_mle_success_rate()        
        reset_mle_stats()

        # store MLE success rate as a dataset attribute
        ds_fit.attrs['MLE_success_rate'] = success_rate
        
        # Save dataset
        gev_dir = fpath.parent.parent / 'gev'
        gev_name = fpath.with_name(
            fpath.stem + f"_gev_{STAT}_{var_suffix}" + fpath.suffix
        ).name
        
        output_path = gev_dir / gev_name
        ds_fit.to_netcdf(output_path)
        print(f"[Rank {rank}] ✍️ Dataset saved to: {output_path}")
        
        # Close datasets to save RAM
        ds_fit.close()
        ds.close()
        
        return (True, fit_type, str(output_path), None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {var}:{m.name}:{fit_type} - {str(e)}\n{traceback.format_exc()}"
        print(f"[Rank {rank}] ❌ {error_msg}")
        return (False, fit_type, None, error_msg)


def main():
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
    
    STAT = sys.argv[1]
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    
    # Only rank 0 does initial setup and task distribution
    ## rank 0 does all the initial setup and distribution because I/O operations
    ## like reading .yaml files and so on don't parallelize well
    if rank == 0:
        start_time = time.time()
        print('='*width)
        print(f"🚀 Starting MPI parallel processing with {size} processes")
        print('='*width)
        
        # Setup CMIP config object
        CMIPConfig = CMIP6EnsembleConfig.from_yaml("config/meta.yaml", 
                                                    "config/qc.yaml")
        
        # Define variables and fit types
        vars = ['tas_annual_max', 'tas_annual_min']
        fit_types = ['raw', 'annmean', 'trend']
        
        # Collect all tasks (each fit is now a separate task)
        all_tasks = []
        
        for var in vars:
            print('='*width)
            print(f"🏋🏼‍♀️ Setting up tasks for: {var}")
            print('='*width)
            
            # Make data directory if it doesn't exist
            os.makedirs(DATA_ROOT / 'CMIP6' / var / 'gev', exist_ok=True)
            data_path = DATA_ROOT / 'CMIP6' / var / 'landonly'
            
            # Make all landonly file names
            fnames = [f for f in data_path.glob("*_landonly.nc")]
            modelname_filepath_matcher = {
                extract_model_name(f): f for f in fnames
            }
            
            # Collect tasks for this variable
            # Now we create 3 tasks per model (one for each fit type)
            for m in CMIPConfig.iter_active_models(var):
                for fit_type in fit_types:
                    all_tasks.append({
                        'var': var,
                        'model': m,
                        'filepath_matcher': modelname_filepath_matcher,
                        'STAT': STAT,
                        'fit_type': fit_type,
                        'width': width
                    })
        
        print(f"\n📋 Total tasks to process: {len(all_tasks)}")
        print(f"   ({len(all_tasks)//3} models × 3 fits per model)")
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
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    print(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    ## each task runs this independently, since it is embarrassingly parallelizable
    my_results = []  # "my" refers to the task that's running this -- it's different for each one
    
    # loop through tasks for this rank and perform operations...
    for task_idx, task in enumerate(my_tasks):
        print(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['model'].name}:{task['fit_type']}")
        
        result = process_single_fit(
            var=task['var'],
            m=task['model'],
            modelname_filepath_matcher=task['filepath_matcher'],
            STAT=task['STAT'],
            fit_type=task['fit_type'],
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
    main()