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
import sys
import shutil
import time

import xarray as xr
import numpy as np
from mpi4py import MPI

from config import DATA_ROOT
from src.mle import ds_mle_fit, reset_mle_stats
from src.cmip_dataclass import CMIP6EnsembleConfig
from src.utils import extract_model_name


def process_single_fit(var, model_with_most, fpath, STAT, fit_type, width, rank):
    """
    Process a single fit for a single variable on the model with most members.
    
    Parameters
    ----------
    var : str
        Variable name
    model_with_most : str
        Name of model with most ensemble members
    fpath : Path
        File path to the dataset
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
        (success, var, fit_type, output_path, error_message)
    """
    try:
        print(f"[Rank {rank}] " + '-'*width)
        print(f"[Rank {rank}] 🪛 Working on {var}:{model_with_most} - {fit_type} fit (all members)")
        print(f"[Rank {rank}] " + '-'*width)
        
        # Open dataset
        ds = xr.open_dataset(fpath)
        
        non_stat = (STAT == 'nonstat')
        stat_type = 'non-stationary' if non_stat else 'stationary'
        
        # Determine which fit to perform
        if fit_type == 'raw':
            print(f"[Rank {rank}] 🥩 Doing {stat_type} GEV fits on raw temperature data...")
            ds_fit = ds_mle_fit(
                ds, 
                var_name='tas', 
                fit_dim='year',
                non_stat=non_stat,
                all_mems=True
            )
            var_suffix = 'raw'
            
        elif fit_type == 'annmean':
            print(f"[Rank {rank}] ⚡️ Doing {stat_type} GEV fits on temp anomalies (annual mean)...")
            ds_fit = ds_mle_fit(
                ds, 
                var_name='t2m_anom_annmean', 
                fit_dim='year',
                non_stat=non_stat,
                all_mems=True
            )
            var_suffix = 'annmean'
            
        elif fit_type == 'trend':
            print(f"[Rank {rank}] ⚡️ Doing {stat_type} GEV fits on temp anomalies (trend)...")
            ds_fit = ds_mle_fit(
                ds, 
                var_name='t2m_anom_trend', 
                fit_dim='year',
                non_stat=non_stat,
                all_mems=True
            )
            var_suffix = 'trend'
        else:
            raise ValueError(f"Unknown fit_type: {fit_type}")
        
        reset_mle_stats()
        
        print(f"[Rank {rank}] ✅ {fit_type} fit complete for {var}")
        
        # Save dataset
        gev_dir = fpath.parent.parent / 'gev'
        gev_name = fpath.with_name(
            fpath.stem + f"_gev_{STAT}_allmems_{var_suffix}" + fpath.suffix
        ).name
        
        output_path = gev_dir / gev_name
        ds_fit.to_netcdf(output_path)
        print(f"[Rank {rank}] ✍️ Dataset saved to: {output_path}")
        
        # Close datasets to save RAM
        ds_fit.close()
        ds.close()
        
        return (True, var, fit_type, str(output_path), None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {var}:{fit_type} - {str(e)}\n{traceback.format_exc()}"
        print(f"[Rank {rank}] ❌ {error_msg}")
        return (False, var, fit_type, None, error_msg)


def find_model_with_most_members(var, CMIPConfig, data_path):
    """
    Find the model with the most ensemble members for a given variable.
    
    Parameters
    ----------
    var : str
        Variable name
    CMIPConfig : CMIP6EnsembleConfig
        CMIP configuration object
    data_path : Path
        Path to data directory
        
    Returns
    -------
    tuple
        (model_name, filepath, n_members, tied_models)
    """
    # Make all landonly file names
    fnames = [f for f in data_path.glob("*_landonly.nc")]
    modelname_filepath_matcher = {
        extract_model_name(f): f for f in fnames
    }
    
    # Calculate number of ensemble members for each active model
    Nens_for_active_models = np.array([
        len(m.all_members) for m in CMIPConfig.iter_active_models(var)
    ])
    
    # Find model(s) with most members
    max_inds = np.where(Nens_for_active_models == np.max(Nens_for_active_models))[0]
    
    # Check for ties
    tied_models = None
    if len(max_inds) > 1:
        all_model_names = np.array(
            [m.name for m in CMIPConfig.iter_active_models(var)]
        )
        tied_models = all_model_names[max_inds[1:]].tolist()
    
    # Select first model with most members
    ind_ = max_inds[0]
    model_with_most = [m.name for m in CMIPConfig.iter_active_models(var)][ind_]
    n_members = np.max(Nens_for_active_models)
    fpath = modelname_filepath_matcher[model_with_most]
    
    return model_with_most, fpath, n_members, tied_models


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get command line arguments
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: srun -n <nprocs> python main_cmip_allmembers_mpi.py STAT")
            print("where STAT is 'stat' or 'nonstat'")
        sys.exit(1)
    
    STAT = sys.argv[1]
    
    if STAT not in ['stat', 'nonstat']:
        if rank == 0:
            print("⚠️ Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
        sys.exit(1)
    
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    
    # Only rank 0 does initial setup
    if rank == 0:
        start_time = time.time()
        print('='*width)
        print(f"🚀 Starting MPI parallel processing with {size} processes")
        print(f"🎯 Strategy: Parallelizing over variables and fit types")
        print('='*width)
        
        # Setup CMIP config object
        CMIPConfig = CMIP6EnsembleConfig.from_yaml("config/meta.yaml", 
                                                    "config/qc.yaml")
        
        # Define variables and fit types
        vars = ['tas_annual_max', 'tas_annual_min']
        fit_types = ['raw', 'annmean', 'trend']
        
        # Collect all tasks (each variable-fit combination is a task)
        all_tasks = []
        
        for var in vars:
            print('='*width)
            print(f"🏋🏼‍♀️ Setting up tasks for: {var}")
            print('='*width)
            
            # Make data directory if it doesn't exist
            os.makedirs(DATA_ROOT / 'CMIP6' / var / 'gev', exist_ok=True)
            data_path = DATA_ROOT / 'CMIP6' / var / 'landonly'
            
            # Find model with most members
            model_with_most, fpath, n_members, tied_models = find_model_with_most_members(
                var, CMIPConfig, data_path
            )
            
            message = (f"🪛 Identified {model_with_most} as model with most ensemble "
                      f"members (has {n_members} members).")
            print(message)
            
            if tied_models:
                print(f"Note: This model was tied with {tied_models}!")
            
            # Create one task for each fit type for this variable
            for fit_type in fit_types:
                all_tasks.append({
                    'var': var,
                    'model_with_most': model_with_most,
                    'fpath': fpath,
                    'STAT': STAT,
                    'fit_type': fit_type,
                    'width': width
                })
        
        print(f"\n📋 Total tasks to process: {len(all_tasks)}")
        print(f"   ({len(vars)} variables × {len(fit_types)} fits per variable)")
        print(f"🖥️  Number of MPI processes: {size}")
        
        if len(all_tasks) <= size:
            print(f"✨ All {len(all_tasks)} tasks can run simultaneously!")
        else:
            print(f"📊 Tasks per process: ~{len(all_tasks) / size:.1f}")
        
        print('='*width)
    else:
        all_tasks = None
    
    # Broadcast tasks to all processes
    all_tasks = comm.bcast(all_tasks, root=0)
    
    # Distribute tasks using round-robin distribution
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    if len(my_tasks) > 0:
        print(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    else:
        print(f"[Rank {rank}] No tasks assigned (more processes than tasks)")
    
    # Process assigned tasks
    my_results = []
    for task_idx, task in enumerate(my_tasks):
        print(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['fit_type']}")
        
        result = process_single_fit(
            var=task['var'],
            model_with_most=task['model_with_most'],
            fpath=task['fpath'],
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
        
        successes = sum(1 for r in flat_results if r[0])
        failures = sum(1 for r in flat_results if not r[0])
        
        # Count by variable and fit type
        breakdown = {}
        for r in flat_results:
            var = r[1]
            fit_type = r[2]
            key = f"{var}:{fit_type}"
            if key not in breakdown:
                breakdown[key] = {'success': 0, 'failure': 0}
            if r[0]:
                breakdown[key]['success'] += 1
            else:
                breakdown[key]['failure'] += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print('='*width)
        print("📊 SUMMARY")
        print('='*width)
        print(f"✅ Successful: {successes}/{len(flat_results)}")
        print(f"❌ Failed: {failures}/{len(flat_results)}")
        print(f"\nBreakdown by variable and fit type:")
        for key, counts in sorted(breakdown.items()):
            total = counts['success'] + counts['failure']
            status = "✅" if counts['success'] == total else "❌"
            print(f"  {status} {key:30s}: {counts['success']}/{total}")
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"⚡ Average time per task: {elapsed/len(flat_results):.2f} seconds")
        
        if failures > 0:
            print("\n❌ Failed tasks:")
            for r in flat_results:
                if not r[0]:
                    print(f"   - {r[1]}:{r[2]}")
                    print(f"     Error: {r[4]}")
        
        print('='*width)
        print("🥳 All done! 🥳")
        print('='*width)


if __name__ == '__main__':
    main()