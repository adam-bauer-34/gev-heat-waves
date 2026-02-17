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


def process_single_fit(var, model_with_most, mem, fpath, STAT, fit_type, width, rank):
    """
    Process a single fit for a single variable on the model with most members.
    
    Parameters
    ----------
    var : str
        Variable name
    model_with_most : str
        Name of model with most ensemble members
    mem : str
        Ensemble member identifier (e.g., 'r1i1p1f3')
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
        (success, var, fit_type, mem, data_dict, coords_dict, attrs_dict, error_message)
        where data_dict is a dictionary of {var_name: numpy_array}
    """
    try:
        print(f"[Rank {rank}] " + '-'*width)
        print(f"[Rank {rank}] 🪛 Working on {var}:{model_with_most}:{mem} - {fit_type} fit")
        print(f"[Rank {rank}] " + '-'*width)
        
        # Open dataset
        ds = xr.open_dataset(fpath).sel(member_id=mem)
        
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
            
        elif fit_type == 'annmean':
            print(f"[Rank {rank}] ⚡️ Doing {stat_type} GEV fits on temp anomalies (annual mean)...")
            ds_fit = ds_mle_fit(
                ds, 
                var_name='t2m_anom_annmean', 
                fit_dim='year',
                non_stat=non_stat,
                all_mems=True
            )
            
        elif fit_type == 'trend':
            print(f"[Rank {rank}] ⚡️ Doing {stat_type} GEV fits on temp anomalies (trend)...")
            ds_fit = ds_mle_fit(
                ds, 
                var_name='t2m_anom_trend', 
                fit_dim='year',
                non_stat=non_stat,
                all_mems=True
            )
        else:
            raise ValueError(f"Unknown fit_type: {fit_type}")
        
        reset_mle_stats()
        
        print(f"[Rank {rank}] ✅ {fit_type} fit complete for {var}:{mem}")

        if STAT == 'nonstat':
            if fit_type == 'raw':
                gev_param_names = [
                    f'loc_{fit_type}', 
                    f'loc_t_{fit_type}',
                    f'scale_{fit_type}', 
                    f'scale_t_{fit_type}', 
                    f'shape_{fit_type}', 
                    f'shape_t_{fit_type}'
                ]

            else:
                gev_param_names = [
                    f'loc_anom_{fit_type}',
                    f'loc_t_anom_{fit_type}',
                    f'scale_anom_{fit_type}',
                    f'scale_t_anom_{fit_type}',
                    f'shape_anom_{fit_type}',
                    f'shape_t_anom_{fit_type}'
                ]
        
        else:
            if fit_type == 'raw':
                gev_param_names = [
                    f'loc_{fit_type}', 
                    f'scale_{fit_type}', 
                    f'shape_{fit_type}'
                ]

            else:
                gev_param_names = [
                    f'loc_anom_{fit_type}',
                    f'scale_anom_{fit_type}',
                    f'shape_anom_{fit_type}'
                ]
        
        # Convert dataset to dictionary of arrays
        data_dict = {}
        for var_name in gev_param_names:
            data_dict[var_name] = ds_fit[var_name].values
        
        # Also store coordinates (except member_id which we'll add later)
        coords_dict = {}
        for coord_name in ds_fit.coords:
            if coord_name != 'member_id':
                coords_dict[coord_name] = ds_fit[coord_name].values
        
        # Store attributes
        attrs_dict = dict(ds_fit.attrs)
        
        # Close datasets to save RAM
        ds_fit.close()
        ds.close()
        
        return (True, var, fit_type, mem, data_dict, coords_dict, attrs_dict, None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {var}:{fit_type}:{mem} - {str(e)}\n{traceback.format_exc()}"
        print(f"[Rank {rank}] ❌ {error_msg}")
        return (False, var, fit_type, mem, None, None, None, error_msg)


def combine_results_into_datasets(all_results, model_with_most, STAT, data_path_root, width):
    """
    Combine results from all workers into datasets organized by variable and fit_type.
    
    Parameters
    ----------
    all_results : list
        Flattened list of results from all workers
    model_with_most : str
        the name of the climate model we used for fitting
    STAT : str
        'stat' or 'nonstat'
    data_path_root : Path
        Root data path
    width : int
        Terminal width for formatting
        
    Returns
    -------
    dict
        Dictionary mapping (var, fit_type) to output file path
    """
    print('='*width)
    print("📦 Combining results into datasets...")
    print('='*width)
    
    # Group results by (var, fit_type)
    grouped = {}
    for result in all_results:
        if result[0]:  # success
            var = result[1]
            fit_type = result[2]
            mem = result[3]
            data_dict = result[4]
            coords_dict = result[5]
            attrs_dict = result[6]
            
            key = (var, fit_type)
            if key not in grouped:
                grouped[key] = {
                    'members': [],
                    'data_dicts': [],
                    'coords_dict': coords_dict,  # Same for all members
                    'attrs_dict': attrs_dict      # Same for all members
                }
            
            grouped[key]['members'].append(mem)
            grouped[key]['data_dicts'].append(data_dict)
    
    # Create and save datasets
    output_paths = {}
    
    for (var, fit_type), group_data in grouped.items():
        print(f"\n🔨 Creating dataset for {var}:{fit_type}")
        
        members = group_data['members']
        data_dicts = group_data['data_dicts']
        coords_dict = group_data['coords_dict']
        attrs_dict = group_data['attrs_dict']
        
        # Determine var_suffix for filename
        if fit_type == 'raw':
            var_suffix = 'raw'
        elif fit_type == 'annmean':
            var_suffix = 'annmean'
        elif fit_type == 'trend':
            var_suffix = 'trend'
        
        # Create data_vars dictionary with member_id dimension
        data_vars = {}
        
        # Get variable names from first data_dict
        var_names = list(data_dicts[0].keys())
        
        for var_name in var_names:
            # Stack arrays along new member_id dimension
            arrays = [d[var_name] for d in data_dicts]
            stacked = np.stack(arrays, axis=0)
            
            # Create dimensions tuple
            # Assuming original dimensions are like ('lat', 'lon') or ('year', 'lat', 'lon')
            # The new dimension will be ('member_id', original_dims...)
            original_shape = arrays[0].shape
            
            # Infer dimension names from coords_dict or use generic names
            if len(original_shape) == 2:
                dims = ('member_id', 'lat', 'lon')
            elif len(original_shape) == 3:
                dims = ('member_id', 'year', 'lat', 'lon')
            else:
                # Generic dimension names
                dims = ('member_id',) + tuple(f'dim_{i}' for i in range(len(original_shape)))
            
            data_vars[var_name] = (dims, stacked)
        
        # Create coords dictionary with member_id added
        coords = {'member_id': members}
        for coord_name, coord_values in coords_dict.items():
            coords[coord_name] = coord_values
        
        # Create dataset
        ds_combined = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs_dict
        )
        
        print(f"   Dataset shape: {ds_combined.dims}")
        print(f"   Members: {len(members)}")
        
        # Determine output path
        gev_dir = data_path_root / 'CMIP6' / var / 'gev'
        os.makedirs(gev_dir, exist_ok=True)
        
        output_filename = f"tas_CMIP6_{model_with_most}_hist+ssp585_1979-2024_landonly_gev_{STAT}_allmems_{var_suffix}.nc"
        output_path = gev_dir / output_filename
        
        # Save dataset
        print(f"   💾 Saving to: {output_path}")
        ds_combined.to_netcdf(output_path)
        
        output_paths[key] = str(output_path)
        
        # Close dataset
        ds_combined.close()
    
    return output_paths


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
        print(f"🎯 Strategy: Parallelizing over variables, fit types, and members")
        print('='*width)
        
        # Setup CMIP config object
        CMIPConfig = CMIP6EnsembleConfig.from_yaml("config/meta.yaml", 
                                                    "config/qc.yaml")
        
        # Define variables and fit types
        vars = ['tas_annual_max', 'tas_annual_min']
        fit_types = ['raw', 'annmean', 'trend']
        
        # Collect all tasks (each variable-fit-member combination is a task)
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
            
            # Create one task for each fit type and member for this variable
            for fit_type in fit_types:
                for mem in CMIPConfig.ensemble_config[model_with_most].ensemble_members:
                    all_tasks.append({
                        'var': var,
                        'model_with_most': model_with_most,
                        'mem': mem,
                        'fpath': fpath,
                        'STAT': STAT,
                        'fit_type': fit_type,
                        'width': width
                    })
        
        print(f"\n📋 Total tasks to process: {len(all_tasks)}")
        print(f"   ({len(vars)} variables × {len(fit_types)} fit types × ~{len(all_tasks)//(len(vars)*len(fit_types))} members)")
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
              f"{task['var']}:{task['fit_type']}:{task['mem']}")
        
        result = process_single_fit(
            var=task['var'],
            model_with_most=task['model_with_most'],
            mem=task['mem'],
            fpath=task['fpath'],
            STAT=task['STAT'],
            fit_type=task['fit_type'],
            width=task['width'],
            rank=rank
        )
        my_results.append(result)
    
    # Gather all results to rank 0
    all_results = comm.gather(my_results, root=0)
    
    # Rank 0 combines results and prints summary
    if rank == 0:
        # Flatten results
        flat_results = [item for sublist in all_results for item in sublist]
        
        successes = sum(1 for r in flat_results if r[0])
        failures = sum(1 for r in flat_results if not r[0])
        
        # Combine successful results into datasets
        if successes > 0:
            output_paths = combine_results_into_datasets(
                flat_results, model_with_most, STAT, DATA_ROOT, width
            )
        else:
            output_paths = {}
        
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
            status = "✅" if counts['success'] == total else "⚠️"
            print(f"  {status} {key:30s}: {counts['success']}/{total}")
            if key in [(k[0] + ':' + k[1]) for k in output_paths.keys()]:
                var_key, fit_key = key.split(':')
                path = output_paths.get((var_key, fit_key), 'N/A')
                print(f"      📁 Output: {path}")
        
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"⚡ Average time per task: {elapsed/len(flat_results):.2f} seconds")
        
        if failures > 0:
            print("\n❌ Failed tasks:")
            for r in flat_results:
                if not r[0]:
                    print(f"   - {r[1]}:{r[2]}:{r[3]}")
                    if r[7]:  # error message
                        print(f"     Error: {r[7][:200]}...")  # Truncate long errors
        
        print('='*width)
        print("🥳 All done! 🥳")
        print('='*width)


if __name__ == '__main__':
    main()