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

import xarray as xr
import numpy as np

from evt_heat_waves.config import MIP_FIT_PATH_DICT, ANOM_TYPE_TO_VAR, MLE_FIT_ATTRS
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats
from evt_heat_waves.utils import extract_model_name

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def process_single_fit(logger, args, var, anom_type, model_with_most, mem, fpath, rank):
    """
    Process a single fit for a single variable on the model with most members.
    
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

    model_with_most : str
        Name of model with most ensemble members

    mem : str
        Ensemble member identifier (e.g., 'r1i1p1f3')

    fpath : Path
        File path to the dataset

    rank : int
        MPI rank of current process
        
    Returns
    -------
    tuple
        (success, var, anom_type, mem, data_dict, coords_dict, attrs_dict, error_message)
        where data_dict is a dictionary of {var_name: numpy_array}
    """
    
    try:
        logger.info(f"[Rank {rank}] Working on {var}:{model_with_most}:{mem} - {anom_type} fit")

        # mapping for data -> variable name in dataset
        try:
            var_name = ANOM_TYPE_TO_VAR[anom_type]
        except KeyError:
            raise ValueError(f"Unknown anom_type: {anom_type}")
        
        # Open dataset
        ds = xr.open_dataset(fpath).sel(member_id=mem)
        
        # do fitting
        ds_fit = ds_mle_fit(
            args,
            ds, 
            var_name=var_name,
            fit_dim='year',
            all_mems=True
        )
        logger.info(f"[Rank {rank}] {anom_type} fit complete for {var}:{mem}")
        
        # reset stats
        reset_mle_stats()

        # build dataset variable names from config param_names + suffix
        # suffix depends on whether this is a raw or anomaly fit
        sfx = anom_type if anom_type == 'raw' else f'anom_{anom_type}'
        gev_param_names = [
            f'{p}_{sfx}' for p in MLE_FIT_ATTRS[args.fit]['param_names']
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
        
        return (True, var, anom_type, mem, data_dict, coords_dict, attrs_dict, None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {var}:{anom_type}:{mem} - {str(e)}\n{traceback.format_exc()}"
        logger.info(f"[Rank {rank}] {error_msg}")
        return (False, var, anom_type, mem, None, None, None, error_msg)


def combine_results_into_datasets(logger, args, all_results, model_with_most, fpath):
    """Combine results from all workers into datasets organized by variable and fit_type.
    
    Parameters
    ----------
    logger: logging.Logger
        for printing

    all_results : list
        Flattened list of results from all workers

    model_with_most : str
        the name of the climate model we used for fitting

    fpath : Path
        path to data

    Returns
    -------
    dict
        Dictionary mapping (var, fit_type) to output file path
    """
    logger.info("Combining results into datasets...")
    
    # Group results by (var, fit_type)
    grouped = {}
    for result in all_results:
        if result[0]:  # success
            var = result[1]
            anom_type = result[2]
            mem = result[3]
            data_dict = result[4]
            coords_dict = result[5]
            attrs_dict = result[6]
            
            key = (var, anom_type)
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
    
    for (var, anom_type), group_data in grouped.items():
        logger.info(f"Creating dataset for {var}:{anom_type}")
        
        members = group_data['members']
        data_dicts = group_data['data_dicts']
        coords_dict = group_data['coords_dict']
        attrs_dict = group_data['attrs_dict']
        
        # Create data_vars dictionary with member_id dimension
        data_vars = {}
        
        # Get variable names from first data_dict
        var_names = list(data_dicts[0].keys())
        
        for var_name in var_names:
            # Stack arrays along new member_id dimension
            arrays = [d[var_name] for d in data_dicts]
            stacked = np.stack(arrays, axis=0)
            
            # Create dimensions tuple
            dims = ('member_id', 'lat', 'lon')

            # set data variable with member_id dimension            
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
        
        logger.info(f"   Dataset shape: {ds_combined.dims}")
        logger.info(f"   Members: {len(members)}")
        
        # Determine output path
        try:
            head_data_path = MIP_FIT_PATH_DICT[args.data]['data']
        except Exception as e:
            raise KeyError(f"Error in data config for GEV fitting: {str(e)}")
        
        gev_dir = head_data_path / var / 'gev' if not args.debug else head_data_path / var / 'gev_debug'
        os.makedirs(gev_dir, exist_ok=True)
        fname = f"{fpath.stem}_gev_{args.fit}_allmems_{anom_type}.nc"
        output_path = gev_dir / fname
        
        # Save dataset
        logger.info(f"   Saving to: {output_path}")
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


def print_summary(flat_results, output_paths, elapsed, logger):
    """Print a summary of MPI task results for most-members fitting.
    
    Parameters
    ----------
    flat_results : list[tuple]
        List of (success, var, anom_type, mem, data_dict, coords_dict, attrs_dict, error_message) tuples.
    output_paths : dict
        Dictionary mapping (var, fit_type) to output file path.
    elapsed : float
        Total elapsed time in seconds.
    logger : logging.Logger
        Logger used to report summary information.
    """
    successes = sum(1 for r in flat_results if r[0])
    failures = len(flat_results) - successes
    
    # Count by variable and fit type
    breakdown = {}
    for success, var, fit_type, *_ in flat_results:
        key = f"{var}:{fit_type}"
        if key not in breakdown:
            breakdown[key] = {'success': 0, 'failure': 0}
        breakdown[key]['success' if success else 'failure'] += 1
    
    logger.info('-' * width)
    logger.info("SUMMARY")
    logger.info('-' * width)
    logger.info(f"Successful: {successes}/{len(flat_results)}")
    logger.info(f"Failed: {failures}/{len(flat_results)}")
    logger.info("Breakdown by variable and fit type:")
    for key, counts in sorted(breakdown.items()):
        total = counts['success'] + counts['failure']
        logger.info(f"  - {key:30s}: {counts['success']}/{total}")
        if key in [(k[0] + ':' + k[1]) for k in output_paths.keys()]:
            var_key, fit_key = key.split(':')
            path = output_paths.get((var_key, fit_key), 'N/A')
            logger.info(f"  - Output: {path}")
    
    logger.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"Average time per task: {elapsed/len(flat_results):.2f} seconds")
    
    if failures > 0:
        logger.info("Failed tasks:")
        for r in flat_results:
            if not r[0]:
                logger.info(f"  - {r[1]}:{r[2]}:{r[3]}")
                if r[7]:  # error message
                    logger.info(f"  - Error: {r[7][:200]}...")  # Truncate long errors
    
    logger.info("-" * width)