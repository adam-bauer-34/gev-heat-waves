"""Main file for GEV fitting of CMIP model with most members - serialized.

Adam Michael Bauer
UChicago
Jan 2026

Each variable-fit combination runs sequentially.
To run: 
    python main_cmip_allmembers_serial.py STAT

Last edited: 4/9/2026
"""

import os
import shutil
import time

import xarray as xr
import numpy as np

from evt_heat_waves.config import MIP_FIT_PATH_DICT, ANOM_TYPE_TO_VAR, MLE_FIT_ATTRS
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats
from evt_heat_waves.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.utils import extract_model_name

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def runner(logger, args):
    """Runner for GEV fitting of CMIP data - all members of the model with the most members.

    Parameters
    ----------
    logger: logging.Logger
        logger object
    
    args: argparse.Namespace
        CLI arguments for fit details
    """
    start_time = time.time()

    # Extract path info
    try:
        data_path = MIP_FIT_PATH_DICT[args.data]['data']
        config_meta_path = MIP_FIT_PATH_DICT[args.data]['config']['meta']
        config_qc_path = MIP_FIT_PATH_DICT[args.data]['config']['qc']
    except Exception as e:
        raise KeyError(f"Error in data config for GEV fitting: {str(e)}")

    # Setup CMIP config object
    CMIPConfig = CMIP6EnsembleConfig.from_yaml(config_meta_path, config_qc_path)

    vars = ['tas_annual_max', 'tas_annual_min']
    anom_types = ['raw', 'annmean', 'trend']

    # Collect all tasks
    all_tasks = []

    for var in vars:
        logger.info(f"Setting up tasks for: {var}")

        os.makedirs(data_path / var / 'gev', exist_ok=True)
        var_data_path = data_path / var / 'landonly'

        model_with_most, fpath, n_members, tied_models = find_model_with_most_members(
            var, CMIPConfig, var_data_path
        )

        message = (f"Identified {model_with_most} as model with most ensemble "
                   f"members (has {n_members} members).")
        logger.info(message)

        if tied_models:
            logger.warning(f"Note: This model was tied with {tied_models}!")

        for anom_type in anom_types:
            for mem in CMIPConfig.ensemble_config[model_with_most].ensemble_members:
                all_tasks.append({
                    'var': var,
                    'anom_type': anom_type,
                    'model_with_most': model_with_most,
                    'mem': mem,
                    'fpath': fpath,
                })

    logger.info(f"Total tasks to process: {len(all_tasks)}")

    # Process all tasks serially
    all_results = []
    for task_idx, task in enumerate(all_tasks):
        logger.info(f"Processing task {task_idx + 1}/{len(all_tasks)}: "
                    f"{task['var']}:{task['anom_type']}:{task['mem']}")

        result = process_single_fit(
            logger=logger,
            args=args,
            var=task['var'],
            anom_type=task['anom_type'],
            model_with_most=task['model_with_most'],
            mem=task['mem'],
            fpath=task['fpath'],
        )
        all_results.append(result)

    # Summarize and combine results
    successes = sum(1 for r in all_results if r[0])
    failures = sum(1 for r in all_results if not r[0])

    if successes > 0:
        # model_with_most and fpath are the same for all successful results within
        # a var, so pull them from the last task for use in combine_results
        output_paths = combine_results_into_datasets(
            logger, args, all_results, all_tasks[-1]['model_with_most'], all_tasks[-1]['fpath']
        )
    else:
        output_paths = {}

    breakdown = {}
    for r in all_results:
        var = r[1]
        anom_type = r[2]
        key = f"{var}:{anom_type}"
        if key not in breakdown:
            breakdown[key] = {'success': 0, 'failure': 0}
        if r[0]:
            breakdown[key]['success'] += 1
        else:
            breakdown[key]['failure'] += 1

    end_time = time.time()
    elapsed = end_time - start_time

    logger.info('-' * width)
    logger.info("SUMMARY")
    logger.info('-' * width)
    logger.info(f"    Successful: {successes}/{len(all_results)}")
    logger.info(f"    Failed: {failures}/{len(all_results)}")
    logger.info(f"    Breakdown by variable and fit type:")
    for key, counts in sorted(breakdown.items()):
        total = counts['success'] + counts['failure']
        logger.info(f"      - {key:30s}: {counts['success']}/{total}")
        var_key, fit_key = key.split(':')
        path = output_paths.get((var_key, fit_key), 'N/A')
        logger.info(f"        Output: {path}")
    logger.info(f"    Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"    Average time per task: {elapsed/len(all_results):.2f} seconds")

    if failures > 0:
        logger.info("Failed tasks:")
        for r in all_results:
            if not r[0]:
                logger.info(f"   - {r[1]}:{r[2]}:{r[3]}")
                if r[7]:
                    logger.info(f"     Error: {r[7][:200]}...")


def process_single_fit(logger, args, var, anom_type, model_with_most, mem, fpath):
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
        
    Returns
    -------
    tuple
        (success, var, anom_type, mem, data_dict, coords_dict, attrs_dict, error_message)
        where data_dict is a dictionary of {var_name: numpy_array}
    """
    try:
        logger.info(f"Working on {var}:{model_with_most}:{mem} - {anom_type} fit")

        try:
            var_name = ANOM_TYPE_TO_VAR[anom_type]
        except KeyError:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        ds = xr.open_dataset(fpath).sel(member_id=mem)

        ds_fit = ds_mle_fit(
            args,
            ds,
            var_name=var_name,
            fit_dim='year',
            all_mems=True
        )
        logger.info(f"{anom_type} fit complete for {var}:{mem}")

        reset_mle_stats()

        sfx = anom_type if anom_type == 'raw' else f'anom_{anom_type}'
        gev_param_names = [
            f'{p}_{sfx}' for p in MLE_FIT_ATTRS[anom_type]['param_names']
        ]

        data_dict = {}
        for vname in gev_param_names:
            data_dict[vname] = ds_fit[vname].values

        coords_dict = {}
        for coord_name in ds_fit.coords:
            if coord_name != 'member_id':
                coords_dict[coord_name] = ds_fit[coord_name].values

        attrs_dict = dict(ds_fit.attrs)

        ds_fit.close()
        ds.close()

        return (True, var, anom_type, mem, data_dict, coords_dict, attrs_dict, None)

    except Exception as e:
        import traceback
        error_msg = f"Error processing {var}:{anom_type}:{mem} - {str(e)}\n{traceback.format_exc()}"
        logger.info(error_msg)
        return (False, var, anom_type, mem, None, None, None, error_msg)


def combine_results_into_datasets(logger, args, all_results, model_with_most, fpath):
    """Combine results from all workers into datasets organized by variable and fit_type.
    
    Parameters
    ----------
    logger: logging.Logger
        for printing

    all_results : list
        List of results from serial processing

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
                    'coords_dict': coords_dict,
                    'attrs_dict': attrs_dict
                }

            grouped[key]['members'].append(mem)
            grouped[key]['data_dicts'].append(data_dict)

    output_paths = {}

    for (var, anom_type), group_data in grouped.items():
        logger.info(f"Creating dataset for {var}:{anom_type}")

        members = group_data['members']
        data_dicts = group_data['data_dicts']
        coords_dict = group_data['coords_dict']
        attrs_dict = group_data['attrs_dict']

        data_vars = {}
        var_names = list(data_dicts[0].keys())

        for var_name in var_names:
            arrays = [d[var_name] for d in data_dicts]
            stacked = np.stack(arrays, axis=0)
            dims = ('member_id', 'lat', 'lon')
            data_vars[var_name] = (dims, stacked)

        coords = {'member_id': members}
        for coord_name, coord_values in coords_dict.items():
            coords[coord_name] = coord_values

        ds_combined = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs_dict
        )

        logger.info(f"   Dataset shape: {ds_combined.dims}")
        logger.info(f"   Members: {len(members)}")

        try:
            head_data_path = MIP_FIT_PATH_DICT[args.data]['data']
        except Exception as e:
            raise KeyError(f"Error in data config for GEV fitting: {str(e)}")

        gev_dir = head_data_path / var / 'gev'
        os.makedirs(gev_dir, exist_ok=True)
        fname = f"{fpath.stem}_gev_{args.fit}_allmems_{anom_type}.nc"
        output_path = gev_dir / fname

        logger.info(f"   Saving to: {output_path}")
        ds_combined.to_netcdf(output_path)

        output_paths[(var, anom_type)] = str(output_path)

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
    fnames = [f for f in data_path.glob("*_landonly.nc")]
    modelname_filepath_matcher = {
        extract_model_name(f): f for f in fnames
    }

    Nens_for_active_models = np.array([
        len(m.all_members) for m in CMIPConfig.iter_active_models(var)
    ])

    max_inds = np.where(Nens_for_active_models == np.max(Nens_for_active_models))[0]

    tied_models = None
    if len(max_inds) > 1:
        all_model_names = np.array(
            [m.name for m in CMIPConfig.iter_active_models(var)]
        )
        tied_models = all_model_names[max_inds[1:]].tolist()

    ind_ = max_inds[0]
    model_with_most = [m.name for m in CMIPConfig.iter_active_models(var)][ind_]
    n_members = np.max(Nens_for_active_models)
    fpath = modelname_filepath_matcher[model_with_most]

    return model_with_most, fpath, n_members, tied_models