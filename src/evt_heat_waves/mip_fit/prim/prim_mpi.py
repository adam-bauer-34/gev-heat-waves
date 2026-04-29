"""Function bank for MPI-enhanced GEV fitting to CMIP data.

Adam Michael Bauer
UChicago
Apr 8 2026
"""

import shutil

import xarray as xr

from evt_heat_waves.config import ANOM_TYPE_TO_VAR
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def process_single_fit(logger, args, var, anom_type, m, modelname_filepath_matcher, rank):
    """Process a single fit for a single model-variable combination.
    
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
        logger.info(f"[Rank {rank}] Working on {var}:{m.name} - {anom_type} fit")
        
        fpath = modelname_filepath_matcher[m.name]
        ds = xr.open_dataset(fpath)
        ds_selected = ds.sel(member_id=m.primary_member)

        # mapping for data -> variable name in dataset
        try:
            var_name = ANOM_TYPE_TO_VAR[anom_type]
        except KeyError:
            raise ValueError(f"Unknown anom_type: {anom_type}")
        
        # do fitting
        ds_fit = ds_mle_fit(
            args,
            ds_selected,
            var_name=var_name,
            fit_dim='year'
        )
        logger.info(f"[Rank {rank}] {anom_type} fit complete.")

        # get MLE success rate; reset immediately after
        success_rate = get_mle_success_rate()        
        reset_mle_stats()

        # store MLE success rate as a dataset attribute
        ds_fit.attrs['MLE_success_rate'] = success_rate
        
        # Save dataset
        gev_dir = fpath.parent.parent / 'gev' if not args.debug else fpath.parent.parent / 'gev_debug'
        gev_name = fpath.with_name(
            fpath.stem + f"_gev_{args.fit}_{anom_type}" + fpath.suffix
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
        logger.warning(f"[Rank {rank}] {error_msg}")
        return (False, anom_type, None, error_msg)


def print_summary(flat_results, logger):
    """Print a summary of MPI task results.

    Parameters
    ----------
    flat_results : list[tuple]
        List of (success, fit_type, output_path, error_message) tuples.
    logger : logging.Logger
        Logger used to report summary information.
    """
    successes = sum(1 for r in flat_results if r[0])
    failures = len(flat_results) - successes

    fit_type_counts = {}
    for success, fit_type, *_ in flat_results:
        if fit_type not in fit_type_counts:
            fit_type_counts[fit_type] = {'success': 0, 'failure': 0}
        fit_type_counts[fit_type]['success' if success else 'failure'] += 1

    logger.info("-" * width)
    logger.info("SUMMARY")
    logger.info('-' * width)
    logger.info(f"Successful: {successes}/{len(flat_results)}")
    logger.info(f"Failed: {failures}/{len(flat_results)}")
    logger.info("Breakdown by fit type:")
    for fit_type, counts in sorted(fit_type_counts.items()):
        total = counts['success'] + counts['failure']
        logger.info(f"  - {fit_type:8s}: {counts['success']}/{total} successful")

    if failures > 0:
        logger.info("Failed tasks:")
        for success, fit_type, output_path, error_message in flat_results:
            if not success:
                logger.info(f"  - {error_message}")

    logger.info("-" * width)