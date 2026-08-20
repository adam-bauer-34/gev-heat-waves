"""Runner function for computing Kuiper statistics for ERA5 data in parallel using MPI.

Adam Bauer
UChicago
Apr 2026
"""

import copy
import shutil
import traceback
import time

import xarray as xr

from evt_heat_waves.config import ERA5_PATH, ANOM_TYPE_TO_VAR
from evt_heat_waves.mle.mle import ds_mle_fit
from evt_heat_waves.era5.kuiper.kuiper_fitting import (
    compute_kuiper_stats,
    _fixed_shape,
)

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def print_summary(flat_results, all_results, logger):
    """Print a summary of MPI task results.

    Parameters
    ----------
    flat_results : list[tuple]
        List of (success, fit_type, output_path, error_message) tuples.
    all_results : list[list[tuple]]
        List of lists of (success, fit_type, output_path, error_message) tuples.
    logger : logging.Logger
        Logger used to report summary information.
    """
    # compute the number of successes and failures
    successes = sum(1 for r in flat_results if r[0])
    failures = sum(1 for r in flat_results if not r[0])

    # Count by fit type
    fit_type_counts = {}
    for r in flat_results:
        fit_type = r[1]
        if fit_type not in fit_type_counts:
            fit_type_counts[fit_type] = {"success": 0, "failure": 0}
        if r[0]:
            fit_type_counts[fit_type]["success"] += 1
        else:
            fit_type_counts[fit_type]["failure"] += 1

    print("-" * width)
    logger.info("SUMMARY")
    logger.info("-" * width)
    logger.info(f"Successful: {successes}/{len(all_results)}")
    logger.info(f"Failed: {failures}/{len(all_results)}")
    logger.info(f"Breakdown by fit type:")
    for fit_type, counts in sorted(fit_type_counts.items()):
        total = counts["success"] + counts["failure"]
        logger.info(f"  - {fit_type:8s}: {counts['success']}/{total} successful")

    if failures > 0:
        logger.info("Failed tasks:")
        for r in all_results:
            if not r[0]:
                logger.info(f"  - {r[3]}")
    logger.info("-" * width)


def process_single_kuiper(logger, args, var, TMIN, anom_type, rank):
    """Process a single fit for a single model-variable combination.

    Parameters
    ----------
    logger: logging.Logger
        logging object

    args: argparse.Namespace
        CLI args, needs to have
            - args.fit (fit type to pass to MLE)
            - args.no_se (whether to calc SEs)

    var : str
        Variable name

    TMIN: int
        starting year for MLE fit

    anom_type: str
        the type of anomaly (trend, raw, annmean)

    rank : int
        MPI rank of current process

    Returns
    -------
    tuple
        (success, anom_type, output_path, error_message)
    """

    try:
        # import data from ERA5/landonly
        landonly_path = ERA5_PATH / "landonly"
        gev_path = ERA5_PATH / "gev"

        # mapping for data -> variable name in dataset
        try:
            var_name = ANOM_TYPE_TO_VAR[anom_type]
            var_name = (
                var_name if anom_type != "raw" else "t2m"
            )  # convert to ERA5 naming convention if using raw data
        except KeyError:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        logger.debug(
            f"The anomaly type {anom_type} was converted to variable name {var_name}"
        )

        # try to import stationary fit dataset to use for kuiper analysis
        # if it doesn't exist, make it using MLE fit
        fpath = (
            gev_path
            / f"era5_{var}_{args.grid}_landonly_gev_{args.fit}_TMIN{TMIN}_{anom_type}.nc"
        )
        logger.debug(
            f"[Rank {rank}] Attempting to open stationary fit dataset for {var}:{anom_type} with TMIN={TMIN} at path: {fpath}"
        )
        try:
            ds = xr.open_dataset(fpath)
            logger.debug(
                f"Successfully opened stationary fit dataset for {var}:{anom_type} with TMIN={TMIN} at path: {fpath}"
            )
        except FileNotFoundError:
            logger.warning(
                f"Stationary fit dataset not found for {var}:{anom_type} with TMIN={TMIN}. "
                f"Running MLE fit to create it for kuiper analysis."
            )
            ds = xr.open_dataset(
                landonly_path / f"era5_{var}_{args.grid}_landonly.nc"
            ).sel(year=slice(TMIN, 2024))
            ds = ds_mle_fit(args, ds, var_name=var_name, fit_dim="year")

        # the misspecification test draws from a free-shape fit of the same data,
        # so it needs that fit alongside the fixed-shape one being tested
        ds_free = None
        if _fixed_shape(args.fit) is not None:
            free_fpath = (
                gev_path
                / f"era5_{var}_{args.grid}_landonly_gev_{args.free_fit}_TMIN{TMIN}_{anom_type}.nc"
            )
            logger.debug(
                f"[Rank {rank}] Attempting to open free-shape ({args.free_fit}) fit dataset at path: {free_fpath}"
            )
            try:
                ds_free = xr.open_dataset(free_fpath)
            except FileNotFoundError:
                logger.warning(
                    f"Free-shape ({args.free_fit}) fit dataset not found for {var}:{anom_type} with TMIN={TMIN}. "
                    f"Running MLE fit to create it for the misspecification test."
                )
                free_args = copy.copy(args)
                free_args.fit = args.free_fit
                ds_free = xr.open_dataset(
                    landonly_path / f"era5_{var}_{args.grid}_landonly.nc"
                ).sel(year=slice(TMIN, 2024))
                ds_free = ds_mle_fit(
                    free_args, ds_free, var_name=var_name, fit_dim="year"
                )

        # do kuiper analysis
        ds_kuiper = compute_kuiper_stats(
            ds,
            var_name=var_name,
            fit_dim="year",
            fit_type=args.fit,
            ds_free=ds_free,
            free_fit_type=args.free_fit,
            n_reps=args.n_reps,
        )

        # check: print kuiper dataset
        logger.debug(f"[Rank {rank}]: Kuiper statistics-fitted dataset:\n {ds_kuiper}")

        # save joined dataset from stationary + kuiper stats
        gev_dir = (
            fpath.parent.parent / "gev"
            if not args.debug
            else fpath.parent.parent / "gev_debug"
        )
        gev_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists
        logger.debug(f"[Rank {rank}] Output directory for GEV fit: {gev_dir}")

        kuiper_name = f"{fpath.stem}_kuiper{fpath.suffix}"
        output_path = gev_dir / kuiper_name

        logger.debug(f"The output path is: {output_path}")

        ds_kuiper.to_netcdf(output_path)  # save kuiper results

        # close kuiper and stationary datasets after saving to keep memory abundant
        ds_kuiper.close()
        ds.close()
        if ds_free is not None:
            ds_free.close()

        # return success, anomaly type, stationary fit output path, nonstationary fit output path, and error msg
        return (True, anom_type, output_path, None)

    except Exception as e:
        error_msg = f"Error processing {var}:{anom_type} function call with TMIN={TMIN} - {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[Rank: {rank}] {error_msg}")

        # return success, anomaly type, stationary fit output path, nonstationary fit output path, and error msg
        return (False, anom_type, None, error_msg)
