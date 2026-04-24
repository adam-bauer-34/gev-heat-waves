"""Function bank for serial (loop-based) GEV fitting to CMIP data.

Adam Michael Bauer
UChicago
Apr 2026
"""

import os
import shutil
import time

import xarray as xr

from evt_heat_waves.config import MIP_FIT_PATH_DICT, ANOM_TYPE_TO_VAR
from evt_heat_waves.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.utils import extract_model_name
from evt_heat_waves.mle.mle import ds_mle_fit, reset_mle_stats, get_mle_success_rate

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def runner(logger, args):
    """Serial runner using explicit loops (no task abstraction)."""

    start_time = time.time()
    logger.info("Starting SERIAL processing (loop-based)")

    # ---- Paths ----
    try:
        head_data_path = MIP_FIT_PATH_DICT[args.data]['data']
        config_meta_path = MIP_FIT_PATH_DICT[args.data]['config']['meta']
        config_qc_path = MIP_FIT_PATH_DICT[args.data]['config']['qc']
    except Exception as e:
        raise KeyError(f"Error in data config for GEV fitting: {str(e)}")

    # ---- Config ----
    CMIPConfig = CMIP6EnsembleConfig.from_yaml(
        config_meta_path,
        config_qc_path
    )

    vars = ['tas_annual_max', 'tas_annual_min']
    anom_types = ['raw', 'annmean', 'trend']

    results = []

    # ---- MAIN LOOPS ----
    for var in vars:
        logger.info("=" * width)
        logger.info(f"VARIABLE: {var}")

        gev_dir = head_data_path / var / 'gev'
        os.makedirs(gev_dir, exist_ok=True)

        data_path = head_data_path / var / 'landonly'

        # Keep this mapping (important)
        fnames = [f for f in data_path.glob("*_landonly.nc")]
        modelname_filepath_matcher = {
            extract_model_name(f): f for f in fnames
        }

        for m in CMIPConfig.iter_active_models(var):

            if m.name not in modelname_filepath_matcher:
                logger.warning(f"No file found for model: {m.name}")
                continue

            fpath = modelname_filepath_matcher[m.name]

            for anom_type in anom_types:
                logger.info("-" * width)
                logger.info(f"{var}:{m.name}:{anom_type}")

                result = process_single_fit(
                    logger=logger,
                    args=args,
                    var=var,
                    anom_type=anom_type,
                    m=m,
                    fpath=fpath,
                )

                results.append(result)

    # ---- SUMMARY ----
    successes = sum(1 for r in results if r[0])
    failures = sum(1 for r in results if not r[0])

    # keep breakdown simple but readable
    breakdown = {}
    for r in results:
        fit_type = r[1]
        if fit_type not in breakdown:
            breakdown[fit_type] = [0, 0]  # [success, failure]
        if r[0]:
            breakdown[fit_type][0] += 1
        else:
            breakdown[fit_type][1] += 1

    elapsed = time.time() - start_time

    logger.info("-" * width)
    logger.info("SUMMARY")
    logger.info("-" * width)
    logger.info(f"Successful: {successes}/{len(results)}")
    logger.info(f"Failed: {failures}/{len(results)}")
    logger.info("Breakdown by fit type:")

    for fit_type in sorted(breakdown.keys()):
        success, failure = breakdown[fit_type]
        total = success + failure
        logger.info(f"  - {fit_type:8s}: {success}/{total} successful")

    logger.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"Average time per task: {elapsed/len(results):.2f} seconds")

    if failures > 0:
        logger.info("Failed tasks:")
        for r in results:
            if not r[0]:
                logger.info(f"  - {r[3]}")
    logger.info("-" * width)



def process_single_fit(logger, args, var, anom_type, m, fpath):
    """Process a single fit for a single model-variable combination."""

    try:
        logger.info(f"Working on {var}:{m.name} - {anom_type} fit")

        ds = xr.open_dataset(fpath)
        ds_selected = ds.sel(member_id=m.primary_member)

        if anom_type not in ANOM_TYPE_TO_VAR:
            raise ValueError(f"Unknown anom_type: {anom_type}")

        var_name = ANOM_TYPE_TO_VAR[anom_type]

        ds_fit = ds_mle_fit(
            args,
            ds_selected,
            var_name=var_name,
            fit_dim='year'
        )

        logger.info(f"{anom_type} fit complete.")

        success_rate = get_mle_success_rate()
        reset_mle_stats()

        ds_fit.attrs['MLE_success_rate'] = success_rate

        gev_dir = fpath.parent.parent / 'gev' if not args.debug else fpath.parent.parent / 'gev_debug'
        gev_name = (
            fpath.stem
            + f"_gev_{args.fit}_{anom_type}"
            + fpath.suffix
        )

        output_path = gev_dir / gev_name
        ds_fit.to_netcdf(output_path)

        logger.info(f"Dataset saved to: {output_path}")

        ds_fit.close()
        ds.close()

        return (True, anom_type, str(output_path), None)

    except Exception as e:
        import traceback
        error_msg = (
            f"Error processing {var}:{m.name}:{anom_type} - {str(e)}\n"
            f"{traceback.format_exc()}"
        )
        logger.warning(error_msg)
        return (False, anom_type, None, error_msg)