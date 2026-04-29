"""Main file for fitting GEV distribution to CMIP/AMIP data.

Adam Bauer
UChicago
Apr 8, 2026
"""

import shutil
import time

from evt_heat_waves.logging_utils import setup_logger, get_git_hash
from evt_heat_waves.era5.cli import parse_args_era5_fit, check_kuiper_config_compatability
from evt_heat_waves.era5 import KUIPER_REGISTRY

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def main():
    """Main function for GEV fitting.
    """

    # parse arguments
    args = parse_args_era5_fit()
    check_kuiper_config_compatability(args)

    # setup logging
    logger = setup_logger(args.debug)

    t0 = time.time()

    # print statement for logging and reproducibility
    logger.info("-" * width)
    logger.info(f"Git hash: {get_git_hash()}")

    # run fitting for the passed data type, member config, and w/wo MPI turned on
    try:
        run_kuiper = KUIPER_REGISTRY['mpi'] if args.mpi else KUIPER_REGISTRY['no_mpi']

    except KeyError:
        raise ValueError(f"Runner for with/without MPI doesn't exist for current setup.")

    logger.info(f"Doing Kuiper statistics analysis for config: {args.grid}|MPI={args.mpi}")
    logger.info("-" * width)
    run_kuiper(logger, args)

    t1 = time.time()

    # log finish
    logger.info("Kuiper statistics analysis complete!")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()