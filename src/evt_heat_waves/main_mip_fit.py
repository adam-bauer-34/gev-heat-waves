"""Main file for fitting GEV distribution to CMIP/AMIP data.

Adam Bauer
UChicago
Apr 8, 2026
"""

import shutil
import time

from evt_heat_waves.logging import setup_logger, get_git_hash
from evt_heat_waves.cli import parse_args_mip_fit
from evt_heat_waves.mip_fit import FIT_REGISTRY

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def main():
    """Main function for GEV fitting.
    """

    # parse arguments
    args = parse_args_mip_fit()

    # setup logging
    logger = setup_logger(args.debug)

    t0 = time.time()

    # print statement for logging and reproducibility
    logger.info("-" * width)
    logger.info(f"Git hash: {get_git_hash()}")

    # run fitting for the passed data type, member config, and w/wo MPI turned on
    try:
        run_data_mem = FIT_REGISTRY[args.member_config]
        run_fit = run_data_mem['mpi'] if args.mpi else run_data_mem['no_mpi']

    except KeyError:
        raise ValueError(f"Runner for data type {args.data} with member config {args.member_config} with/without MPI doesn't exist for current setup.")

    logger.info(f"Doing GEV fitting for config: {args.data}|{args.member_config}|MPI={args.mpi}")
    logger.info("-" * width)
    run_fit(logger, args)

    t1 = time.time()

    # log finish
    logger.info("GEV fitting complete!")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()