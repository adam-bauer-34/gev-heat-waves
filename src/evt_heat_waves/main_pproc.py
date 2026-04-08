"""Main file for preprocessing ERA5, CMIP6, and AMIP data.

Adam Bauer
UChicago
Apr 8, 2026
"""

import time

from evt_heat_waves.config import parse_args_pproc
from evt_heat_waves.logging import setup_logger, get_git_hash
from evt_heat_waves.pproc import PPROC_REGISTRY

def main():
    """Main function for simulation setup and running.
    """

    # parse arguments
    args = parse_args_pproc()

    # setup logging
    logger = setup_logger(args.debug)

    t0 = time.time()

    # print statement for logging and reproducibility
    logger.info(f"Git hash: {get_git_hash()}")

    # run preprocessing for the passed data type
    try:
        run_pproc = PPROC_REGISTRY[args.data_type]['runner']
    except KeyError:
        raise ValueError(f"Data type {args.data_type} doesn't exist in data registry:\n{PPROC_REGISTRY}")

    run_pproc(logger, args)

    t1 = time.time()

    # log finish
    logger.info("Preprocessing complete!")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()