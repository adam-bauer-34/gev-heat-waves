"""Main file for bootstrapping Kuiper stats.

Adam Michael Bauer
UChicago
Apr 8, 2026
"""

import time
import shutil

from evt_heat_waves.logging import setup_logger, get_git_hash
from evt_heat_waves.cli import parse_args_bootstrap
from evt_heat_waves.kuiper.bootstrap import get_bootstrapped_kuipers

width = shutil.get_terminal_size(fallback=(80, 20)).columns

def main():
    """Main function for simulation setup and running.
    """

    # parse arguments
    args = parse_args_bootstrap()

    # setup logging
    logger = setup_logger(args.debug)

    t0 = time.time()

    # print statement for logging and reproducibility
    logger.info("-" * width)
    logger.info(f"Git hash: {get_git_hash()}")

    # run main function
    get_bootstrapped_kuipers(logger, args)

    t1 = time.time()

    # log finish
    logger.info("Bootstrapping calculation complete!")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()