"""Main file for fitting GEV distribution to CMIP/AMIP data.

Adam Bauer
UChicago
Apr 8, 2026
"""

import shutil
import time

from mpi4py import MPI

from evt_heat_waves.logging_utils import setup_logger, get_git_hash
from evt_heat_waves.cli import parse_args_era5_fit
from evt_heat_waves.era5 import FIT_REGISTRY

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def main():
    args = parse_args_era5_fit()

    # Initialize MPI early if needed, otherwise use a dummy comm
    comm = MPI.COMM_WORLD if args.mpi else None
    rank = comm.Get_rank() if comm else 0

    logger = setup_logger(args.debug)
    t0 = time.time()

    if rank == 0:
        logger.info("-" * width)
        logger.info(f"Git hash: {get_git_hash()}")
        logger.info(f"Doing GEV fit for config: {args.fit}|MPI={args.mpi}")
        logger.info("-" * width)

    try:
        run_fit = FIT_REGISTRY['mpi'] if args.mpi else FIT_REGISTRY['no_mpi']
    except KeyError:
        raise ValueError("Runner for with/without MPI doesn't exist for current setup.")

    if args.mpi or rank == 0:
        run_fit(logger, args, comm=comm)

    if comm:
        comm.Barrier()

    if rank == 0:
        t1 = time.time()
        logger.info("GEV fitting complete!")
        logger.info(f"Total runtime: {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()