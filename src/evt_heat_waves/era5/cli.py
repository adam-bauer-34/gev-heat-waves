"""CLI parsing functions.

Adam Bauer
UChicago
Apr 9, 2026
"""

import argparse


def parse_args_bootstrap():
    """Parse CLI arguments for Kuiper bootstrapping calculation.
    """

    parser = argparse.ArgumentParser(
        description="Bootstrapping Kuiper statistics on a synthetic grid."
    )

    parser.add_argument(
        "--tmin",
        type=int,
        default=1979,
        help="minimum time of interval analyzed; sets sample size for Kuiper stats; equal to 2024 - tmin"
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization)."
    )

    return parser.parse_args()


def parse_args_era5_fit():
    """Parse CLI arguments for ERA5 fitting.
    """

    parser = argparse.ArgumentParser(
        description="Fitting GEV distribution to climate data."
    )

    parser.add_argument(
        "--fit",
        type=str,
        default='stat_new',
        help="The type of GEV fit to perform. Each option holds different parameters constant or assumes (non)stationary in the data"
    )

    parser.add_argument(
        "--grid",
        type=str,
        default='1deg',
        choices=['1deg', '0.5deg'],
        help="Grid of the data to preprocess (only relevant for ERA5)"
    )

    parser.add_argument(
        "--mpi",
        action='store_true',
        default=False,
        help="[DEPRECIATED] Use MPI to paralllelize execution? (requires HPC config)"
    )

    parser.add_argument(
        "--no_se",
        action='store_true',
        default=False,
        help="Turns off standard error calculation."
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization)."
    )

    return parser.parse_args()


def check_kuiper_config_compatability(args):
    """Check if dependent CLI inputs are compatable.

    Parameters
    ----------
    args: argparse.Namespace
        CLI args
    """

    if args.fit != 'stat' and args.fit != 'stat_new' and args.fit != 'stat_lax':
        raise ValueError(f"For Kuiper analysis, args.fit must be 'stat', 'stat_new', or 'stat_lax'. Got {args.fit}.")