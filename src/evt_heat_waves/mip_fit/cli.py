"""CLI parsing functions.

Adam Bauer
UChicago
Apr 9, 2026
"""

import argparse


def parse_args_mip_fit():
    """Parse CLI arguments for GEV fitting.
    """

    parser = argparse.ArgumentParser(
        description="Fitting GEV distribution to climate data."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="cmip",
        choices=['cmip', 'amip'],
        help="The data type to fit"
    )

    parser.add_argument(
        "--fit",
        type=str,
        default='nonstat',
        choices=['nonstat', 'stat', 'stat_fixed_xi', 'nonstat_fixed_xi_loc_only'],
        help="The type of GEV fit to perform. Each option holds different parameters constant or assumes (non)stationary in the data"
    )

    parser.add_argument(
        "--member_config",
        type=str,
        default='prim',
        choices=['prim', 'most'],
        help="Do fits for each model's primary member (primary) or all of the members for the model with the most members (most)"
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