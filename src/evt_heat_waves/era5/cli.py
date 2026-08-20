"""CLI parsing functions.

Adam Bauer
UChicago
Apr 9, 2026
"""

import argparse

from evt_heat_waves.config import MLE_FIT_ATTRS

# the Kuiper analysis assumes a stationary fit, so every fit type except the
# nonstationary ones is allowed; fits like 'stat_gumbel' hold the shape
# parameter fixed rather than estimating it via MLE
KUIPER_FIT_TYPES = tuple(f for f in MLE_FIT_ATTRS if not f.startswith("nonstat"))


def parse_args_bootstrap():
    """Parse CLI arguments for Kuiper bootstrapping calculation."""

    parser = argparse.ArgumentParser(
        description="Bootstrapping Kuiper statistics on a synthetic grid."
    )

    parser.add_argument(
        "--tmin",
        type=int,
        default=1979,
        help="minimum time of interval analyzed; sets sample size for Kuiper stats; equal to 2024 - tmin",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization).",
    )

    return parser.parse_args()


def parse_args_era5_fit():
    """Parse CLI arguments for ERA5 fitting."""

    parser = argparse.ArgumentParser(
        description="Fitting GEV distribution to climate data."
    )

    parser.add_argument(
        "--fit",
        type=str,
        default="stat_new",
        help="The type of GEV fit to perform. Each option holds different parameters constant or assumes (non)stationary in the data",
    )

    parser.add_argument(
        "--free_fit",
        type=str,
        default="stat_new",
        help="The free-shape fit whose parameters generate the samples for the Kuiper misspecification test. Only used when --fit holds the shape parameter fixed",
    )

    parser.add_argument(
        "--n_reps",
        type=int,
        default=1,
        help="Replicates per gridcell for the Kuiper misspecification test. Draws use common random numbers, so the paired difference against the null is low-variance even at n_reps=1",
    )

    parser.add_argument(
        "--grid",
        type=str,
        default="1deg",
        choices=["1deg", "0.5deg"],
        help="Grid of the data to preprocess (only relevant for ERA5)",
    )

    parser.add_argument(
        "--mpi",
        action="store_true",
        default=False,
        help="[DEPRECIATED] Use MPI to paralllelize execution? (requires HPC config)",
    )

    parser.add_argument(
        "--no_se",
        action="store_true",
        default=False,
        help="Turns off standard error calculation.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization).",
    )

    return parser.parse_args()


def check_kuiper_config_compatability(args):
    """Check if dependent CLI inputs are compatable.

    Parameters
    ----------
    args: argparse.Namespace
        CLI args
    """

    if args.fit not in KUIPER_FIT_TYPES:
        raise ValueError(
            f"For Kuiper analysis, args.fit must be one of {list(KUIPER_FIT_TYPES)}. Got {args.fit}."
        )

    # the misspecification test draws from the free-shape fit, so that fit has to
    # be stationary and has to actually leave the shape parameter free
    if args.free_fit not in KUIPER_FIT_TYPES:
        raise ValueError(
            f"args.free_fit must be one of {list(KUIPER_FIT_TYPES)}. Got {args.free_fit}."
        )

    if "shape" not in MLE_FIT_ATTRS[args.free_fit]["param_names"]:
        raise ValueError(
            f"args.free_fit must leave the shape parameter free, but {args.free_fit} holds it fixed."
        )

    if args.n_reps < 1:
        raise ValueError(f"args.n_reps must be at least 1. Got {args.n_reps}.")
