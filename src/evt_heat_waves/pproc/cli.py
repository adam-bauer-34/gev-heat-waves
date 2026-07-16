"""CLI parsing functions.

Adam Bauer
UChicago
Apr 9, 2026
"""

import argparse


def parse_args_pproc():
    """Parse CLI arguments for preprocessing."""

    parser = argparse.ArgumentParser(
        description="Preprocessing climate model data for later analysis."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="era5",
        choices=["era5", "cmip", "amip", "pop"],
        help="The data type to preprocess",
    )

    parser.add_argument(
        "--grid",
        type=str,
        default="1deg",
        choices=["1deg", "0.5deg"],
        help="Grid of the data to preprocess (only relevant for ERA5)",
    )

    parser.add_argument(
        "--make_check_plots",
        action="store_true",
        default=False,
        help="Make check plots for land masking and regridding?",
    )

    parser.add_argument(
        "--bypass-checks",
        action="store_true",
        default=False,
        help="Bypass manual checking of meta.yaml and qc.yaml files when created for CMIP or AMIP analysis.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization).",
    )

    return parser.parse_args()
