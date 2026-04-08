"""Config file that loads paths from YAML.

Adam Bauer
UChicago
Jan 2026
"""

import yaml
import argparse 

from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "paths.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_ROOT = Path(CONFIG["DATA_ROOT"])

FIGS_PATH = Path(CONFIG["FIGS_PATH"])
ERA5_PATH = DATA_ROOT / CONFIG["ERA5_DIR"]
CMIP_PATH = DATA_ROOT / CONFIG["CMIP_DIR"]
AMIP_PATH = DATA_ROOT / CONFIG["AMIP_DIR"]
STATS_PATH = DATA_ROOT / CONFIG["STATS_DIR"]


def parse_args_pproc():
    """Parse CLI arguments for preprocessing.
    """

    parser = argparse.ArgumentParser(
        description="Preprocessing climate model data for later analysis."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="era5",
        choices=['era5', 'cmip', 'amip'],
        help="The data type to preprocess"
    )

    parser.add_argument(
        "--grid",
        type=str,
        default='1deg',
        choices=['1deg', '0.5deg'],
        help="Grid of the data to preprocess (only relevant for ERA5)"
    )

    parser.add_argument(
        "--make_check_plots",
        action='store_true',
        default=False,
        help='Make check plots for land masking and regridding?'
    )

    parser.add_argument(
        "--bypass-checks",
        action='store_true',
        default=False,
        help="Bypass manual checking of meta.yaml and qc.yaml files when created for CMIP or AMIP analysis."
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization)."
    )

    return parser.parse_args()

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
        default='pr',
        choices=['prim', 'most'],
        help="Do fits for each model's primary member (primary) or all of the members for the model with the most members (most)"
    )

    parser.add_argument(
        "--mpi",
        action='store_true',
        default=False,
        help="Use MPI to paralllelize execution? (requires HPC config)"
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Whether to run in debug mode (more verbose logging, no parallelization)."
    )

    return parser.parse_args()


def check_fitting_config_compatability(args):
    """Check if dependent CLI inputs are compatable.

    Parameters
    ----------
    args: argparse.Namespace
        CLI args
    """

    # do later if it makes sense
    pass

# quick test
if __name__ == "__main__":
    args = parse_args_pproc()
    print(f"Args for preprocessing: {args}")

    args_fit = parse_args_fitting()
    print(f"Args for fitting: {args_fit}")