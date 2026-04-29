"""Runner function for bootstrapped Kuiper statistics calculation.

Adam Bauer
UChicago
Apr 8, 2026
"""

import xarray as xr
import numpy as np

from scipy.stats import genextreme
from astropy.stats import kuiper

from evt_heat_waves.config import STATS_PATH
from evt_heat_waves.mle.mle import _mle_fit


def get_bootstrapped_kuipers(logger,
                            args, N_BOOTSTRAP=1000,
                            loc=2.0, scale=1.0, shape=-0.25):
    """Compute bootstrapped Kuiper statistics. This procedure mimics
    having a grid with 1000 gridboxes where each gridcell has the same
    GEV distribution.

    Parameters
    ----------
    logger: logging.Logger
        logger object

    args: argparse.Namespace
        CLI arguments
    
    N_BOOTSTRAP: int (default = 1000)
        number of bootstrapped samples to take

    loc: float (default = 2)
        location parameter of GEV

    scale: float (default = 1)
        scale parameter of GEV

    shape: float (default = -1/4)
        shape parameter of GEV
    """

    TMIN = args.tmin
    N_YEARS = 2024 - int(TMIN)  # hard coded from number of years in climate model record

    np.random.seed(42)  # set seed for reproducibility

    logger.info("Computing bootstrapped Kuiper statistics...")
    # now do bootstrapping technique with the same parameters

    boot_ks = np.zeros(N_BOOTSTRAP)

    # for each bootstrapping iteration, do:
    for n in range(N_BOOTSTRAP):
        # take a sample of GEV distribution values
        tmp_sample = genextreme.rvs(c=-shape,
                                    loc=loc,
                                    scale=scale,
                                    size=N_YEARS
                                    )
        
        # fit a GEV to those data
        # shape_hat, loc_hat, scale_hat = genextreme.fit(tmp_sample)
        loc_hat, scale_hat, shape_hat = _mle_fit(tmp_sample, SAMPLE_THRES=10, non_stat=False)

        # compute the Kuiper statistic of fitted params -> GEV
        tmp_k, _ = kuiper(tmp_sample,
                            lambda x: genextreme.cdf(x,
                                                    -shape_hat, loc_hat, scale_hat))
        
        # store
        boot_ks[n] = tmp_k

    # make dataset for saving
    ds_boot = xr.Dataset(
        data_vars={'boot_ks': (['iter'], boot_ks)},
        coords={'iter': (['iter'], np.arange(0, N_BOOTSTRAP, 1))},
        attrs={
            'shape': shape,
            'loc': loc,
            'scale': scale,
        }
    )

    filepath = STATS_PATH / f'bootstrapped_ks_{TMIN}.nc'  # save to general stats data folder
    ds_boot.to_netcdf(filepath)  # save
    logger.info(f'Bootstrapped Kuiper statistics saved to: {filepath}')