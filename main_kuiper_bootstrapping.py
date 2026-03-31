"""Main file for GEV fitting of ERA5 data.

Adam Michael Bauer
UChicago
11.10.2025

To run: python main_kuiper_bootstrapping.py

Last edited: 2/5/2026, 12:45 AM CST
"""

import sys

import xarray as xr
import numpy as np

from scipy.stats import genextreme
from astropy.stats import kuiper

from src.mle import _mle_fit
from config import DATA_ROOT


TMIN = sys.argv[1]
N_YEARS = 2024 - int(TMIN)  # hard coded from number of years in climate model record

np.random.seed(42)  # set seed for reproducibility

print("Computing bootstrapped Kuiper statistics...")
# now do bootstrapping technique with the same parameters

# number of bootstraps to do
N_BOOTSTRAP = int(1e3)

# set parameters
shape, loc, scale = (-0.25, 20., 1.5)  # from Cael's original .matlab script
boot_ks = np.zeros(N_BOOTSTRAP)

# for each bootstrapping iteration, do:
for n in range(N_BOOTSTRAP):
    # take a sample of GEV distribution values
    tmp_sample = genextreme.rvs(c=-shape,
                                loc=loc,
                                scale=scale,
                                size=N_YEARS  # hard coded from number of years in climate model record
                                )
    
    # fit a GEV to those data
    # shape_hat, loc_hat, scale_hat = genextreme.fit(tmp_sample)
    loc_hat, scale_hat, shape_hat = _mle_fit(tmp_sample, SAMPLE_THRES=10, non_stat=False)
    # print(f"Fitted params: {shape_hat}, {loc_hat}, {scale_hat}")

    # compute the Kuiper statistic of fitted params -> GEV
    tmp_k, _ = kuiper(tmp_sample,
                        lambda x: genextreme.cdf(x,
                                                -shape_hat, loc_hat, scale_hat))
    
    # store
    boot_ks[n] = tmp_k

# print("Bootstrapped Kuiper statistics complete.")

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

filepath = DATA_ROOT / 'stats' / f'bootstrapped_ks_{TMIN}.nc'  # save to general stats data folder
ds_boot.to_netcdf(filepath)  # save
print(f'Bootstrapped Kuiper statistics saved to: {filepath}')