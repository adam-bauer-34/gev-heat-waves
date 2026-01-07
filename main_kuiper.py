"""Main file for GEV fitting of ERA5 data.

Adam Michael Bauer
UChicago
11.10.2025

To run: main_gev_fit.py GRID DO_BOOTSTRAPPING
"""

import sys

import xarray as xr
import numpy as np

from src.kuiper import compute_kuiper_stats

np.random.seed(42)  # set seed for reproducibility

# import command line grid
GRID = sys.argv[1]
STAT = sys.argv[2]
DO_BOOTSTRAPPING = int(sys.argv[3])

print("Importing land-masked data...")
# define variables and open datasets
vars = ['t2m_annual_max', 't2m_annual_min']
dss = [xr.open_dataset('data/ERA5/landonly/era5_' + VAR
                       + '_' + GRID + '_landonly_gev_' + STAT + '.nc') for VAR in vars]

print("Computing kuiper statistics...")
# carry out GEV fitting for each dataset
print("-" * 40)
print("Kuiper stats for raw data:")
print("-" * 40)

dss_with_fit = [compute_kuiper_stats(ds, var_name='t2m', print_summary=False) for ds in dss]

print("-" * 40)
print("Kuiper stats for anomalies:")
print("-" * 40)

dss_with_fit_on_both = [compute_kuiper_stats(ds, var_name='t2m_anom', print_summary=False) for ds in dss_with_fit]

print("Kuiper statistics computed.")

print("Saving datasets...")
# save datasets
for VAR, ds_masked in zip(vars, dss_with_fit_on_both):
    ds_masked.to_netcdf('data/ERA5/landonly/era5_' + VAR + '_' + GRID + '_landonly_gev_' + STAT + '_kuiper.nc')

print("Datasets saved to data/ERA5/landonly/")

# If we also want to create data for bootstrapped Kuiper statistics
# where parameters never change, carry out the below
if DO_BOOTSTRAPPING:
    from scipy.stats import genextreme
    from astropy.stats import kuiper
    from src.mle import _mle_fit

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
                                    size=len(dss[0]['t2m'].year)
                                    )
        
        # fit a GEV to those data
        # shape_hat, loc_hat, scale_hat = genextreme.fit(tmp_sample)
        loc_hat, scale_hat, shape_hat = _mle_fit(tmp_sample, SAMPLE_THRES=10, non_stat=False)
        print(f"Fitted params: {shape_hat}, {loc_hat}, {scale_hat}")

        # compute the Kuiper statistic of fitted params -> GEV
        tmp_k, _ = kuiper(tmp_sample,
                          lambda x: genextreme.cdf(x,
                                                   -shape_hat, loc_hat, scale_hat))
        
        # store
        boot_ks[n] = tmp_k

    print("Bootstrapped Kuiper statistics complete.")

    # make dataset for saving
    ds_boot = xr.Dataset(
        data_vars={'boot_ks': (['iter'], boot_ks)},
        coords={'iter': (['iter'], np.arange(0, N_BOOTSTRAP, 1))},
        attrs={
            'shape': shape,
            'loc': loc,
            'scale': scale
        }
    )

    filepath = 'data/stats/bootstrapped_ks.nc'  # save to general stats data folder
    ds_boot.to_netcdf(filepath)  # save
    print('Bootstrapped Kuiper statistics saved to: {}'.format(filepath))