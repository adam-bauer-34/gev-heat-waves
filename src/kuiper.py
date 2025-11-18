"""Scripting functions for computing Kuiper statistics.

Adam Michael Bauer
UChicago
11.8.2025
"""

import numpy as np
import xarray as xr

from scipy.stats import genextreme
from astropy.stats import kuiper

def compute_kuiper_stats(ds, var_name='t2m', print_summary=False):
    """write if works
    """
    if var_name == 't2m':
        gev_append = '_raw'
    
    elif var_name == 't2m_anom':
        gev_append = '_anom'

    # compute Kuiper statistics for observed and synthetic data
    da_ko = xr.apply_ufunc(
        _kuiper,
        ds[var_name],
        ds['shape' + gev_append],
        ds['loc'+ gev_append],
        ds['scale'+ gev_append],
        input_core_dims=[['year'], [], [], []],
        output_core_dims=[['kuiper']],
        vectorize=True,
        dask='parallelize',
        output_dtypes=[float]
    )

    # now handle synthetic obs + kuiper calculation.
    # first do synthetic draws via fitted distributions
    print("Doing synthetic bootstrapping fits + computing Kuiper statistics...")
    da_ks = xr.apply_ufunc(
        _kuiper_syn,
        ds['shape'+ gev_append],
        ds['loc'+ gev_append],
        ds['scale'+ gev_append],
        len(ds[var_name].year),
        input_core_dims=[[], [], [], []],
        output_core_dims=[['kuiper']],
        vectorize=True,
        dask='parallelize',
        output_dtypes=[float]
    )

    # assign kuiper statistics to dataset
    if var_name == 't2m':
        ds = ds.assign(obs_k_raw=(('lat', 'lon'), da_ko.data[:, :, 0]))
        ds = ds.assign(syn_k_raw=(('lat', 'lon'), da_ks.data[:, :, 0]))

    elif var_name == 't2m_anom':
        ds = ds.assign(obs_k_anom=(('lat', 'lon'), da_ko.data[:, :, 0]))
        ds = ds.assign(syn_k_anom=(('lat', 'lon'), da_ks.data[:, :, 0]))

    if print_summary:
        data = ds['obs_k' + gev_append].values.flatten()
        data = data[np.isfinite(data)]  # screen nans
        data = data[data >= 0]  # screen out -1 from ocean values

        print(f"Summary statistics for observation-based Kuiper statistics:")
        print("-" * 50)
        print(f"Number of samples: {data.size}")
        print(f"Minimum:          {np.min(data):.4f}")
        print(f"Maximum:          {np.max(data):.4f}")
        print(f"Mean:             {np.mean(data):.4f}")
        print(f"Median:           {np.median(data):.4f}")
        print(f"5th percentile:   {np.percentile(data, 5):.4f}")
        print(f"95th percentile:  {np.percentile(data, 95):.4f}")
        print(f"Std. deviation:   {np.std(data):.4f}")

        data = ds['syn_k' + gev_append].values.flatten()
        data = data[np.isfinite(data)]  # screen nans
        data = data[data >= 0]  # screen out -1 from ocean values

        print(f"Summary statistics for bootstrapped, synthetic Kuiper statistics:")
        print("-" * 50)
        print(f"Number of samples: {data.size}")
        print(f"Minimum:          {np.min(data):.4f}")
        print(f"Maximum:          {np.max(data):.4f}")
        print(f"Mean:             {np.mean(data):.4f}")
        print(f"Median:           {np.median(data):.4f}")
        print(f"5th percentile:   {np.percentile(data, 5):.4f}")
        print(f"95th percentile:  {np.percentile(data, 95):.4f}")
        print(f"Std. deviation:   {np.std(data):.4f}")

    return ds

def _kuiper(sample, shape, loc, scale):
    sample = sample[np.isfinite(sample)]

    if len(sample) < 10:
        return np.array([-1])

    else:
        k, _ = kuiper(sample,
                           lambda x: genextreme.cdf(x,
                                                c=shape, loc=loc, scale=scale))
        return np.array([k])
    

def _kuiper_syn(shape, loc, scale, N_SAMPLES):
    if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
        return np.full(N_SAMPLES, np.nan)
    
    else:
        tmp_sample = genextreme.rvs(shape, loc=loc,
                                    scale=scale, size=N_SAMPLES)
        shape_hat, loc_hat, scale_hat = genextreme.fit(tmp_sample)
        tmp_k = _kuiper(tmp_sample, shape_hat, loc_hat, scale_hat)
        return tmp_k