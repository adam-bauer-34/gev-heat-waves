"""Scripting functions for computing Kuiper statistics.

Adam Michael Bauer
UChicago
11.8.2025
"""

import numpy as np
import xarray as xr

from scipy.stats import genextreme
from mle_claude import _mle_fit
from astropy.stats import kuiper


def compute_kuiper_stats(ds, var_name='t2m', print_summary=True):
    """write if works
    """
    if var_name == 't2m':
        gev_append = '_raw'
    
    elif var_name == 't2m_anom_annmean':
        gev_append = '_anom_annmean'

    elif var_name == 't2m_anom_trend':
        gev_append = "_anom_trend"

    else:
        raise ValueError(f"{var_name} argument in compute_kuiper_stats is invalid.")

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
    # print("Doing synthetic bootstrapping fits + computing Kuiper statistics...")
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

    elif var_name == 't2m_anom_annmean':
        ds = ds.assign(obs_k_anom_annmean=(('lat', 'lon'), da_ko.data[:, :, 0]))
        ds = ds.assign(syn_k_anom_annmean=(('lat', 'lon'), da_ks.data[:, :, 0]))

    elif var_name == 't2m_anom_trend':
        ds = ds.assign(obs_k_anom_trend=(('lat', 'lon'), da_ko.data[:, :, 0]))
        ds = ds.assign(syn_k_anom_trend=(('lat', 'lon'), da_ks.data[:, :, 0]))

    else:
        raise ValueError(f"Invalid variable name {var_name}.")

    if print_summary:
        _print_summary(ds, 'obs_k', gev_append)
        _print_summary(ds, 'syn_k', gev_append)

    return ds


def _kuiper(sample, shape, loc, scale, SAMPLE_THRES=10):
    if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
        return np.array([np.nan])
        
    sample = sample[np.isfinite(sample)]

    if len(sample) < SAMPLE_THRES:
        return np.array([-1])

    else:
        k, _ = kuiper(sample,
                           lambda x: genextreme.cdf(x,
                                                c=-shape, loc=loc, scale=scale))
        return np.array([k])
    

def _kuiper_syn(shape, loc, scale, N_SAMPLES):
    if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
        return np.full(N_SAMPLES, np.nan)
    
    else:
        tmp_sample = genextreme.rvs(-shape, loc=loc,
                                    scale=scale, size=N_SAMPLES)
        loc_hat, scale_hat, shape_hat = _mle_fit(tmp_sample)
        tmp_k = _kuiper(tmp_sample, shape_hat, loc_hat, scale_hat)
        return np.array([tmp_k])


def _print_summary(ds, var_name, gev_append):
    """Print summary statistics of the Kuiper variables.
    """

    data = ds[var_name + gev_append].values.flatten()
    data = data[np.isfinite(data)]  # screen nans
    data = data[data >= 0]  # screen out -1 from ocean values

    if var_name == 'obs_k':
        print_message = f"Summary statistics for observation-based Kuiper statistics:"

    elif var_name == 'syn_k':
        print_message =f"Summary statistics for bootstrapped, synthetic Kuiper statistics:"

    print(print_message)
    print("-" * 50)
    print(f"Number of samples: {data.size}")
    print(f"Minimum:          {np.min(data):.4f}")
    print(f"Maximum:          {np.max(data):.4f}")
    print(f"Mean:             {np.mean(data):.4f}")
    print(f"Median:           {np.median(data):.4f}")
    print(f"5th percentile:   {np.percentile(data, 5):.4f}")
    print(f"95th percentile:  {np.percentile(data, 95):.4f}")
    print(f"Std. deviation:   {np.std(data):.4f}")