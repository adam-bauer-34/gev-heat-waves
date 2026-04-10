"""Scripting functions for computing Kuiper statistics.

Adam Michael Bauer
UChicago
11.8.2025
"""

import numpy as np
import xarray as xr

from scipy.stats import genextreme
from astropy.stats import kuiper

from evt_heat_waves.config import MLE_FIT_ATTRS, MLE_FULL_PARAM_NAMES
from evt_heat_waves.mle.mle import _mle_fit

# suffix map to get shape, loc, and scale names
suffix_map = {
    't2m':             'raw',
    'tas':             'raw',
    't2m_anom_annmean':'anom_annmean',
    't2m_anom_trend':  'anom_trend',
}


def compute_kuiper_stats(args, ds, var_name='t2m', fit_dim='year'):
    """Compute Kuiper statistics at each gridcell.

    Parameters
    ----------
    args: argparse.Namespace
        CLI arguments, this needs:
            - args.fit

    ds: xarray.Dataset
        the input dataset containing the data to fit

    var_name: str
        the variable name in the dataset to fit the GEV distribution to

    fit_dim: str
        the dimension over which to fit the GEV distribution (e.g., 'year')

    Returns
    -------
    ds: xarray.Dataset
        the input dataset with added GEV parameters as new variables
    """
    # subselect data
    da = ds[var_name]
    shapes = ds[f'shape_{suffix_map[var_name]}']
    locs = ds[f'loc_{suffix_map[var_name]}']
    scales = ds[f'scale_{suffix_map[var_name]}']

    # define ufunc dict for obs analysis
    ufunc_kwargs_obs = dict(
        input_core_dims=[[fit_dim], [], [], []],
        output_core_dims=[['kuiper']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'kuiper': 1}
            }
    )

    # compute Kuiper statistics for observed and synthetic data
    da_ko = xr.apply_ufunc(
        _kuiper,
        da,
        shapes,
        locs,
        scales,
        **ufunc_kwargs_obs
    )

    # now handle synthetic obs + kuiper calculation.
    # first do synthetic draws via fitted distributions
    ufunc_kwargs_syn = dict(
        input_core_dims=[[], [], []],
        output_core_dims=[['kuiper']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        kwargs={'N_SAMPLES': len(ds[var_name].year)}
        dask_gufunc_kwargs={
            'output_sizes': {'kuiper': 1}
            }
    )

    da_ks = xr.apply_ufunc(
        _kuiper_syn,
        shapes,
        locs,
        scales,
        **ufunc_kwargs_syn
    )

    # assign kuiper statistics to dataset
    ds = _assign_kuiper(ds, var_name, da_ko, da_ks)

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
        return k
    

def _kuiper_syn(shape, loc, scale, N_SAMPLES):
    if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
        return np.array([np.nan])
    
    else:
        tmp_sample = genextreme.rvs(-shape, loc=loc,
                                    scale=scale, size=N_SAMPLES)
        loc_hat, scale_hat, shape_hat = _mle_fit(tmp_sample)

        # catch if MLE fails
        if np.isnan(loc_hat) or np.isnan(scale_hat) or np.isnan(shape_hat):
            return np.array([np.nan])
        
        else:
            tmp_k = _kuiper(tmp_sample, shape_hat, loc_hat, scale_hat)
            return tmp_k

def _assign_kuiper(ds, var_name, da_ko, da_ks):
    """Assign Kuiper statistics for observed and synthetic data to the dataset.

    Resolves the variable name suffix using the same suffix_map as
    _assign_params, then assigns obs_k and syn_k arrays in one place
    rather than repeating ds.assign() calls for every var_name.
    """

    if var_name not in suffix_map:
        raise ValueError(
            f"Unknown var_name {var_name!r}. "
            f"Expected one of: {list(suffix_map)}"
        )

    sfx = suffix_map[var_name]

    ds = ds.assign({f'obs_k_{sfx}': (('lat', 'lon'), da_ko.data[:, :, 0])})
    ds = ds.assign({f'syn_k_{sfx}': (('lat', 'lon'), da_ks.data[:, :, 0])})

    return ds