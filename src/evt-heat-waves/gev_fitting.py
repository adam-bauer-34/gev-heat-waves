"""Scripting functions for GEV fitting.

Adam Michael Bauer
UChicago
11.8.2025
"""

import numpy as np
import xarray as xr

from scipy.stats import genextreme

def gev_fit(ds, var_name, fit_dim='year', parallel=True):
    """Write if it works
    """
    
    # subselect variable to do the fitting over
    da = ds[var_name]

    # carry out either parallelized or non-parallelized fit
    if parallel:
        gev_params = xr.apply_ufunc(
            _gev_fitter,
            da,
            input_core_dims=[[fit_dim]],
            output_core_dims=[['gev_params']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
        )

    else:
        gev_params = xr.apply_ufunc(
            _gev_fitter,
            da,
            input_core_dims=[['year']],
            output_core_dims=[['gev_params']]
        )

    # assign shape, loc, and scale parameters to their (lat, lon) coords
    if var_name == 't2m':
        ds = ds.assign(shape_raw = (('lat', 'lon'), gev_params.data[:, :, 0]))
        ds = ds.assign(loc_raw = (('lat', 'lon'), gev_params.data[:, :, 1]))
        ds = ds.assign(scale_raw = (('lat', 'lon'), gev_params.data[:, :, 2]))

    elif var_name == 't2m_anom':
        ds = ds.assign(shape_anom = (('lat', 'lon'), gev_params.data[:, :, 0]))
        ds = ds.assign(loc_anom = (('lat', 'lon'), gev_params.data[:, :, 1]))
        ds = ds.assign(scale_anom = (('lat', 'lon'), gev_params.data[:, :, 2]))

    # return the amended dataset
    return ds


def _gev_fitter(data, SAMPLE_THRES=10):
    """GEV fitting function for a given array of data.

    Parameters
    ----------
    data: (N,) array
        the data we fit the GEV distribution to

    SAMPLE_THRES: int
        the threshold for doing the fit. if the number of points is
        less than this, the function will return nans.

    Returns
    -------
    (c, loc, shape): (3,) array
        parameters that describe the distribution
        NOTE: we will return NaNs if SAMPLE_THRES is exceeded or if something goes
        wrong in the fitting process.
    """

    # only take finite values
    data = data[np.isfinite(data)]

    # if the number of points is less than the sample threshold,
    # return NaNs for the fitted parameters
    if len(data) < SAMPLE_THRES:
        return np.array([np.nan] * 3)
    
    # try to do the GEV distribution fit and return parameters
    try:
        c, loc, scale = genextreme.fit(data)
        return np.array([c, loc, scale])
    
    # if something weird happens (failure to fit, etc), return nans
    except Exception:
        return np.array([np.nan] * 3)