"""Functions to compute the standard errors of the MLE parameters using the Hessian.

Adam Bauer
UChicago
Apr 9, 2026
"""

import numpy as np
from . import HESS_REGISTRY

def get_standard_errors(theta_opt, temps, fit_type='stat'):
    """Compute the standard errors of the MLE parameters.

    Parameters
    ----------
    theta_opt: (N_params,) array-like
        The optimal MLE parameters

    temps: (N_years,) array-like
        temperature data used for MLE fitting

    fit_type: str, optional
        the type of fit, can be ('stat', 'loc_trend', or 'nonstat')

    Returns
    -------
    se: (N_params,) array
        The standard errors of the MLE parameters
    """

    try:
        get_hess = HESS_REGISTRY[fit_type]

    except KeyError:
        msg = (f"Hessian for fit type {fit_type} not implemented. Either enable --no_se in CLI or implement Hessian in mle/hess.py and add to HESS_REGISTRY.")
        raise ValueError(msg)

    # compute the hessian
    hess = get_hess(theta_opt, temps)

    # invert the Hessian
    # NOTE: we take the negative because we fit to the negative loglik and we want the hessian of the loglik
    inv_hess = np.linalg.inv(-hess)

    # standard errors are the square root of the diagonal of the Hessian
    se = np.sqrt(np.diag(inv_hess))

    return se


if __name__ == "__main__":
    import shutil
    width = shutil.get_terminal_size(fallback=(80, 20)).columns

    from scipy.stats import genextreme
    from mle import _mle_fit
    import time

    # simple test case
    np.random.seed(42)
    sample_sizes = [10**i for i in range(1, 4)]
    non_stat_l2 = []
    stat_l2 = []
    times = []
    print("=" * width)
    print("TEST RESULTS")
    print("=" * width)

    for ss in sample_sizes:
        t0 = time.time()
        # print(f"Sample size: {ss}")
        data = genextreme.rvs(c=-0.1, loc=2, scale=1, size=ss)
        opt_theta = _mle_fit(data, non_stat=False)
        print(opt_theta)
        hess = get_hessian(opt_theta, data, fit_type='stat')
        se = get_standard_errors(opt_theta, data, fit_type='stat')

        print(f"[STAT] Hessian with sample size {ss}: {hess}")
        print(f"[STAT] Standard errors with sample size {ss}: {se}")

    # testing incorporation into xarray framework
    import xarray as xr

    lat = np.array([0, 10])
    lon = np.array([180, 270])
    year = np.arange(1979, 2025, 1)

    t2m = np.zeros((len(year), len(lat), len(lon)))

    for i in range(len(lon)):
        for j in range(len(lat)):
            t2m[:, j, i] = genextreme.rvs(c=-0.1, loc=2, scale=1, size=len(year))

    # create a dummy dataset
    ds = xr.Dataset(
        data_vars={
            't2m': (['year', 'lat', 'lon'], t2m)
        },
        coords={
            'year': (['year'], year),
            'lat': (['lat'], lat),
            'lon': (['lon'], lon)
        }
    )

    da = ds['t2m']

    non_stat = False
    fit_type = 'stat'

    # apply the MLE fit
    gev_params = xr.apply_ufunc(
        _mle_fit,
        da,
        non_stat,
        input_core_dims=[['year'], []],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'gev_params' : 3}
        }
    )

    gev_params = gev_params.assign_coords(gev_params=['loc', 'scale', 'shape'])

    se = xr.apply_ufunc(
        get_standard_errors,
        gev_params,  # (lat, lon, param)
        da,          # (lat, lon, year)
        kwargs={
            'fit_type': fit_type 
        },
        input_core_dims=[['gev_params'], ['year']],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'gev_params': 3 if not non_stat else 6}
        }
    )
    print(f"[STAT] Standard errors from xarray apply_ufunc:\n{se}")


    print("=" * width)
    print("STARTING NONSTAT TESTS")
    print("=" * width)
    
    # single point
    non_stat_hess = _get_hessian_nonstat([1, 1, 0.5, 0.5, -0.05, -0.05],
                                         np.array([5.]), testing=True)
    print(f"[NONSTAT] Hessian with single data point: {non_stat_hess}")

    # apply ufunc
    t2m = np.zeros((len(year), len(lat), len(lon)))

    params_nonstat = [1.0, 0.01, 1.0, 0.02, 0.1, 0.001]
    years_norm = np.arange(0, len(year), 1) / len(year)

    for i in range(len(lon)):
        for j in range(len(lat)):
            t2m[:, j, i] = np.array([genextreme.rvs(
                                        c=-(params_nonstat[4] + params_nonstat[5] * t),
                                        loc=(params_nonstat[0] + params_nonstat[1] * t),
                                        scale=(params_nonstat[2] + params_nonstat[3] * t), size=1)[0] for t in years_norm])

    # create a dummy dataset
    ds = xr.Dataset(
        data_vars={
            't2m': (['year', 'lat', 'lon'], t2m)
        },
        coords={
            'year': (['year'], year),
            'lat': (['lat'], lat),
            'lon': (['lon'], lon)
        }
    )

    da = ds['t2m']

    non_stat = True
    fit_type = 'nonstat'

    # apply the MLE fit
    gev_params = xr.apply_ufunc(
        _mle_fit,
        da,
        non_stat,
        input_core_dims=[['year'], []],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'gev_params' : 6}
        }
    )

    gev_params = gev_params.assign_coords(gev_params=['loc', 'loc_t', 'scale', 'scale_t', 'shape', 'shape_t'])

    se = xr.apply_ufunc(
        get_standard_errors,
        gev_params,  # (lat, lon, param)
        da,          # (lat, lon, year)
        kwargs={
            'fit_type': fit_type 
        },
        input_core_dims=[['gev_params'], ['year']],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'gev_params': 6 if non_stat else 3}
        }
    )
    print(f"[NONSTAT] Standard errors from xarray apply_ufunc:\n{se}")