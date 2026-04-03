"""
Maximum likelihood fitting function for stationary or nonstationary GEV.

Provides:
- top-level function that ingests an xarray dataset and variable name to fit
  GEV to
- function to fit GEV at some grid point
- negative log-likelihood of GEV distribution
- GEV PDF
- Hessian-based standard errors for MLE parameter estimates

Last edited: 4/3/2026
"""

import warnings

import numpy as np
import xarray as xr

# ignore divide by zero / overflow warnings that pop up during
# scipy.optimize.minimize calls and don't really impact performance
warnings.simplefilter('ignore', RuntimeWarning)

from scipy.optimize import minimize
from scipy.differentiate import hessian as scipy_hessian


# ============================================================
# Top-level fitting function
# ============================================================

def ds_mle_fit(ds, var_name, fit_dim='year', non_stat=False, all_mems=False, parallel=True):
    """Fit (potentially nonstationary) GEV distribution to each (lat, lon) pair
    of an xarray Dataset via maximum likelihood estimation.

    Parameters
    ----------
    ds: xarray.Dataset
        the input dataset containing the data to fit

    var_name: str
        the variable name in the dataset to fit the GEV distribution to

    fit_dim: str
        the dimension over which to fit the GEV distribution (e.g., 'year')

    non_stat: bool
        whether to fit a nonstationary GEV (True) or stationary GEV (False)

    all_mems: bool
        fit to all ensemble members? (only applicable for one CMIP model)

    parallel: bool
        whether to use dask parallelization for the fitting

    Returns
    -------
    ds: xarray.Dataset
        the input dataset with added GEV parameters and their standard errors
        as new variables
    """

    # subselect variable to do the fitting over
    da = ds[var_name]

    if non_stat:
        N_param_dims = 6
    else:
        N_param_dims = 3

    # shared kwargs for both fitting and SE apply_ufunc calls
    ufunc_kwargs = dict(
        input_core_dims=[[fit_dim], []],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'gev_params': N_param_dims}}
    )

    # carry out either parallelized or non-parallelized fit
    if parallel:
        gev_params = xr.apply_ufunc(
            _mle_fit,
            da,
            non_stat,
            **ufunc_kwargs
        )

    else:
        gev_params = xr.apply_ufunc(
            _mle_fit,
            da,
            non_stat,
            **ufunc_kwargs  # fixed: was missing comma before vectorize=True in original
        )

    # ---- Standard errors via Hessian -----------------------------------
    # SE output always has N_param_dims entries (NaN-masked for frozen params
    # in the stationary case)
    se_ufunc_kwargs = dict(
        input_core_dims=[[fit_dim], ['gev_params'], []],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'gev_params': N_param_dims}}
    )

    gev_se = xr.apply_ufunc(
        _mle_se,
        da,
        gev_params,
        non_stat,
        **se_ufunc_kwargs
    )

    # assign shape, loc, and scale parameters (and their SEs) to the dataset.
    # since we fit over all ensemble members, we need another : compared to the
    # non-all_mems case because MLE fitting output has an additional dimension
    # (mem, lat, lon, param) instead of (lat, lon, param)
    ds = _assign_params(ds, var_name, gev_params, gev_se, non_stat, all_mems)

    # return the amended dataset
    return ds


# ============================================================
# Dataset variable assignment (parameters + SEs)
# ============================================================

def _assign_params(ds, var_name, gev_params, gev_se, non_stat, all_mems):
    """Assign fitted GEV parameters and their standard errors to the dataset.

    Resolves the spatial dimensions and variable name suffix from the calling
    context, then loops over the parameter list to assign each array and its
    corresponding SE in one place rather than repeating ds.assign() calls for
    every (var_name, non_stat, all_mems) combination.
    """

    # spatial dims depend on whether we're fitting across ensemble members
    spatial_dims = ('member_id', 'lat', 'lon') if all_mems else ('lat', 'lon')

    # map variable name to the suffix used in dataset variable names
    suffix_map = {
        't2m':             'raw',
        'tas':             'raw',
        't2m_anom_annmean':'anom_annmean',
        't2m_anom_trend':  'anom_trend',
    }
    sfx = suffix_map.get(var_name, var_name)

    # build ordered list of (param_name, param_index) pairs matching the order
    # that _mle_fit returns parameters in its output array
    if non_stat:
        param_names = [
            f'loc_{sfx}',     f'loc_t_{sfx}',
            f'scale_{sfx}',   f'scale_t_{sfx}',
            f'shape_{sfx}',   f'shape_t_{sfx}',
        ]
    else:
        param_names = [f'loc_{sfx}', f'scale_{sfx}', f'shape_{sfx}']

    # assign each parameter and its SE to the dataset
    for i, pname in enumerate(param_names):
        ds = ds.assign({pname:        (spatial_dims, gev_params.data[..., i])})
        ds = ds.assign({f'se_{pname}': (spatial_dims, gev_se.data[..., i])})

    return ds


# ============================================================
# Standard error computation (single grid point)
# ============================================================

def _mle_se(data, params, non_stat=False):
    """Compute standard errors for MLE GEV parameters at one (lat, lon) point
    via the observed Fisher information (negative inverse Hessian of the NLL).

    Parameters
    ----------
    data : 1-D array of shape (N_years,)
    params : 1-D array of MLE parameters
        non_stat=False → length 3: [loc_0, scale_0, shape_0]
        non_stat=True  → length 6: [loc_0, loc_1, scale_0, scale_1, shape_0, shape_1]

    Returns
    -------
    se : 1-D array, same length as params (NaN where computation failed)
    """
    # return NaNs if the upstream fit already failed
    if np.any(~np.isfinite(params)):
        return np.full_like(params, np.nan)

    # only take finite values
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return np.full_like(params, np.nan)

    # Build the full 6-parameter vector that _negative_log_likelihood expects.
    # For the stationary case the optimizer returns only [loc_0, scale_0, shape_0];
    # we reconstruct the frozen 6-vector so the NLL signature is always satisfied.
    if non_stat:
        # params already: [loc_0, loc_1, scale_0, scale_1, shape_0, shape_1]
        params_full = np.array(params, dtype=float)
        active_idx = np.arange(6)           # all 6 parameters are free
    else:
        # params: [loc_0, scale_0, shape_0] — reconstruct full 6-vector with
        # trend parameters frozen to zero, matching the stationary constraints
        loc_0, scale_0, shape_0 = params
        params_full = np.array([loc_0, 0.0, scale_0, 0.0, shape_0, 0.0], dtype=float)
        active_idx = np.array([0, 2, 4])    # only loc_0, scale_0, shape_0 are free

    # wrap NLL so scipy.differentiate.hessian gets f: (m, ...) -> (...)
    # the time normalization must match _negative_log_likelihood exactly
    def nll_for_hessian(x):
        return _negative_log_likelihood(x, data, non_stat=True)
        # we always pass non_stat=True here because we're always evaluating
        # all 6 parameters; the stationary constraint is enforced by fixing
        # the inactive entries, not by branching inside the NLL

    try:
        res = scipy_hessian(nll_for_hessian, params_full)
        H = res.ddf                                     # (6, 6)

        if not np.all(np.isfinite(H)):
            return np.full(len(params), np.nan)

        # for the stationary case, extract the (3, 3) sub-block corresponding
        # to the free parameters before inverting — the frozen rows/cols have
        # near-zero curvature and make the full matrix nearly singular
        H_active = H[np.ix_(active_idx, active_idx)]   # (3,3) or (6,6)

        # check positive-definiteness (H = d²NLL/dθ² should be PD at a minimum)
        eigvals = np.linalg.eigvalsh(H_active)
        if not np.all(eigvals > 0):
            return np.full(len(params), np.nan)

        cov_active = np.linalg.inv(H_active)            # Cramér-Rao covariance
        var_active = np.diag(cov_active)

        # guard against numerical noise giving tiny negative variances
        se_active = np.where(var_active > 0, np.sqrt(var_active), np.nan)

        # map back to the output length (3 or 6)
        se_out = np.full(len(params), np.nan)
        se_out[np.arange(len(params))] = se_active      # 1-to-1 for non_stat
        # for stationary, active_idx = [0,2,4] → se_out indices [0,1,2]
        if not non_stat:
            se_out = se_active                           # length 3, already aligned

        return se_out

    except Exception:
        return np.full(len(params), np.nan)


# ============================================================
# Single grid-point MLE fit
# ============================================================

def _mle_fit(data, non_stat=False, SAMPLE_THRES=10):
    """Fit a potentially nonstationary GEV distribution to data via MLE.
    """

    # on first call, give the function these two attributes to track
    # success and failure of MLE across grid points.
    if not hasattr(_mle_fit, 'success_count'):
        _mle_fit.success_count = 0
        _mle_fit.fail_count = 0

    # only take finite values
    data = data[np.isfinite(data)]

    # if the number of points is less than the sample threshold,
    # return NaNs for the fitted parameters
    if len(data) < SAMPLE_THRES:
        if non_stat:
            return np.array([np.nan] * 6)
        else:
            return np.array([np.nan] * 3)

    # try to do the GEV distribution fit and return parameters
    # tune initial guess based on the moments of the samples
    samp_mean = np.mean(data)
    samp_std = np.std(data)
    EULER_CONST = 0.5772156649

    scale_guess = samp_std * np.sqrt(6) / np.pi  # initial guess for scale
    loc_guess = samp_mean + scale_guess * EULER_CONST  # initial guess for loc
    shape_guess = -0.1  # initial guess for shape

    initial_guess = [loc_guess, 0.0,  # loc parameters
                     scale_guess, 0.0,  # scale parameters
                     shape_guess, 0.0  # shape parameters
    ]

    # set up constraints for MLE fit for nonstationary or stationary cases
    # nonstationary just requires the scale parameter to be positive for all time
    ## turns out, this requires two constraints: scale_0 >= 0 and scale_0 + scale_1 * T >= 0
    # stationary sets the trend in parameters to zero, and keeps the scale parameter positive
    if non_stat:
        cons = ({'type': 'ineq',
                 'fun': lambda x: x[2] + x[3]}  # scale_0 + scale_1 * time >= 0
                 )

    else:
        cons = ({'type': 'eq',
                 'fun': lambda x: x[1]},  # loc_1 = 0 (no trend)
                {'type': 'eq',
                 'fun': lambda x: x[3]},  # scale_1 = 0 (no trend)
                {'type': 'eq',
                 'fun': lambda x: x[5]})  # shape_1 = 0 (no trend)

    bounds = ((None, None),
            (None, None),
            (0.0, None),  # sigma >= 0 (this bound + inequality constraint above ensures \sigma_t >= for all t when fit is nonstationary)
            (None, None),
            (-1, 1),  # bar{xi} can't be too big or MLE is unstable
            (-1, 1))  # xi' can't be too big for same reason

    # do MLE fit
    fit = minimize(_negative_log_likelihood,
                    initial_guess,
                    args=(data, non_stat),
                    method='SLSQP',  # SLSQP to allow for constraints
                    constraints=cons,
                    bounds=bounds,
                    # jac=_grad_negative_log_likelihood
                    )

    # if the fit is successful, return parameters, else return nans
    if fit.success:
        # print("MLE fit successful.")
        _mle_fit.success_count += 1
        total = _mle_fit.success_count + _mle_fit.fail_count
        success_rate = _mle_fit.success_count / total
        # print(f"\r  ↳ MLE Success rate: {_mle_fit.success_count}/{total} ({success_rate:.1%})", end='', flush=True)
        if non_stat:
            return np.array(fit.x)  # return all 6 parameters
        else:
            return np.array([fit.x[0], fit.x[2], fit.x[4]])  # loc_0, scale_0, shape_0

    else:
        #print("WARNING: MLE fit failed: {}".format(fit.message))
        _mle_fit.fail_count += 1
        total = _mle_fit.success_count + _mle_fit.fail_count
        success_rate = _mle_fit.success_count / total
        # print(f"\r  ↳ MLE Success rate: {_mle_fit.success_count}/{total} ({success_rate:.1%})", end='', flush=True)
        if non_stat:
            return np.array([np.nan] * 6)  # return nans for failed fit
        else:
            return np.array([np.nan] * 3)  # return nans for failed fit


# ============================================================
# Utility: MLE diagnostics
# ============================================================

def reset_mle_stats(silent=True):
    """Reset MLE function stats.
    """
    _mle_fit.success_count = 0
    _mle_fit.fail_count = 0
    if not silent:
        print("\nMLE stats reset.")


def get_mle_success_rate():
    """Report the MLE success rate.
    """
    total = _mle_fit.success_count + _mle_fit.fail_count
    return _mle_fit.success_count / total  # success rate of MLE algorithm


# ============================================================
# Negative log-likelihood and PDF
# ============================================================

def _negative_log_likelihood(params, data, non_stat=False):
    if non_stat:
        loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
    else:
        loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
        loc_1 = scale_1 = shape_1 = 0

    time = np.arange(0, len(data), 1) / len(data)  # normalized time variable

    log_likelihood = - np.sum(
        np.log([_gev_pdf(x=x,
                      loc=loc_0 + loc_1 * t,
                      scale=scale_0 + scale_1 * t,
                      shape=shape_0 + shape_1 * t) for x, t in zip(data, time)]
        )
    )

    return log_likelihood


def _gev_pdf(x, loc, scale, shape,
                 ret_nan=False, pen=np.exp(-50)):
    """Compute the PDF of the GEV distribution at some point x.

    Function returns the PDF value at point x for parameters (loc, scale, shape),
    or a penalty of pen when x lies outside the range of support for the PDF.

    Parameters
    ----------
    x: array-like
        Points to evaluate the PDF at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    ret_nan: bool (=False)
        return nan instead of penalty if x is outside the support of the PDF

    pen: float (=exp(-50))
        penalty (usually really really small, so log-likelihood is really big)

    Returns
    -------
    pdf: array-like
        PDF evaluated at x

    Examples
    --------
    >>> _gev_pdf_pen(2, 1.0, 0.5, -0.1)
    0.24110591428617528

    >>> _gev_pdf_pen(10, 1.0, 0.5, -0.1)
    1.9287498479639178e-22  # penalty returns really pen (= np.exp(-50)) when x outside support

    >>> _gev_pdf_pen(10, 1.0, 0.5, -0.1, ret_nan=True)
    nan  # turning on ret_nan returns nan instead of low PDF outside support
    """

    if shape > 0:
        support_lb = loc - scale / shape
        if x < support_lb:
            if ret_nan:
                return np.nan  # returning nan sometimes more convenient for evaluation than large penalty
            else:
                return pen  # large penalty for unsupported values
        else:
            s = (x - loc) / scale  # standardized variable

            t_x = (1 + shape * s)**(-1 / shape)  # transformation to Frechet case (assuming scale !=0)

            # eval PDF
            pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
            return pdf

    elif shape < 0:
        support_ub = loc - scale / shape
        if x > support_ub:
            if ret_nan:
                return np.nan
            else:
                return pen  # large penalty for unsupported values

        else:
            s = (x - loc) / scale  # standardized variable

            t_x = (1 + shape * s)**(-1 / shape)  # transformation to reveresed Weibull case (assuming scale !=0)

            # eval PDF
            pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
            return pdf

    else:
        s = (x - loc) / scale  # standardized variable

        if shape == 0:
            t_x = np.exp(-s)  # transformation for Gumbel case
        else:
            t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

        # eval PDF
        pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
        return pdf


# ============================================================
# Test / __main__
# ============================================================

# test cases
if __name__ == '__main__':
    import pandas as pd
    from scipy.stats import genextreme
    import time

    # simple test case
    np.random.seed(42)
    sample_sizes = [10**i for i in range(1, 5)]
    non_stat_l2 = []
    stat_l2 = []
    times = []
    for ss in sample_sizes:
        t0 = time.time()
        print(f"Sample size: {ss}")
        data = genextreme.rvs(c=-0.1, loc=2, scale=1, size=ss)
        params = _mle_fit(data, SAMPLE_THRES=10, non_stat=False)
        se     = _mle_se(data, params, non_stat=False)

        tmp_stat_l2 = np.sqrt(
            (params[0] - 2)**2 +
            (params[1] - 1)**2 +
            (params[2] - 0.1)**2
        )
        stat_l2.append(tmp_stat_l2)
        print(f"Stationary fit params: {params}")
        print(f"Stationary fit SEs:    {se}")

        params_nonstat = [1.0, 0.01, 1.0, 0.02, 0.1, 0.001]
        years = np.arange(0, ss, 1) / ss  # 100 years of real data
        data_nonstat = np.array([genextreme.rvs(
            c=-(params_nonstat[4] + params_nonstat[5] * t),
            loc=(params_nonstat[0] + params_nonstat[1] * t),
            scale=(params_nonstat[2] + params_nonstat[3] * t),
            size=1)[0] for t in years])

        fitted_param_nonstat = _mle_fit(data_nonstat, SAMPLE_THRES=10, non_stat=True)
        se_nonstat           = _mle_se(data_nonstat, fitted_param_nonstat, non_stat=True)

        tmp_nonstat_l2 = np.sqrt(
            (fitted_param_nonstat[0] - params_nonstat[0])**2 +
            (fitted_param_nonstat[1] - params_nonstat[1])**2 +
            (fitted_param_nonstat[2] - params_nonstat[2])**2 +
            (fitted_param_nonstat[3] - params_nonstat[3])**2 +
            (fitted_param_nonstat[4] - params_nonstat[4])**2 +
            (fitted_param_nonstat[5] - params_nonstat[5])**2
        )
        non_stat_l2.append(tmp_nonstat_l2)
        print(f"Nonstationary fit params: {fitted_param_nonstat}")
        print(f"Nonstationary fit SEs:    {se_nonstat}")

        # record elapsed time for this iteration
        times.append(time.time() - t0)

    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'sample_size': sample_sizes,
        'stationary_l2_error': stat_l2,
        'nonstationary_l2_error': non_stat_l2,
        'iteration_time': times
    })

    # print dataframe and save to data/checks
    print(df)

    # df.to_csv('data/checks/mle_sample_size_l2_stat_nonstat.csv', index=False)
    # print("\nDataFrame saved to data/checks/mle_sample_size_l2_stat_nonstat.csv")

    """
    import matplotlib.pyplot as plt 

    xs = np.arange(-100, 100, 1.)
    ys = np.log([_gev_pdf(x, loc=20, scale=1.5, shape=0.25)
                 for x in xs])

    plt.plot(xs, ys)
    plt.show()
    """