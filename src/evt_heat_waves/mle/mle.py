"""
Maximum likelihood fitting function for stationary or nonstationary GEV.

Provides:
- top-level function that ingests an xarray dataset and variable name to fit
GEV to
- function to fit GEV at some grid point
- negative log-likelihood of GEV distribution
- GEV PDF

Last edited: 4/30/2026, 5:12 PM CST
"""

import warnings

# ignore divide by zero / overflow warnings that pop up during
# scipy.optimize.minimize calls and don't really impact performance
warnings.simplefilter('ignore', RuntimeWarning)

import numpy as np
import xarray as xr

from scipy.optimize import minimize
from evt_heat_waves.config import MLE_FIT_ATTRS, MLE_FULL_PARAM_NAMES
from evt_heat_waves.mle.utils import get_bounds, get_constraints, get_initial_guess
from evt_heat_waves.mle.grad import grad_negative_log_likelihood
from evt_heat_waves.mle.se import get_standard_errors


def ds_mle_fit(args, ds, var_name, fit_dim='year',
               all_mems=False):
    """Fit (potentially nonstationary) GEV distribution to each (lat, lon) pair
    of an xarray Dataset via maximum likelihood estimation.

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

    all_mems: bool
        fit to all ensemble members? (only applicable for one CMIP model)

    Returns
    -------
    ds: xarray.Dataset
        the input dataset with added GEV parameters as new variables
    """

    # subselect variable to do the fitting over
    da = ds[var_name]

    # number of params in this fit
    N_PARAMS = MLE_FIT_ATTRS[args.fit]['param_names']

    # set ufunc keyword args
    ufunc_kwargs_fit = dict(
        input_core_dims=[[fit_dim], []],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'gev_params': N_PARAMS}
            }
    )

    # carry out either parallelized or non-parallelized fit
    gev_params = xr.apply_ufunc(
        _mle_fit,
        da,
        args.fit,
        **ufunc_kwargs_fit
    )

    # assign parameter names to gev_params coordinate
    gev_params = gev_params.assign_coords(gev_params=N_PARAMS)

    # set ufunc attrs for se calc
    ufunc_kwargs_se = dict(input_core_dims=[['gev_params'], ['year']],
        output_core_dims=[['gev_params']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'gev_params': N_PARAMS}
        })

    # if we also calculate the standard errors, compute, otherwise fill nans
    # then map into dataset
    if not args.no_se:
        gev_se = xr.apply_ufunc(
            get_standard_errors,
            gev_params,  # (lat, lon, param)
            da,          # (lat, lon, year)
            kwargs={
                'fit_type': args.fit 
            },
            **ufunc_kwargs_se
        )

        # assign parameter names
        ds = _assign_params(args, ds, var_name, gev_params, gev_se, all_mems)
    
    else:
        # assign parameter names, with SEs set to NaN
        gev_se = xr.full_like(gev_params, np.nan)
        ds = _assign_params(args, ds, var_name, gev_params, gev_se, all_mems)

    # return the amended dataset
    return ds


def _mle_fit(data, fit_type='stat', SAMPLE_THRES=10):
    """Fit a potentiallly nonstationary GEV distribution to data via MLE.
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
        return np.array([np.nan] * MLE_FIT_ATTRS[fit_type]['N_params'])
    
    # set up initial guess, constraints, and bounds for MLE fit for fit type
    initial_guess = get_initial_guess(data)
    cons = get_constraints(fit_type)
    bounds = get_bounds(fit_type)

    # do MLE fit
    fit = minimize(_negative_log_likelihood,
                    initial_guess,
                    args=(data),
                    method='SLSQP',  # SLSQP to allow for constraints
                    constraints=cons,
                    bounds=bounds,
                    # jac=grad_negative_log_likelihood
                    )

    # if the fit is successful, return parameters, else return nans
    if fit.success:
        _mle_fit.success_count += 1
        return _extract_params(np.array(fit.x), fit_type)
    else:
        _mle_fit.fail_count += 1
        return np.array([np.nan] * MLE_FIT_ATTRS[fit_type]['N_params'])


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


def _negative_log_likelihood(params, data):
    """Get the negative log likelihood of the nonstationary GEV (with some params)
    set to zero by SLSQP constraints.

    Parameters
    ----------
    params: (6,) list
        the GEV params
    
    data: array-like
        the data to fit the GEV to
    
    Returns
    -------
    log_likelihood: float
        the total NEGATIVE log likelihood
    """
    loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params

    time = np.arange(0, len(data), 1) / len(data)  # normalized time variable

    # get the log likelihood
    log_likelihood = - np.sum(
        np.log([_gev_pdf(x=x,
                      loc=loc_0 + loc_1 * t,
                      scale=scale_0 + scale_1 * t,
                      shape=shape_0 + shape_1 * t) for x, t in zip(data, time)]
        )
    )

    return log_likelihood


def _gev_pdf(x, loc, scale, shape,
                 ret_nan=False, pen=np.exp(-40)):
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
    

def _extract_params(fit_x, fit_type):
    """Extract active parameters from the full MLE solution vector.

    The MLE always solves over MLE_FULL_PARAM_NAMES (length 6). This returns
    only the indices corresponding to the fit_type's param_names, in order.
    """
    param_names = MLE_FIT_ATTRS[fit_type]['param_names']
    idx = [MLE_FULL_PARAM_NAMES.index(p) for p in param_names]
    return fit_x[idx]
    

def _assign_params(args, ds, var_name, gev_params, gev_se, all_mems):
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

    # look up param names from global config
    if args.fit not in MLE_FIT_ATTRS:
        raise ValueError(
            f"Unknown fit_type {args.fit}. "
            f"Expected one of: {list(MLE_FIT_ATTRS)}"
        )
    param_names = [f'{p}_{sfx}' for p in MLE_FIT_ATTRS[args.fit]['param_names']]

    # assign each parameter and its SE to the dataset
    for i, pname in enumerate(param_names):
        ds = ds.assign({pname:         (spatial_dims, gev_params.data[..., i])})
        ds = ds.assign({f'se_{pname}': (spatial_dims, gev_se.data[..., i])})

    return ds


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

        tmp_stat_l2 = np.sqrt(
            (params[0] - 2)**2 +
            (params[1] - 1)**2 +
            (params[2] - 0.1)**2
        )
        stat_l2.append(tmp_stat_l2)
        print(f"Stationary fit params: {params}")

        params_nonstat = [1.0, 0.01, 1.0, 0.02, 0.1, 0.001]
        years = np.arange(0, ss, 1) / ss  # 100 years of real data
        data_nonstat = np.array([genextreme.rvs(
            c=-(params_nonstat[4] + params_nonstat[5] * t),
            loc=(params_nonstat[0] + params_nonstat[1] * t),
            scale=(params_nonstat[2] + params_nonstat[3] * t),
            size=1)[0] for t in years])
    
        fitted_param_nonstat = _mle_fit(data_nonstat, SAMPLE_THRES=10, non_stat=True)
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