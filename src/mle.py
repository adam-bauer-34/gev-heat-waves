"""
Maximum likelihood fitting function for stationary or nonstationary GEV.

Provides:
- top-level function that ingests an xarray dataset and variable name to fit
GEV to
- function to fit GEV at some grid point
- negative log-likelihood of GEV distribution
- GEV PDF

Last edited: 1/29/2026, 10:22 AM CST
"""

import warnings

import numpy as np
import xarray as xr

# ignore divide by zero / overflow warnings that pop up during
# scipy.optimize.minimize calls and don't really impact performance
warnings.simplefilter('ignore', RuntimeWarning)

from scipy.optimize import minimize

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
        the input dataset with added GEV parameters as new variables
    """

    # subselect variable to do the fitting over
    da = ds[var_name]

    if non_stat:
        N_param_dims = 6
    else:
        N_param_dims = 3

    # carry out either parallelized or non-parallelized fit
    if parallel:
        gev_params = xr.apply_ufunc(
            _mle_fit,
            da,
            non_stat,
            input_core_dims=[[fit_dim], []],
            output_core_dims=[['gev_params']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={
                'output_sizes': {'gev_params' : N_param_dims}
            }
        )

    else:
        gev_params = xr.apply_ufunc(
            _mle_fit,
            da,
            non_stat,
            input_core_dims=[[fit_dim], []],
            output_core_dims=[['gev_params']]
        )

    if non_stat and not all_mems:
        # assign shape, loc, and scale parameters to their (lat, lon) coords
        if var_name == 't2m' or var_name == 'tas':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_raw = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(loc_t_raw = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(scale_raw = (('lat', 'lon'), gev_params.data[:, :, 2]))
            ds = ds.assign(scale_t_raw = (('lat', 'lon'), gev_params.data[:, :, 3]))
            ds = ds.assign(shape_raw = (('lat', 'lon'), gev_params.data[:, :, 4]))
            ds = ds.assign(shape_t_raw = (('lat', 'lon'), gev_params.data[:, :, 5]))

        elif var_name == 't2m_anom_annmean':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(loc_t_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(scale_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 2]))
            ds = ds.assign(scale_t_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 3]))
            ds = ds.assign(shape_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 4]))
            ds = ds.assign(shape_t_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 5]))

        elif var_name == 't2m_anom_trend':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(loc_t_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(scale_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 2]))
            ds = ds.assign(scale_t_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 3]))
            ds = ds.assign(shape_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 4]))
            ds = ds.assign(shape_t_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 5]))

    elif not non_stat and not all_mems:
        if var_name == 't2m' or var_name == 'tas':
            ds = ds.assign(loc_raw = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(scale_raw = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(shape_raw = (('lat', 'lon'), gev_params.data[:, :, 2]))

        elif var_name == 't2m_anom_annmean':
            ds = ds.assign(loc_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(scale_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(shape_anom_annmean = (('lat', 'lon'), gev_params.data[:, :, 2]))

        elif var_name == 't2m_anom_trend':
            ds = ds.assign(loc_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(scale_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(shape_anom_trend = (('lat', 'lon'), gev_params.data[:, :, 2]))

    # since we fit over all ensemble members, we need another : compared to the above
    # because MLE fitting output has an additional dimension
    # (mem, lat, lon, param) instead of (lat, lon, param)
    elif non_stat and all_mems:
        # assign shape, loc, and scale parameters to their (lat, lon) coords
        if var_name == 't2m' or var_name == 'tas':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 0]))
            ds = ds.assign(loc_t_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 1]))
            ds = ds.assign(scale_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 2]))
            ds = ds.assign(scale_t_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 3]))
            ds = ds.assign(shape_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 4]))
            ds = ds.assign(shape_t_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 5]))

        elif var_name == 't2m_anom_annmean':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 0]))
            ds = ds.assign(loc_t_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 1]))
            ds = ds.assign(scale_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 2]))
            ds = ds.assign(scale_t_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 3]))
            ds = ds.assign(shape_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 4]))
            ds = ds.assign(shape_t_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 5]))

        elif var_name == 't2m_anom_trend':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 0]))
            ds = ds.assign(loc_t_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 1]))
            ds = ds.assign(scale_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 2]))
            ds = ds.assign(scale_t_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 3]))
            ds = ds.assign(shape_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 4]))
            ds = ds.assign(shape_t_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 5]))

    elif not non_stat and all_mems:
        if var_name == 't2m' or var_name == 'tas':
            ds = ds.assign(loc_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 0]))
            ds = ds.assign(scale_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 1]))
            ds = ds.assign(shape_raw = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 2]))

        elif var_name == 't2m_anom_annmean':
            ds = ds.assign(loc_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 0]))
            ds = ds.assign(scale_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 1]))
            ds = ds.assign(shape_anom_annmean = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 2]))

        elif var_name == 't2m_anom_trend':
            ds = ds.assign(loc_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 0]))
            ds = ds.assign(scale_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 1]))
            ds = ds.assign(shape_anom_trend = (('member_id', 'lat', 'lon'), gev_params.data[:, :, :, 2]))

    # return the amended dataset
    return ds

def _mle_fit(data, non_stat=False, SAMPLE_THRES=10):
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
        print(f"\r  ↳ MLE Success rate: {_mle_fit.success_count}/{total} ({success_rate:.1%})", end='', flush=True)
        if non_stat:
            return np.array(fit.x)  # return all 6 parameters
        else:
            return np.array([fit.x[0], fit.x[2], fit.x[4]])  # loc_0, scale_0, shape_0
    
    else:
        #print("WARNING: MLE fit failed: {}".format(fit.message))
        _mle_fit.fail_count += 1
        total = _mle_fit.success_count + _mle_fit.fail_count
        success_rate = _mle_fit.success_count / total
        print(f"\r  ↳ MLE Success rate: {_mle_fit.success_count}/{total} ({success_rate:.1%})", end='', flush=True)
        if non_stat:
            return np.array([np.nan] * 6)  # return nans for failed fit
        else:
            return np.array([np.nan] * 3)  # return nans for failed fit


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
    
def _grad_negative_log_likelihood(params, data, non_stat=False):
    """Analytic gradients of the negative log-likelihood function.
    """

    grad = np.zeros_like(params)

    if non_stat:
        loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
    else:
        loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
        loc_1 = scale_1 = shape_1 = 0

    time = np.arange(0, len(data), 1) / len(data)  # normalized time variable

    # compute the gradient of each stationary component
    grad[0] = np.sum(
        [
            _gev_negloglik_grad_loc0(x=x,
                                    loc_0=loc_0,
                                    loc_1=loc_1,
                                    scale_0=scale_0,
                                    scale_1=scale_1,
                                    shape_0=shape_0,
                                    shape_1=shape_1,
                                    time=t) for x, t in zip(data, time)
        ]
    )
    
    grad[2] = np.sum(
        [
            _gev_negloglik_grad_scale0(x=x,
                                      loc_0=loc_0,
                                      loc_1=loc_1,
                                      scale_0=scale_0,
                                      scale_1=scale_1,
                                      shape_0=shape_0,
                                      shape_1=shape_1,
                                      time=t) for x, t in zip(data, time)
        ]
    )

    grad[4] = np.sum(
        [
            _gev_negloglik_grad_shape0(x=x,
                                       loc_0=loc_0,
                                       loc_1=loc_1,
                                       scale_0=scale_0,
                                       scale_1=scale_1,
                                       shape_0=shape_0,
                                       shape_1=shape_1,
                                       time=t) for x, t in zip(data, time)
        ]
    )
    
    # if nonstationry, compute the gradient for each trend bit
    if non_stat:
        grad[1] = np.sum(
            [
                _gev_negloglik_grad_loc1(x=x,
                                         loc_0=loc_0,
                                         loc_1=loc_1,
                                         scale_0=scale_0,
                                         scale_1=scale_1,
                                         shape_0=shape_0,
                                         shape_1=shape_1,
                                         time=t) for x, t in zip(data, time)
            ]
        )
        
        grad[3] = np.sum(
            [
                _gev_negloglik_grad_scale1(x=x,
                                           loc_0=loc_0,
                                           loc_1=loc_1,
                                           scale_0=scale_0,
                                           scale_1=scale_1,
                                           shape_0=shape_0,
                                           shape_1=shape_1,
                                           time=t) for x, t in zip(data, time)
            ]
        )
        
        grad[5] = np.sum(
            [
                _gev_negloglik_grad_shape1(x=x,
                                           loc_0=loc_0,
                                           loc_1=loc_1,
                                           scale_0=scale_0,
                                           scale_1=scale_1,
                                           shape_0=shape_0,
                                           shape_1=shape_1,
                                           time=t) for x, t in zip(data, time)
            ]
        )

    return grad

def _gev_negloglik_grad_loc0(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Gradient of negative log-likelihood with respect to loc_0.
    Placeholder implementation that returns zeros of the same shape as x.
    """

    loc = loc_0 + loc_1 * time
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time

    if shape > 0:
        support_lb = loc - scale / shape
        if x < support_lb:
            return 0.0  # grad = 0 for unsupported values
        else:
            tx = _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)
            dtx_dloc = _dhepler_dloc(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)

            piece1 = (1 + 1 / shape) * dtx_dloc / tx
            piece2 = - tx**(-1 - 1/shape) * dtx_dloc / shape

            return piece1 + piece2
        
    elif shape < 0:
        support_ub = loc - scale / shape
        if x > support_ub:
            return 0.0  # grad = 0 for unsupported values
            
        else:
            tx = _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)
            dtx_dloc = _dhepler_dloc(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)

            piece1 = (1 + 1 / shape) * dtx_dloc / tx
            piece2 = - tx**(-1 - 1/shape) * dtx_dloc / shape

            return piece1 + piece2
    
    ## NOT FUNCTIONAL YET -- WILL ADD SPECIAL CASE FOR GUMBEL DISTRIBUTION LATER
    else:
        s = (x - loc) / scale  # standardized variable

        if shape == 0:
            t_x = np.exp(-s)  # transformation for Gumbel case
        else:
            t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

        # eval PDF
        pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
        return pdf


def _gev_negloglik_grad_loc1(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Gradient of negative log-likelihood with respect to loc_1 (trend).
    Placeholder implementation that returns zeros of the same shape as x.
    """
    return time * _gev_negloglik_grad_loc0(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)


def _gev_negloglik_grad_scale0(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Gradient of negative log-likelihood with respect to scale_0.
    Placeholder implementation that returns zeros of the same shape as x.
    """

    loc = loc_0 + loc_1 * time
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time

    if shape > 0:
        support_lb = loc - scale / shape
        if x < support_lb:
            return 0.0  # grad = 0 for unsupported values
        else:
            tx = _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)
            dtx_dscale = _dhepler_dscale(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)

            piece1 = 1 / scale
            piece2 = (1 + 1 / shape) * dtx_dscale / tx
            piece3 = - tx**(-1 - 1/shape) * dtx_dscale / shape

            return piece1 + piece2 + piece3
        
    elif shape < 0:
        support_ub = loc - scale / shape
        if x > support_ub:
            return 0.0  # grad = 0 for unsupported values
            
        else:
            tx = _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)
            dtx_dscale = _dhepler_dscale(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)

            piece1 = 1 / scale
            piece2 = (1 + 1 / shape) * dtx_dscale / tx
            piece3 = - tx**(-1 - 1/shape) * dtx_dscale / shape

            return piece1 + piece2 + piece3
    
    ## NOT FUNCTIONAL YET -- WILL ADD SPECIAL CASE FOR GUMBEL DISTRIBUTION LATER
    else:
        s = (x - loc) / scale  # standardized variable

        if shape == 0:
            t_x = np.exp(-s)  # transformation for Gumbel case
        else:
            t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

        # eval PDF
        pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
        return pdf


def _gev_negloglik_grad_scale1(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Gradient of negative log-likelihood with respect to scale_1 (trend).
    Placeholder implementation that returns zeros of the same shape as x.
    """
    return time * _gev_negloglik_grad_scale0(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)


def _gev_negloglik_grad_shape0(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Gradient of negative log-likelihood with respect to shape_0.
    Placeholder implementation that returns zeros of the same shape as x.
    """
    loc = loc_0 + loc_1 * time
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time


    if shape > 0:
        support_lb = loc - scale / shape
        if x < support_lb:
            return 0.0  # grad = 0 for unsupported values
        
        else:
            tx = _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)
            dtx_dshape = _dhepler_dshape(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)

            piece1 = (-1/shape**2) * np.log(tx)
            piece2 = (1 + 1/shape) * dtx_dshape / tx
            piece3 = tx**(-1/shape) * (
                (1/shape**2) * np.log(tx) - dtx_dshape / (shape * tx)
            )

            return piece1 + piece2 + piece3
        
    elif shape < 0:
        support_ub = loc - scale / shape
        if x > support_ub:
            return 0.0  # grad = 0 for unsupported values
        
        else:
            tx = _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)
            dtx_dshape = _dhepler_dshape(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)

            piece1 = (-1/shape**2) * np.log(tx)
            piece2 = (1 + 1/shape) * dtx_dshape / tx
            piece3 = tx**(-1/shape) * (
                (1/shape**2) * np.log(tx) - dtx_dshape / (shape * tx)
            )

            return piece1 + piece2 + piece3
    
    ## NOT FUNCTIONAL YET -- WILL ADD SPECIAL CASE FOR GUMBEL DISTRIBUTION LATER
    else:
        s = (x - loc) / scale  # standardized variable

        if shape == 0:
            t_x = np.exp(-s)  # transformation for Gumbel case
        else:
            t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

        # eval PDF
        pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
        return pdf


def _gev_negloglik_grad_shape1(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Gradient of negative log-likelihood with respect to shape_1 (trend).
    Placeholder implementation that returns zeros of the same shape as x.
    """
    return time * _gev_negloglik_grad_shape0(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time)


def _helper(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Helper function to compute standardized variable and transformation.
    """

    loc = loc_0 + loc_1 * time
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time

    tx = 1 + shape * (x - loc) / scale
    return tx


def _dhepler_dloc(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Helper function to compute derivative of standardized variable transformation
    with respect to loc parameter.
    """
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time

    dtx_dloc = -shape / scale
    return dtx_dloc


def _dhepler_dscale(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Helper function to compute derivative of standardized variable transformation
    with respect to scale parameter.
    """
    loc = loc_0 + loc_1 * time
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time

    dtx_dscale = -shape * (x - loc) / (scale**2)
    return dtx_dscale


def _dhepler_dshape(x, loc_0, loc_1, scale_0, scale_1, shape_0, shape_1, time):
    """Helper function to compute derivative of standardized variable transformation
    with respect to shape parameter.
    """
    loc = loc_0 + loc_1 * time
    scale = scale_0 + scale_1 * time
    shape = shape_0 + shape_1 * time

    dtx_dshape = (x - loc) / scale
    return dtx_dshape


# test cases
if __name__ == '__main__':
    import pandas as pd
    from scipy.stats import genextreme
    import time

    # simple test case
    np.random.seed(42)
    sample_sizes = [10**i for i in range(2, 3)]
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

    """
    import matplotlib.pyplot as plt 

    xs = np.arange(-100, 100, 1.)
    ys = np.log([_gev_pdf(x, loc=20, scale=1.5, shape=0.25)
                 for x in xs])

    plt.plot(xs, ys)
    plt.show()
    """