"""Maximum likelihood fitting function for stationary or nonstationary GEV.

Adam Michael Bauer
UChicago
11.11.2025
"""

import numpy as np
import xarray as xr
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import genextreme

def ds_mle_fit(ds, var_name, fit_dim='year', non_stat=False, parallel=True):
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

    parallel: bool
        whether to use dask parallelization for the fitting

    Returns
    -------
    ds: xarray.Dataset
        the input dataset with added GEV parameters as new variables
    """

    # subselect variable to do the fitting over
    da = ds[var_name]

    # carry out either parallelized or non-parallelized fit
    if parallel:
        gev_params = xr.apply_ufunc(
            _mle_fit,
            da,
            input_core_dims=[[fit_dim]],
            output_core_dims=[['gev_params']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
        )

    else:
        gev_params = xr.apply_ufunc(
            _mle_fit,
            da,
            input_core_dims=[[fit_dim]],
            output_core_dims=[['gev_params']]
        )

    if non_stat:
        # assign shape, loc, and scale parameters to their (lat, lon) coords
        if var_name == 't2m':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_0_raw = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(loc_1_raw = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(scale_0_raw = (('lat', 'lon'), gev_params.data[:, :, 2]))
            ds = ds.assign(scale_1_raw = (('lat', 'lon'), gev_params.data[:, :, 3]))
            ds = ds.assign(shape_0_raw = (('lat', 'lon'), gev_params.data[:, :, 4]))
            ds = ds.assign(shape_1_raw = (('lat', 'lon'), gev_params.data[:, :, 5]))


        elif var_name == 't2m_anom':
            # assign shape, loc, and scale parameters to their (lat, lon) coords
            ds = ds.assign(loc_0_anom = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(loc_1_anom = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(scale_0_anom = (('lat', 'lon'), gev_params.data[:, :, 2]))
            ds = ds.assign(scale_1_anom = (('lat', 'lon'), gev_params.data[:, :, 3]))
            ds = ds.assign(shape_0_anom = (('lat', 'lon'), gev_params.data[:, :, 4]))
            ds = ds.assign(shape_1_anom = (('lat', 'lon'), gev_params.data[:, :, 5]))

    else:
        if var_name == 't2m':
            ds = ds.assign(loc_raw = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(scale_raw = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(shape_raw = (('lat', 'lon'), gev_params.data[:, :, 2]))

        elif var_name == 't2m_anom':
            ds = ds.assign(loc_anom = (('lat', 'lon'), gev_params.data[:, :, 0]))
            ds = ds.assign(scale_anom = (('lat', 'lon'), gev_params.data[:, :, 1]))
            ds = ds.assign(shape_anom = (('lat', 'lon'), gev_params.data[:, :, 2]))

    # return the amended dataset
    return ds

def _mle_fit(data, SAMPLE_THRES=10, non_stat=False):
    """Fit a potentiallly nonstationary GEV distribution to data via MLE.
    """

    # only take finite values
    data = data[np.isfinite(data)]

    # if the number of points is less than the sample threshold,
    # return NaNs for the fitted parameters
    if len(data) < SAMPLE_THRES:
        return np.array([np.nan] * 3)
    
    # try to do the GEV distribution fit and return parameters
    initial_guess = [np.mean(data), 0,  # loc parameters
                     np.std(data), 0.0,  # scale parameters
                     0.1, 0.0  # shape parameters
    ]

    # set up constraints for MLE fit for nonstationary or stationary cases
    # nonstationary just requires the scale parameter to be positive for all time
    # stationary sets the trend in parameters to zero, and keeps the scale parameter positive
    if non_stat:
        cons = ({'type': 'ineq',
                 'fun': lambda x: x[2] + x[3] * len(data)})  # scale_0 + scale_1 * time > 0

    else:
        cons = ({'type': 'ineq',
                 'fun': lambda x: x[2] + x[3] * len(data)},  # scale_0 + scale_1 * time > 0
                {'type': 'eq',
                 'fun': lambda x: x[1]},  # loc_1 = 0 (no trend)
                {'type': 'eq',
                 'fun': lambda x: x[3]},  # scale_1 = 0 (no trend)
                {'type': 'eq',
                 'fun': lambda x: x[5]})  # shape_1 = 0 (no trend)
        
    # do MLE fit
    fit = minimize(_negative_log_likelihood,
                    initial_guess,
                    args=(data, non_stat),
                    method='SLSQP',  # SLSQP to allow for constraints
                    constraints=cons
                    )

    # if the fit is successful, return parameters, else return nans
    if fit.success:
        #print("MLE fit successful.")
        if non_stat:
            return np.array(fit.x)  # return all 6 parameters
        else:
            return np.array([fit.x[0], fit.x[2], fit.x[4]])  # loc_0, scale_0, shape_0
    
    else:
        print("WARNING: MLE fit failed: {}".format(fit.message))
        if non_stat:
            return np.array([np.nan] * 6)  # return nans for failed fit
        else:
            return np.array([np.nan] * 3)  # return nans for failed fit


def _negative_log_likelihood(params, data, non_stat=False):
    if non_stat:
        loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
    else:
        loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
        loc_1 = scale_1 = shape_1 = 0

    time = np.arange(0, len(data), 1) / len(data)  # normalized time variable

    log_likelihood = - np.sum(
        np.log([_gev_pdf_pen(x=x,
                      loc=loc_0 + loc_1 * t,
                      scale=scale_0 + scale_1 * t,
                      shape=shape_0 + shape_1 * t) for x, t in zip(data, time)]
        )
    )

    return log_likelihood

def _gev_pdf(x, loc, scale, shape):
    """Compute the PDF of the GEV distribution at some point x.

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

    Returns
    -------
    pdf: array-like
        PDF evaluated at x
    """

    if shape > 0:
        support_lb = loc - scale / shape
        support_data_mask = x >= support_lb
    
    elif shape == 0:
        support_data_mask = np.ones_like(x, dtype=bool)  # all real numbers are supported
    
    else:
        support_ub = loc - scale / shape
        support_data_mask = x <= support_ub

    x = np.asarray(x)
    x_sup = np.where(support_data_mask, x, np.nan)  # set unsupported values to nan

    s = (x_sup - loc) / scale  # standardized variable

    if shape == 0:
        t_x = np.exp(-s)  # transformation for Gumbel case
    else:
        t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

    # eval PDF
    pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
    return pdf

def _gev_pdf_pen(x, loc, scale, shape, pen=np.exp(-50)):
    """Compute the PDF of the GEV distribution at some point x.

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

    Returns
    -------
    pdf: array-like
        PDF evaluated at x
    """

    if shape > 0:
        support_lb = loc - scale / shape
        if x < support_lb:
            return pen  # large penalty for unsupported values
        else:
            s = (x - loc) / scale  # standardized variable

            if shape == 0:
                t_x = np.exp(-s)  # transformation for Gumbel case
            else:
                t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

            # eval PDF
            pdf = (1 / scale) * t_x**(shape + 1) * np.exp(-t_x)
            return pdf
        
    elif shape < 0:
        support_ub = loc - scale / shape
        if x > support_ub:
            return pen  # large penalty for unsupported values
        else:
            s = (x - loc) / scale  # standardized variable

            if shape == 0:
                t_x = np.exp(-s)  # transformation for Gumbel case
            else:
                t_x = (1 + shape * s)**(-1 / shape)  # transformation (assuming scale !=0)

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

# test cases
if __name__ == '__main__':
    # simple test case
    np.random.seed(42)
    sample_sizes = [10**i for i in range(2, 6)]
    non_stat_l2 = []
    stat_l2 = []
    for ss in sample_sizes:
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
        time = np.arange(0, ss, 1) / ss  # 100 years of real data
        data_nonstat = np.array([genextreme.rvs(
            c=-(params_nonstat[4] + params_nonstat[5] * t),
            loc=(params_nonstat[0] + params_nonstat[1] * t),
            scale=(params_nonstat[2] + params_nonstat[3] * t),
            size=1)[0] for t in time])
    
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

    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'sample_size': sample_sizes,
        'stationary_l2_error': stat_l2,
        'nonstationary_l2_error': non_stat_l2
    })

    # print dataframe and save to data/checks
    print(df)

    df.to_csv('../data/checks/mle_sample_size_l2_stat_nonstat.csv', index=False)
    print("\nDataFrame saved to ../data/checks/mle_sample_size_l2_stat_nonstat.csv")

