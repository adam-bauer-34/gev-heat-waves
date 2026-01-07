import numpy as np
import xarray as xr

def compute_ecdf(values, extend_lower=True,
                extend_upper=False, ub=None):
    """
    Compute the empirical cumulative distribution function (ECDF)
    of an xarray DataArray.

    Parameters
    ----------
    values : numpy.ndarray
        Input data. Will be flattened before computing ECDF.

    Returns
    -------
    x : numpy.ndarray
        Sorted data values.
    y : numpy.ndarray
        ECDF probabilities corresponding to x.
    """
    # Extract and flatten values
    values = values[~np.isnan(values)]

    # Sort values
    x = np.sort(values)

    # Compute ECDF probabilities
    #y = np.arange(1, len(x) + 1) / len(x)
    n = len(x)
    y = (np.arange(1, n + 1) - 0.5) / n

    if extend_lower and x[0] > 0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([0.0], y))

    if extend_upper and x[1] < ub:
        x = np.concatenate((x, [ub]))
        y = np.concatenate((y, [1.0]))    
    
    return x, y