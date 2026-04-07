"""Functions to compute the Hessian matrix of the
negative log likelihood function.

Adam Michael Bauer
UChicago
Last edited: 4/6/2026
"""

import numpy as np

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

    # compute the Hessian matrix
    hess = get_hessian(theta_opt, temps, fit_type=fit_type)

    # invert the Hessian
    # NOTE: we take the negative because we fit to the negative loglik and we want the hessian of the loglik
    inv_hess = np.linalg.inv(-hess)

    # standard errors are the square root of the diagonal of the Hessian
    se = np.sqrt(np.diag(inv_hess))

    return se

def get_hessian(theta_opt, temps, fit_type='stat'):
    """Compute the Hessian matrix of the negative log likelihood function.

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
    hess: (N_params, N_params) array
        The Hessian of the negative log likelihood function
    """

    if fit_type == 'stat':
        hess = _get_hessian_stat(theta_opt, temps)

    elif fit_type == 'loc_trend':
        hess = _get_hessian_loc_trend(theta_opt, temps)

    elif fit_type == 'nonstat':
        hess = _get_hessian_nonstat(theta_opt, temps)

    else:
        raise ValueError(f"Invalid fit type: {fit_type}")
    
    return hess

def _get_hessian_stat(theta_opt, temps):
    """Compute the Hessian matrix for the stationary GEV fit.

    Parameters
    ----------
    theta_opt: (3,) array-like
        The optimal MLE parameters (loc, scale, shape)

    temps: (N_years,) array-like
        temperature data used for MLE fitting

    Returns
    -------
    hess: (3, 3) array
        The Hessian of the negative log likelihood function for the stationary GEV fit
    """

    hess = np.zeros((len(theta_opt), len(theta_opt)))
    loc, scale, shape = theta_opt  # unpack parameters

    # compute the Hessian using analytical formulas for the second derivatives
    # of the negative log likelihood function for the stationary GEV distribution
    hess[0, 0] = np.sum(_get_dloc2(temps, loc, scale, shape))

    hess[0, 1] = np.sum(_get_dloc_dscale(temps, loc, scale, shape))
    hess[1, 0] = hess[0, 1]  # Hessian is symmetric

    hess[0, 2] = np.sum(_get_dloc_dshape(temps, loc, scale, shape))
    hess[2, 0] = hess[0, 2]  # Hessian is symmetric

    hess[1, 1] = np.sum(_get_dscale2(temps, loc, scale, shape))

    hess[1, 2] = np.sum(_get_dscale_dshape(temps, loc, scale, shape))
    hess[2, 1] = hess[1, 2]  # Hessian is symmetric

    hess[2, 2] = np.sum(_get_dshape2(temps, loc, scale, shape))

    return hess

def _get_hessian_loc_trend(theta_opt, temps):
    """Get the Hessian matrix when there is a trend in the location parameter only.

    Parameters
    ----------
    theta_opt: (4,) array-like
        The optimal MLE parameters (loc, loc_t, scale, shape)

    temps: (N_years,) array-like
        temperature data used for MLE fitting

    Returns
    -------
    hess: (4, 4) array
        The Hessian of the negative log likelihood function for the loc trend-only GEV fit
    """

    # setup
    hess = np.zeros((len(theta_opt), len(theta_opt)))
    loc0, loc_t, scale, shape = theta_opt  # unpack parameters
    years = np.arange(0, len(temps), 1) / len(temps)  # make normalized time variable 
    loc = loc0 + loc_t * years  # compute location parameter at each time point

    # comput the Hessian using analytical formulas for the second derivatives
    ## sub block with loc0 and loc_t
    ## diagonals 
    hess[0, 0] = np.sum(_get_dloc2(temps, loc, scale, shape))
    hess[1, 1] = np.sum(years**2 * _get_dloc2(temps, loc, scale, shape))

    # off diagonals
    hess[0, 1] = np.sum(years * _get_dloc2(temps, loc, scale, shape))
    hess[1, 0] = hess[0, 1]  # Hessian is symmetric

    # scale and loc derivs
    hess[0, 2] = np.sum(_get_dloc_dscale(temps, loc, scale, shape))
    hess[2, 0] = hess[0, 2]  # Hessian is symmetric

    # shape and loc derivatives
    hess[0, 3] = np.sum(_get_dloc_dshape(temps, loc, scale, shape))
    hess[3, 0] = hess[0, 3]  # Hessian is symmetric

    # scale and shape diagonals
    hess[2, 2] = np.sum(_get_dscale2(temps, loc, scale, shape))
    hess[3, 3] = np.sum(_get_dshape2(temps, loc, scale, shape))

    # scale and loc trend derivatives
    hess[1, 2] = np.sum(years * _get_dloc_dscale(temps, loc, scale, shape))
    hess[2, 1] = hess[1, 2]  # Hessian is symmetric

    # shape and loc trend derivatives
    hess[1, 3] = np.sum(years * _get_dloc_dshape(temps, loc, scale, shape))
    hess[3, 1] = hess[1, 3]  # Hessian is symmetric

    return hess


def _get_hessian_nonstat(theta_opt, temps, testing=False):
    """Get the Hessian matrix when there is a trend in the location parameter only.

    Parameters
    ----------
    theta_opt: (6,) array-like
        The optimal MLE parameters (loc, loc_t, scale, scale_t, shape, shape_t)

    temps: (N_years,) array-like
        temperature data used for MLE fitting

    testing: bool, optional
        whether we use a single data point for numerical testing

    Returns
    -------
    hess: (4, 4) array
        The Hessian of the negative log likelihood function for the loc trend-only GEV fit
    """

    # setup
    hess = np.zeros((len(theta_opt), len(theta_opt)))
    loc0, loc_t, scale0, scale_t, shape0, shape_t = theta_opt  # unpack parameters
    if testing:
        years = np.array([1.])
    else:
        years = np.arange(0, len(temps), 1) / len(temps)  # make normalized time variable 
    loc = loc0 + loc_t * years  # compute location parameter at each time point
    scale = scale0 + scale_t * years  # compute scale parameter at each time point
    shape = shape0 + shape_t * years  # compute shape parameter at each time point

    # comput the Hessian using analytical formulas for the second derivatives
    # the strategy: do this in six sub block, 2x2 matrices
    
    # BLOCK 1: loc and loc
    # diagonals 
    hess[0, 0] = np.sum(_get_dloc2(temps, loc, scale, shape))
    hess[1, 1] = np.sum(years**2 * _get_dloc2(temps, loc, scale, shape))

    # off diagonals
    hess[0, 1] = np.sum(years * _get_dloc2(temps, loc, scale, shape))
    hess[1, 0] = hess[0, 1]  # symmetry of sub blocks

    # BLOCK 2: loc and scale
    # diagonals
    hess[0, 2] = np.sum(_get_dloc_dscale(temps, loc, scale, shape))
    hess[1, 3] = np.sum(years**2 * _get_dloc_dscale(temps, loc, scale, shape))

    # off diagonals
    hess[0, 3] = np.sum(years * _get_dloc_dscale(temps, loc, scale, shape))
    hess[1, 2] = hess[0, 3]  # symmetry of sub blocks

    # exploit symmetry of Hessian
    hess[2, 0] = hess[0, 2]
    hess[3, 1] = hess[1, 3]
    hess[3, 0] = hess[0, 3]
    hess[2, 1] = hess[1, 2]

    # BLOCK 3: loc and shape
    # diagonals
    hess[0, 4] = np.sum(_get_dloc_dshape(temps, loc, scale, shape))
    hess[1, 5] = np.sum(years**2 * _get_dloc_dshape(temps, loc, scale, shape))

    # off diagonals
    hess[0, 5] = np.sum(years * _get_dloc_dshape(temps, loc, scale, shape))
    hess[1, 4] = hess[0, 5]  # symmetry of sub blocks

    # exploit symmetry of Hessian
    hess[4, 0] = hess[0, 4]
    hess[5, 1] = hess[1, 5]
    hess[5, 0] = hess[0, 5]
    hess[4, 1] = hess[1, 4]

    # BLOCK 4: scale and scale
    # diagonals
    hess[2, 2] = np.sum(_get_dscale2(temps, loc, scale, shape))
    hess[3, 3] = np.sum(years**2 * _get_dscale2(temps, loc, scale, shape))

    # off diagonals
    hess[2, 3] = np.sum(years * _get_dscale2(temps, loc, scale, shape))
    hess[3, 2] = hess[2, 3]  # symmetry of sub blocks

    # BLOCK 5: scale and shape
    # diagonals
    hess[2, 4] = np.sum(_get_dscale_dshape(temps, loc, scale, shape))
    hess[3, 5] = np.sum(years**2 * _get_dscale_dshape(temps, loc, scale, shape))

    # off diagonals
    hess[2, 5] = np.sum(years * _get_dscale_dshape(temps, loc, scale, shape))
    hess[3, 4] = hess[2, 5]  # symmetry of sub blocks

    # exploit symmetry of Hessian
    hess[4, 2] = hess[2, 4]
    hess[5, 3] = hess[3, 5]
    hess[5, 2] = hess[2, 5]
    hess[4, 3] = hess[3, 4]

    # BLOCK 6: shape and shape
    # diagonals
    hess[4, 4] = np.sum(_get_dshape2(temps, loc, scale, shape))
    hess[5, 5] = np.sum(years**2 * _get_dshape2(temps, loc, scale, shape))

    # off diagonals
    hess[4, 5] = np.sum(years * _get_dshape2(temps, loc, scale, shape))
    hess[5, 4] = hess[4, 5]  # symmetry of sub blocks

    return hess

def _get_dloc2(temps: np.ndarray, loc: float,
              scale: float, shape: float) -> np.ndarray:
    """Compute the second derivative of the neg log likelihood function
    with respect to the location parameter.
    
    Parameters
    ----------
    temps: (N_years,) array-like
        temperature value to evaluate at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    Returns
    -------
    dloc2: float
    """

    dnom = (temps * shape - loc * shape + scale)**2
    
    num = 1 + shape
    num *= (1 + shape * (temps - loc) / scale)**(-1/shape)
    num *= (
        -1 + shape * (
            1 + shape * (temps - loc) / scale
        )**(1/shape)
    )

    return num / dnom

def _get_dloc_dscale(temps: np.ndarray, loc: float,
                    scale: float, shape: float) -> np.ndarray:
    """Compute the mixed partial derivative of the neg log likelihood function
    with respect to the location parameter and scale parameter.
    
    Parameters
    ----------
    temps: float
        temperature value to evaluate at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    Returns
    -------
    dloc_dscale: float
    """

    dnom = (temps * shape - loc * shape + scale)**2
    dnom *= scale

    num = (1 + shape * (temps - loc) / scale)**(-1/shape)
    num *= (
        -temps + loc + scale - scale * (
            1 + shape * (temps - loc) / scale
        )**(1/shape)
        - shape * scale * (
            1 + shape * (temps - loc) / scale
        )**(1/shape) 
    )

    return num / dnom

def _get_dloc_dshape(temps: np.ndarray, loc: float,
                    scale: float, shape: float) -> np.ndarray:
    """Compute the mixed partial derivative of the neg log likelihood function
    with respect to the location parameter and shape parameter.
    
    Parameters
    ----------
    temps: float
        temperature value to evaluate at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    Returns
    -------
    dloc_dshape: float
    """

    denom = (temps * shape - loc * shape + scale)**2

    num = shape * (temps - loc) - (temps - loc) * (1 + shape) + scale
    num += (
        (temps - loc) * (1 + shape) * (
        1 + shape * (temps - loc) / scale
    )**(-1/shape)
    ) / shape
    num -= (scale * np.log(
        1 + shape * (temps - loc) / scale
    ) * (
        1 + shape * (temps - loc) / scale
    )**((-1 + shape) / shape)) / shape**2

    return num / denom

def _get_dscale2(temps: np.ndarray, loc: float,
              scale: float, shape: float) -> np.ndarray:
    """Compute the second derivative of the neg log likelihood function
    with respect to the scale parameter.
    
    Parameters
    ----------
    temps: (N_years,) array-like
        temperature value to evaluate at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    Returns
    -------
    dloc2: float
    """

    dnom = scale**2 * (temps * shape - loc * shape + scale)**2

    num = (1 + shape * (temps - loc) / scale)**(-1/shape)
    num *= (
        (temps - loc) * (
            2 * scale + (temps - loc) * (shape - 1)
        )
        + (
            (temps * shape - loc * shape + scale) / scale
        )**(1 / shape) * (
            scale**2 - 2 * scale * (temps - loc) - shape * (temps - loc)**2
        )
    )

    return num / dnom

def _get_dscale_dshape(temps: np.ndarray, loc: float,
                    scale: float, shape: float) -> np.ndarray:
    """Compute the mixed partial derivative of the neg log likelihood function
    with respect to the scale parameter and shape parameter.
    
    Parameters
    ----------
    temps: float
        temperature value to evaluate at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    Returns
    -------
    dscale_dshape: float
    """

    denom = scale * (temps * shape - loc * shape + scale)**2

    num = (temps - loc) * shape - (temps - loc) * (1 + shape) + scale
    num += (
        (temps - loc) * (1 + shape) * (
            1 + shape * (temps - loc) / scale
        )**(-1/shape)
    ) / shape
    num -= (
        scale * np.log(
            1 + shape * (temps - loc) / scale
        ) * (
            1 + shape * (temps - loc) / scale
        )**((-1 + shape) / shape)
    ) / shape**2
    num *= temps - loc

    return num / denom

def _get_dshape2(temps: np.ndarray, loc: float,
              scale: float, shape: float) -> np.ndarray:
    """Compute the second derivative of the neg log likelihood function
    with respect to the shape parameter.
    
    Parameters
    ----------
    temps: (N_years,) array-like
        temperature value to evaluate at

    loc: float
        location parameter

    scale: float
        scale parameter

    shape: float
        shape parameter

    Returns
    -------
    dshape2: float
    """

    # compute first piece
    piece1 = -temps + loc - 3 * temps * shape + 3 * loc * shape - 2 * scale
    piece1 += (
        (2 * scale + (temps - loc) * shape * (3 + shape))
        * (1 + shape * (temps - loc) / scale)**(1/shape)
    )
    piece1 *= shape**2 * (temps - loc)

    # second piece
    piece2 = 2 * shape * (temps * shape - loc * shape + scale)
    piece2 *= np.log(
        1 + shape * (temps - loc) / scale
    ) 
    piece2 *= (
        -temps + loc - temps * shape + loc * shape - scale + scale * (
            1 + shape * (temps - loc) / scale
        )**(1 + 1/shape)
    )
    
    # third piece
    piece3 = (temps * shape - loc * shape + scale)**2
    piece3 *= np.log(1 + shape * (temps - loc) / scale)**2

    # combine 
    dshape2 = piece1 - piece2 - piece3

    # multiply by common prefactor
    dshape2 /= shape**4 * (temps * shape - loc * shape + scale)**2
    dshape2 *= (
        1 + shape * (temps - loc) / scale
    )**(-1/shape)

    return dshape2


if __name__ == "__main__":
    import shutil
    width = shutil.get_terminal_size(fallback=(80, 20)).columns

    import pandas as pd
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

    non_stat_hess = _get_hessian_nonstat([1, 1, 0.5, 0.5, -0.05, -0.05],
                                         np.array([5.]), testing=True)
    print(f"[NONSTAT] Hessian with single data point: {non_stat_hess}")