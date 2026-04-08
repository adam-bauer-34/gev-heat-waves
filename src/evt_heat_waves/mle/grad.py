"""Gradient of nonstationary likelihood function.

Adam Bauer
UChicago
Apr 8, 2026
"""

import numpy as np


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