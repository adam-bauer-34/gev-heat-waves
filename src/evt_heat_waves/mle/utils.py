"""Utility functions for setting up MLE.

Adam Bauer
UChicago
Apr 8 2026
"""

import numpy as np

from evt_heat_waves.config import MLE_FIT_ATTRS, MLE_FULL_PARAM_NAMES

def get_bounds(fit_type):
    """Return a length-6 tuple of (lo, hi) bound pairs for scipy.optimize.minimize.

    The MLE always solves over MLE_FULL_PARAM_NAMES. Params present in the
    fit_type's bounds config get their specified bounds; all others (inactive
    params pinned by constraints) fall back to (None, None).

    Parameters
    ----------
    fit_type: str
        type of fit. must be implemented in config.
    """
    bounds_map = MLE_FIT_ATTRS[fit_type].get('bounds', {})

    # validate no unknown param names crept into the config
    unknown = [p for p in bounds_map if p not in MLE_FULL_PARAM_NAMES]
    if unknown:
        raise ValueError(
            f"fit_type {fit_type!r} has bounds for unknown params: {unknown}"
        )

    return tuple(
        tuple(bounds_map[p]) if p in bounds_map else (None, None)
        for p in MLE_FULL_PARAM_NAMES
    )

def get_constraints(fit_type):
    """Return a list of scipy constraint dicts for the given fit_type.

    Parameters
    ----------
    fit_type: str
        type of fit. must be implemented in config.
    """
    attrs        = MLE_FIT_ATTRS[fit_type]
    param_names  = MLE_FULL_PARAM_NAMES
    descriptors  = attrs.get('constraints', [])
    return [_make_constraint(d, param_names) for d in descriptors]


def _make_constraint(descriptor, param_names):
    """Resolve a constraint descriptor dict into a scipy-compatible constraint dict.

    Parameters
    ----------
    descriptor  : dict with keys 'type', 'fn', 'params' (list of param name strings)
    param_names : ordered list of param names for this fit_type (from MLE_FIT_ATTRS)

    Supported fn values
    -------------------
    fix_param           : forces x[i] == 0  (equality)
    scale_positive_trend: enforces x[i0] + x[i1] >= 0  (inequality)
    """
    fn_name = descriptor['fn']
    ctype   = descriptor['type']

    # resolve param names -> indices using the ordered param_names list
    try:
        idx = [param_names.index(p) for p in descriptor['params']]
    except ValueError as e:
        raise ValueError(
            f"Constraint param not found in param_names {param_names}: {e}"
        )

    if fn_name == 'fix_param':
        return {'type': ctype, 'fun': lambda x, i=idx[0]: x[i]}

    if fn_name == 'scale_positive_trend':
        return {'type': ctype, 'fun': lambda x, i=idx: x[i[0]] + x[i[1]]}

    raise ValueError(f"Unknown constraint fn: {fn_name!r}")

def get_initial_guess(data):
    """Get an initial guess for the MLE optimization using the moments of the data.

    Parameters
    ----------
    data: (N_years,) array-like
        temperature data
    
    Returns
    -------
    init_guess: (6,) array-like
        initial guess for the MLE optimization
    """
    EULER_CONST = 0.5772156649

    # tune initial guess based on the moments of the samples
    samp_mean = np.mean(data)
    samp_std = np.std(data)

    scale_guess = samp_std * np.sqrt(6) / np.pi  # initial guess for scale
    loc_guess = samp_mean + scale_guess * EULER_CONST  # initial guess for loc
    shape_guess = -0.1  # initial guess for shape

    guess_map = {
        'loc': loc_guess,
        'loc_t': 0.0,
        'scale': scale_guess,
        'scale_t': 0.0,
        'shape': shape_guess,
        'shape_t': 0.0
    }

    # validate no params have crept into MLE_FULL_PARAM_NAMES without a guess
    missing = [p for p in MLE_FULL_PARAM_NAMES if p not in guess_map]
    if missing:
        raise ValueError(f"No initial guess defined for params: {missing}")

    return np.array([guess_map[p] for p in MLE_FULL_PARAM_NAMES])