"""Scripting functions for computing Kuiper statistics.

Adam Michael Bauer
UChicago
11.8.2025
"""

import numpy as np
import xarray as xr

from scipy.stats import genextreme
from astropy.stats import kuiper

from evt_heat_waves.config import MLE_FIT_ATTRS, MLE_FULL_PARAM_NAMES
from evt_heat_waves.mle.mle import _mle_fit

# suffix map to get shape, loc, and scale names
suffix_map = {
    't2m':             'raw',
    'tas':             'raw',
    't2m_anom_annmean':'anom_annmean',
    't2m_anom_trend':  'anom_trend',
}

# constrained shape values smaller than this in magnitude are Gumbel fits, whose
# shape parameter is treated as identically zero
GUMBEL_SHAPE_TOL = 1e-3

# base seed for the misspecification draws
KUIPER_SEED = 20260819


def compute_kuiper_stats(ds, var_name='t2m', fit_dim='year', fit_type='stat',
                         ds_free=None, free_fit_type=None, n_reps=1,
                         seed=KUIPER_SEED):
    """Compute Kuiper statistics at each gridcell.

    Parameters
    ----------
    ds: xarray.Dataset
        the input dataset containing the data to fit

    var_name: str
        the variable name in the dataset to fit the GEV distribution to

    fit_dim: str
        the dimension over which to fit the GEV distribution (e.g., 'year')

    fit_type: str
        the MLE fit type the input parameters came from; determines whether the
        shape parameter is read from the dataset or held fixed (identically zero
        for 'stat_gumbel')

    ds_free: xarray.Dataset or None
        a fit of the same data with the shape parameter left free, used as the
        data-generating truth for the misspecification test. Skipped if None, or
        if fit_type doesn't pin the shape parameter

    free_fit_type: str or None
        the MLE fit type ds_free came from

    n_reps: int
        replicates per gridcell for the misspecification test

    seed: int
        base seed for the misspecification draws; combined with a per-gridcell
        index so results are reproducible

    Returns
    -------
    ds: xarray.Dataset
        the input dataset with added GEV parameters as new variables
    """
    # subselect data
    da = ds[var_name]
    locs = ds[f'loc_{suffix_map[var_name]}']
    scales = ds[f'scale_{suffix_map[var_name]}']

    # for fits that don't estimate the shape parameter (e.g., stat_gumbel), the
    # shape isn't in the fitted dataset: hold it at the fit's fixed value, which
    # is zero identically for Gumbel rather than the near-zero constraint value
    fixed_shape = _fixed_shape(fit_type)
    if fixed_shape is None:
        shapes = ds[f'shape_{suffix_map[var_name]}']
    else:
        shapes = xr.full_like(locs, fixed_shape)

    # define ufunc dict for obs analysis
    ufunc_kwargs_obs = dict(
        input_core_dims=[[fit_dim], [], [], []],
        output_core_dims=[['kuiper']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={
            'output_sizes': {'kuiper': 1}
            }
    )

    # compute Kuiper statistics for observed and synthetic data
    da_ko = xr.apply_ufunc(
        _kuiper,
        da,
        shapes,
        locs,
        scales,
        **ufunc_kwargs_obs
    )

    # now handle synthetic obs + kuiper calculation.
    # first do synthetic draws via fitted distributions
    ufunc_kwargs_syn = dict(
        input_core_dims=[[], [], []],
        output_core_dims=[['kuiper']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        kwargs={'N_SAMPLES': ds[var_name].sizes[fit_dim], 'fit_type': fit_type},
        dask_gufunc_kwargs={
            'output_sizes': {'kuiper': 1}
            }
    )

    da_ks = xr.apply_ufunc(
        _kuiper_syn,
        shapes,
        locs,
        scales,
        **ufunc_kwargs_syn
    )

    # assign kuiper statistics to dataset
    ds = _assign_kuiper(ds, var_name, da_ko, da_ks)

    # misspecification test: only defined when this fit pins the shape parameter
    # and we have a free-shape fit to act as the data-generating truth
    if fixed_shape is not None and ds_free is not None:
        ds = _add_misspecification_kuiper(
            ds,
            ds_free,
            var_name=var_name,
            fit_dim=fit_dim,
            fit_type=fit_type,
            free_fit_type=free_fit_type,
            fixed_shape=fixed_shape,
            locs=locs,
            scales=scales,
            n_reps=n_reps,
            seed=seed,
        )

    return ds


def _add_misspecification_kuiper(ds, ds_free, var_name, fit_dim, fit_type,
                                 free_fit_type, fixed_shape, locs, scales,
                                 n_reps, seed):
    """Add the Kuiper misspecification statistic to the dataset.

    Answers "if the free-shape fit is the truth, how much Kuiper distance does
    pinning the shape parameter cost?" Samples are drawn from the free-shape fit
    and then refit with the fixed-shape model, so the statistic carries both the
    fitted-parameter bias and the cost of the wrong shape.

    Emits two variables: 'mis_k_{sfx}' from those samples, and 'syn_k_paired_
    {sfx}', its null companion drawn from the fixed-shape fit using the *same*
    uniforms. The pairing is what makes the difference of the two readable at
    small n_reps; comparing 'mis_k' against the unpaired 'syn_k' throws that
    variance reduction away.
    """

    if free_fit_type is None:
        raise ValueError(
            "free_fit_type is required to compute the misspecification statistic."
        )

    if _fixed_shape(free_fit_type) is not None:
        raise ValueError(
            f"free_fit_type {free_fit_type!r} holds the shape parameter fixed, so it "
            f"can't act as the free-shape truth for the misspecification test."
        )

    sfx = suffix_map[var_name]

    # free-shape fit parameters act as the data-generating distribution
    shapes_free = ds_free[f'shape_{sfx}']
    locs_free = ds_free[f'loc_{sfx}']
    scales_free = ds_free[f'scale_{sfx}']

    if locs_free.sizes != locs.sizes:
        raise ValueError(
            f"Free-shape fit grid {dict(locs_free.sizes)} doesn't match the "
            f"fixed-shape fit grid {dict(locs.sizes)}."
        )

    # deterministic per-gridcell seeds: reproducible across runs, and independent
    # between cells so the draw noise doesn't imprint structure on the maps
    cell_ids = xr.DataArray(
        np.arange(locs.size, dtype=np.int64).reshape(locs.shape),
        dims=locs.dims,
        coords=locs.coords,
    )

    ufunc_kwargs_mis = dict(
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[['kuiper_mis']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        kwargs={
            'N_SAMPLES': ds[var_name].sizes[fit_dim],
            'fit_type': fit_type,
            'fixed_shape': fixed_shape,
            'n_reps': n_reps,
            'seed': seed,
        },
        dask_gufunc_kwargs={
            'output_sizes': {'kuiper_mis': 2}
            }
    )

    da_mis = xr.apply_ufunc(
        _kuiper_mis_pair,
        shapes_free,
        locs_free,
        scales_free,
        locs,
        scales,
        cell_ids,
        **ufunc_kwargs_mis
    )

    ds = ds.assign({f'mis_k_{sfx}': (('lat', 'lon'), da_mis.data[:, :, 0])})
    ds = ds.assign({f'syn_k_paired_{sfx}': (('lat', 'lon'), da_mis.data[:, :, 1])})

    return ds


def _kuiper(sample, shape, loc, scale, SAMPLE_THRES=10):
    if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
        return np.array([np.nan])
        
    sample = sample[np.isfinite(sample)]

    if len(sample) < SAMPLE_THRES:
        return np.array([-1])

    else:
        k, _ = kuiper(sample,
                           lambda x: genextreme.cdf(x,
                                                c=-shape, loc=loc, scale=scale))
        return np.array([k])


def _kuiper_syn(shape, loc, scale, N_SAMPLES, fit_type='stat'):
    if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
        return np.array([np.nan])

    else:
        tmp_sample = genextreme.rvs(-shape, loc=loc,
                                    scale=scale, size=N_SAMPLES)

        # refit the synthetic draw with the *same* fit type as the observations,
        # so a fixed-shape fit (e.g., stat_gumbel) stays fixed-shape here too
        fit_params = _mle_fit(tmp_sample, fit_type=fit_type)

        # catch if MLE fails
        if np.any(np.isnan(fit_params)):
            return np.array([np.nan])

        else:
            # map the fitted vector onto (loc, scale, shape); fits that hold the
            # shape fixed don't return it, so fall back to that fixed value
            # (zero identically for Gumbel)
            fitted = dict(zip(MLE_FIT_ATTRS[fit_type]['param_names'], fit_params))
            loc_hat = fitted['loc']
            scale_hat = fitted['scale']
            shape_hat = fitted.get('shape', _fixed_shape(fit_type))

            tmp_k = _kuiper(tmp_sample, shape_hat, loc_hat, scale_hat)
            return tmp_k


def _kuiper_mis_pair(shape_free, loc_free, scale_free, loc_fix, scale_fix,
                     cell_id, N_SAMPLES, fit_type, fixed_shape, n_reps, seed):
    """Kuiper statistic for samples drawn from the free-shape fit, plus its
    null companion drawn from the fixed-shape fit, both refit fixed-shape.

    Returns (mis_k, syn_k_paired), each averaged over n_reps.
    """

    params = (shape_free, loc_free, scale_free, loc_fix, scale_fix)
    if np.any(np.isnan(params)):
        return np.array([np.nan, np.nan])

    rng = np.random.default_rng([seed, int(cell_id)])

    mis_ks, syn_ks = [], []
    for _ in range(n_reps):
        # common random numbers: one set of uniforms pushed through both
        # distributions, so the two samples differ only by their parameters and
        # the paired difference isn't swamped by draw-to-draw noise. Clipped to
        # keep the tails off +/-inf.
        u = np.clip(rng.random(N_SAMPLES), 1e-12, 1 - 1e-12)

        # draw from the free-shape fit (the assumed truth) and from the
        # fixed-shape fit (the calibrated null)
        mis_ks.append(
            _refit_kuiper(
                genextreme.ppf(u, -shape_free, loc=loc_free, scale=scale_free),
                fit_type, fixed_shape
            )
        )
        syn_ks.append(
            _refit_kuiper(
                genextreme.ppf(u, -fixed_shape, loc=loc_fix, scale=scale_fix),
                fit_type, fixed_shape
            )
        )

    mis_ks, syn_ks = np.array(mis_ks), np.array(syn_ks)

    # if every replicate's MLE failed there's nothing to average
    if not np.isfinite(mis_ks).any() or not np.isfinite(syn_ks).any():
        return np.array([np.nan, np.nan])

    return np.array([np.nanmean(mis_ks), np.nanmean(syn_ks)])


def _refit_kuiper(sample, fit_type, fixed_shape):
    """Refit a sample with the fixed-shape model and return its Kuiper stat."""

    fit_params = _mle_fit(sample, fit_type=fit_type)

    if np.any(np.isnan(fit_params)):
        return np.nan

    fitted = dict(zip(MLE_FIT_ATTRS[fit_type]['param_names'], fit_params))

    return float(
        _kuiper(sample, fitted.get('shape', fixed_shape),
                fitted['loc'], fitted['scale'])[0]
    )


def _fixed_shape(fit_type):
    """Get the shape parameter held fixed by this fit type, if any.

    Fits like 'stat_gumbel' and 'stat_minxi' pin the shape parameter via an
    SLSQP equality constraint, so it's absent from both the fitted parameter
    vector and the fitted dataset. The Gumbel constraint value is a small
    nonzero number only to keep the MLE off the exact-zero branch of the GEV
    PDF, so it's snapped back to zero identically here.

    Parameters
    ----------
    fit_type: str
        the MLE fit type

    Returns
    -------
    float or None
        the fixed shape value, or None if the shape is estimated by the MLE
    """

    if fit_type not in MLE_FIT_ATTRS:
        raise ValueError(
            f"Unknown fit_type {fit_type!r}. "
            f"Expected one of: {list(MLE_FIT_ATTRS)}"
        )

    attrs = MLE_FIT_ATTRS[fit_type]

    # shape is a free parameter of the fit, so it lives in the fitted dataset
    if 'shape' in attrs['param_names']:
        return None

    # otherwise pull the constrained value out of the fit config
    for cons in attrs.get('constraints', []):
        if cons['params'] == ['shape']:
            value = float(cons['value'])
            return 0.0 if abs(value) < GUMBEL_SHAPE_TOL else value

    raise ValueError(
        f"Fit type {fit_type!r} neither fits nor constrains the shape parameter."
    )


def _assign_kuiper(ds, var_name, da_ko, da_ks):
    """Assign Kuiper statistics for observed and synthetic data to the dataset.

    Resolves the variable name suffix using the same suffix_map as
    _assign_params, then assigns obs_k and syn_k arrays in one place
    rather than repeating ds.assign() calls for every var_name.
    """

    if var_name not in suffix_map:
        raise ValueError(
            f"Unknown var_name {var_name!r}. "
            f"Expected one of: {list(suffix_map)}"
        )

    sfx = suffix_map[var_name]

    ds = ds.assign({f'obs_k_{sfx}': (('lat', 'lon'), da_ko.data[:, :, 0])})
    ds = ds.assign({f'syn_k_{sfx}': (('lat', 'lon'), da_ks.data[:, :, 0])})

    return ds