"""Helper plotting and computing functions for the scale parameter
sensitivity vs location parameter sensitivity analysis"""

import warnings
import numpy as np
import xarray as xr
from evt_heat_waves.config import (
    ERA5_PATH,
    MLE_FIT_ATTRS,
    FIGS_PATH,
)

trend_vars = ["loc_t", "scale_t", "shape_t"]


def get_critical_return_period(medmu, medsig, xi=0.0):
    """Get return period where misfits in the scale parameter (sigma) dominate those of the location parameter (mu)

    Parameters
    ----------
    medmu: float
        misfit in location parameter

    medsig: float
        misfit in scale parameter

    Returns
    -------
    p: float
        the return period
    """

    # compute probabilities for xi = 0 and xi != 0 cases
    if xi == 0:
        pw = np.exp(-np.exp(-medmu / medsig))
    else:
        if 1 + xi * (medmu / medsig) <= 0:
            return 1e10
        else:
            pw = np.exp(-((1 + (medmu * xi) / medsig) ** (-1 / xi)))

    # compute return period
    p = np.log10((1 - pw) ** (-1))
    if p == np.inf:
        return 1e10
    else:
        return p


def get_critical_return_period_dataset(
    ds_cmip, ds_era5, data_type, YEAR, xi=0.0, op="mean"
):
    """Get critical return period dataset

    ds_cmip: xarray.Dataset
        CMIP dataset of location and scale parameters

    ds_era5: xarray.Dataset
        ERA5 dataset

    data_type: str
        type of data from era5 config

    YEAR: float
        the year to evaluate trends

    xi: float
        shape parameter value

    op: str
        operation to apply (default: 'mean')

    Returns
    -------
    ds: xarray.Dataset
        dataset of critical return periods
    """

    mu_era5 = (
        ds_era5[f"loc_{data_type}"] + ds_era5[f"loc_t_{data_type}"] * YEAR
    ).values
    sig_era5 = (ds_era5[f"scale_{data_type}"]).values

    dmu = abs(ds_cmip.mus.mean(dim="model") - mu_era5)
    dsig = abs(ds_cmip.scales.mean(dim="model") - sig_era5)

    crit_ps = np.zeros((len(ds_era5.lat), len(ds_era5.lon)))

    for i in range(crit_ps.shape[0]):
        for j in range(crit_ps.shape[1]):
            crit_ps[i, j] = get_critical_return_period(dmu[i, j], dsig[i, j], xi=xi)

    # make dataset
    ds = xr.Dataset(
        data_vars={"crit_ps": (["lat", "lon"], crit_ps)},
        coords={"lat": ds_era5.lat, "lon": ds_era5.lon},
    )

    return ds


def get_cmip_locs_scales(cmip_data_config, CMIPConfig, ds_era5, YEAR):
    """Get CMIP dataset of location and scale parameters

    cmip_data_config: dict
        data config with information for data loading

    CMIPConfig: CMIP6EnsembleConfig
        configuration for CMIP6 ensemble

    ds_era5: xarray.Dataset
        era5 data (needed to set lat, lon for new arrays)

    YEAR: float
        the year that we should evaluate trends to compute mu / sigma

    Returns
    -------
    ds: xarray.Dataset
        dataset of location and scale parameters
    """

    fit = cmip_data_config["fit"]
    cmip_variable = cmip_data_config["var"]
    data_type = cmip_data_config["data_type"]
    N_active_models = cmip_data_config["N_active_models"]
    modelname_filepath_matcher = cmip_data_config["model_file_matcher"]

    mus_prim = np.zeros((N_active_models, len(ds_era5.lat), len(ds_era5.lon)))
    scales_prim = np.zeros((N_active_models, len(ds_era5.lat), len(ds_era5.lon)))

    for idx, m in enumerate(list(CMIPConfig.iter_active_models(cmip_variable))):
        print(f"⚒️ Working on {m.name}...")

        # open dataset
        tmp_ds = xr.open_dataset(modelname_filepath_matcher[m.name])

        # carry out calculations for both nonstationary and stationary cases
        tmp_loc = (
            tmp_ds[f"loc_{data_type}"].values
            + tmp_ds[f"loc_t_{data_type}"].values * YEAR
            if "loc_t" in MLE_FIT_ATTRS[fit]["param_names"]
            else tmp_ds[f"loc_{data_type}"].values
        )
        tmp_scale = (
            tmp_ds[f"scale_{data_type}"].values
            + tmp_ds[f"scale_t_{data_type}"].values * YEAR
            if "scale_t" in MLE_FIT_ATTRS[fit]["param_names"]
            else tmp_ds[f"scale_{data_type}"].values
        )
        try:
            tmp_shape = (
                tmp_ds[f"shape_{data_type}"].values
                + tmp_ds[f"shape_t_{data_type}"].values * YEAR
                if "shape_t" in MLE_FIT_ATTRS[fit]["param_names"]
                else tmp_ds[f"shape_{data_type}"].values
            )

        except KeyError:
            # set shape value based on the constraint
            for cons in MLE_FIT_ATTRS[fit]["constraints"]:
                if cons["params"][0] == "shape":
                    tmp_shape = cons["value"]

        # store the loc / scale parameters
        mus_prim[idx] = tmp_loc
        scales_prim[idx] = tmp_scale

        tmp_ds.close()

    # make array of model names
    models = []
    for m in CMIPConfig.iter_active_models(cmip_variable):
        models.append(m.name)

    # make dataset
    ds = xr.Dataset(
        data_vars={
            "mus": (["model", "lat", "lon"], mus_prim),
            "scales": (["model", "lat", "lon"], scales_prim),
        },
        coords={"model": models, "lat": ds_era5.lat, "lon": ds_era5.lon},
    )

    return ds


def get_era5_dataset(era5_data_config):
    """Import ERA5 dataset based on configuration."""
    ds = xr.open_dataset(
        ERA5_PATH
        / "gev"
        / f"era5_{era5_data_config['var']}_{era5_data_config['GRID']}_landonly_gev_{era5_data_config['fit']}_TMIN{era5_data_config['TMIN']}_{era5_data_config['anom']}.nc"
    )
    return ds


def percent_above_below_threshold(
    da, threshold, direction="above", weighted=False, pop_da=None
):
    """
    Compute the percent of grid cells in a DataArray that are above or below
    a given threshold. NaN cells are excluded from both the numerator and
    denominator.

    Parameters
    ----------
    da : xr.DataArray
        Data with 'lat' and 'lon' dims (or coords broadcastable to them).
    threshold : float
        Threshold value to compare against.
    direction : str, optional
        'above' (strictly greater than threshold) or 'below' (strictly less
        than threshold). Default is 'above'.
    weighted : bool, optional
        If True, weight by cos(latitude) to account for grid cell area
        shrinking toward the poles. Only affects the area-based percent.
        Default is False.
    pop_da : xr.DataArray, optional
        Gridded population COUNT (not density) per cell, on the exact same
        lat/lon grid as `da` (matching shape/coords). If provided, also
        computes the percent of world population living in cells that
        satisfy the threshold condition. Regrid population data to match
        `da`'s grid before passing it in (e.g. via `.interp_like(da)` for
        a quick nearest/linear regrid, or a proper conservative regridder
        like xesmf if cell-area accuracy matters for population totals).

    Returns
    -------
    float
        Percent (0-100) of valid grid cells satisfying the condition, if
        `pop_da` is not provided (unchanged from previous behavior).
    dict
        {'percent_area': float, 'percent_population': float} if `pop_da`
        is provided.
    """
    if direction == "above":
        mask = da > threshold
    elif direction == "below":
        mask = da < threshold
    else:
        raise ValueError("direction must be 'above' or 'below'")

    valid = da.notnull()
    n_valid = int(valid.sum())

    if n_valid == 0:
        warnings.warn("All grid cells are NaN; returning NaN.")
        pct_area = float("nan")
    else:
        if weighted:
            weights = np.cos(np.deg2rad(da.lat))
            weights = weights.broadcast_like(da).where(valid)
            numerator = weights.where(mask).sum(skipna=True)
            denominator = weights.sum(skipna=True)
        else:
            numerator = mask.where(valid).sum(skipna=True)
            denominator = n_valid
        pct_area = float((numerator / denominator) * 100)

    if pop_da is None:
        return pct_area

    # --- Population-weighted percent ---
    if pop_da.shape != da.shape:
        raise ValueError(
            "pop_da must be on the same lat/lon grid as da (matching shape). "
            "Regrid it first, e.g. pop_da.interp_like(da)."
        )

    # Exclude cells where da itself is NaN (no threshold value to compare),
    # and cells where population data is NaN (e.g. ocean cells in some
    # population products).
    pop_valid = pop_da.where(valid)
    pop_total = pop_valid.sum(skipna=True)

    if float(pop_total) == 0:
        warnings.warn(
            "Total population in valid cells is zero; returning NaN for percent_population."
        )
        pct_pop = float("nan")
    else:
        pop_matching = pop_valid.where(mask).sum(skipna=True)
        pct_pop = float((pop_matching / pop_total) * 100)

    return {"percent_area": pct_area, "percent_population": pct_pop}
