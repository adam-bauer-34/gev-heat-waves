"""Plot GEV parameter fields and compute summary metrics.

This module provides functions for visualizing generalized extreme value (GEV)
fit parameters on map projections, computing area-weighted averages for
parameter fields, and reporting trend significance for time-varying GEV
parameters. It supports parameter-specific color scaling, trend stippling, and
saving figure outputs.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

from pathlib import Path

from evt_heat_waves.plotting.utils import make_figure_filename
from evt_heat_waves.config import MLE_FIT_ATTRS

# set labels
colorbar_labels = {
    "loc": r"$\mu$ ($^\circ$C)",
    "loc_t": r"$\mu_{trend}$ ($^\circ$C/decade)",
    "scale": r"$\sigma$ ($^\circ$C)",
    "scale_t": r"$\sigma_{trend}$ ($^\circ$C/decade)",
    "shape": r"$\xi$ ($-$)",
    "shape_t": r"$\xi_{trend}$ (decade$^{-1}$)",
}
panel_labels = ["A", "B", "C", "D", "E", "F"]

always_negatives = {"max": ["shape"], "min": ["loc", "shape"]}

always_positives = {
    "max": ["loc", "scale"],
    "min": ["scale"],
}  # location and scale parameters are always positive
trend_params = [
    "loc_t",
    "scale_t",
    "shape_t",
]  # trend parameters with associated standard errors

# mapping for number of parameters to grid shape and figure size
param_num_to_grid = {
    2: {
        "grid": (1, 2),
        "figsize": (12, 6),
        "subplot_adjust": {"left": 0.05, "right": 0.85, "top": 0.85, "bottom": 0.15},
    },
    3: {
        "grid": (3, 1),
        "figsize": (10, 20),
        "subplot_adjust": {"top": 0.6, "bottom": 0.1},
    },
    4: {
        "grid": (2, 2),
        "figsize": (12, 12),
        "subplot_adjust": {"left": 0.05, "right": 0.85, "top": 0.85, "bottom": 0.15},
    },
    5: {
        "grid": (2, 3),
        "figsize": (18, 12),
        "subplot_adjust": {"left": 0.05, "right": 0.85, "top": 0.85, "bottom": 0.15},
    },
    6: {
        "grid": (3, 2),
        "figsize": (20, 30),
        "subplot_adjust": {"top": 0.6, "bottom": 0.1},
    },
}


def plot_gev_parameters(
    ds_stat,
    ds_trend,
    N_params,
    anom_type,
    param_panel_order,
    ex_type="max",
    moments=False,
    suptitle=None,
    save_figs=True,
    fname="gev_parameters",
    output_dir=Path("figs/analysis"),
):
    """
    Plot four xarray DataArrays (shape, loc, scale, k) on a 2x2 world map grid.

    Parameters
    ----------
    shape, loc, scale, k : xr.DataArray
        2D DataArrays with coordinates (lat, lon)
    titles : list or tuple of str, optional
        Custom titles for the subplots. Defaults to ['Shape', 'Location', 'Scale', 'k'].
    save_fig : bool, optional
        If True, saves the figure to `fname`.
    fname : str, optional
        Output filename if save_fig=True.
    """

    # Set up the plotting grid
    fig, axes = plt.subplots(
        *param_num_to_grid[N_params]["grid"],
        figsize=param_num_to_grid[N_params]["figsize"],
        subplot_kw={"projection": ccrs.EqualEarth()},
    )

    always_negative = always_negatives[ex_type]
    always_positive = always_positives[ex_type]

    if anom_type == "raw":
        anom_var_name = anom_type
    else:
        anom_var_name = f"anom_{anom_type}"

    if ds_trend is None:
        ds_trend = ds_stat.copy()

    if moments:
        das = []
        moment_panels = ["mean", "std", "mean_trend"]
        for idx, var in enumerate(moment_panels):
            if var == "mean":
                das.append(
                    ds_trend[f"loc_{anom_var_name}"]
                    + 0.577 * ds_trend[f"scale_{anom_var_name}"]
                )
            elif var == "std":
                das.append(ds_trend[f"scale_{anom_var_name}"] * np.pi * 6 ** (-0.5))
            else:
                das.append(
                    ds_trend[f"loc_t_{anom_var_name}"]
                    / len(ds_trend.year.values)
                    * 10.0
                )
    else:
        das = [
            (
                ds_trend[f"{param}_{anom_var_name}"] / len(ds_trend.year.values) * 10.0
                if param in trend_params
                else ds_stat[f"{param}_{anom_var_name}"]
            )
            for param in param_panel_order
        ]

    for idx, (ax, da, param) in enumerate(zip(axes.flatten(), das, param_panel_order)):
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="white")

        # Determine colormap and vmin/vmax based on parameter type
        N_BINS = 10
        HALF_BINS = N_BINS // 2  # 5 bins from each half of RdYlBu

        # Full RdYlBu colors (blue -> yellow -> red), extract halves
        full_cmap = plt.get_cmap("RdYlBu_r", N_BINS)
        colors = full_cmap(np.linspace(0, 1, N_BINS))
        blue_colors = colors[:HALF_BINS]  # blue end (negative side)
        red_colors = colors[HALF_BINS:]  # red end (positive side)

        if param in always_positive:
            cmap = mcolors.ListedColormap(red_colors)
            vmin = np.nanpercentile(da.values[da.values > 0], 5)
            vmax = np.nanpercentile(da.values[da.values > 0], 95)
            norm = mcolors.BoundaryNorm(
                np.linspace(vmin, vmax, HALF_BINS + 1), ncolors=HALF_BINS
            )
        elif param in always_negative:
            cmap = mcolors.ListedColormap(blue_colors)
            vmin = np.nanpercentile(da.values[da.values < 0], 5)
            vmax = np.nanpercentile(da.values[da.values < 0], 95)
            norm = mcolors.BoundaryNorm(
                np.linspace(vmin, vmax, HALF_BINS + 1), ncolors=HALF_BINS
            )
        else:
            cmap = plt.get_cmap("RdYlBu_r", N_BINS)
            vmax = np.nanpercentile(np.abs(da.values), 95)
            vmin = -vmax
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # Plot the parameter
        im = da.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
        )

        # Add stippling for trend parameters where sign is consistent within a 2 SE CI.
        # The SE is scaled by the same factor as the parameter (/ N_years * 10 for per-decade).
        if param in trend_params:
            n_years = len(ds_trend.year.values)
            se = ds_trend[f"se_{param}_{anom_var_name}"] / n_years * 10.0

            ci_lower = da - 2 * se
            ci_upper = da + 2 * se

            # Sign is consistent if both CI bounds are strictly the same sign
            sign_consistent = ((ci_lower > 0) & (ci_upper > 0)) | (
                (ci_lower < 0) & (ci_upper < 0)
            )

            # Build a lon/lat mesh matching the DataArray grid
            lons, lats = np.meshgrid(da.lon.values, da.lat.values)
            stipple_mask = sign_consistent.values  # boolean 2-D array

            ax.scatter(
                lons[stipple_mask],
                lats[stipple_mask],
                s=1.75,
                c="black",
                marker=".",
                transform=ccrs.PlateCarree(),
                zorder=5,
                linewidths=0,
                alpha=1.0,
                rasterized=True,
            )

        # Add an aligned colorbar on the right of each plot
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.04, shrink=0.99)
        cbar.set_label(colorbar_labels[param])
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    for ax in axes.flatten():
        ax.set_global()

    # add labels
    for idx, ax in enumerate(axes.flatten()):
        ax.text(
            0.02,
            1.05,
            f"{panel_labels[idx]}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        ax.set_title("")

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.subplots_adjust(**param_num_to_grid[N_params]["subplot_adjust"])

    if save_figs:
        fname = make_figure_filename(name=fname, outdir=output_dir)
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {fname}")
    plt.show()


def area_weighted_average(
    ds: xr.Dataset, var: str, plo: float = 0, phi: float = 100
) -> float:
    """
    Compute the area-weighted average of the 't2m' variable in an xarray Dataset.

    Area weighting accounts for the fact that grid cells at higher latitudes
    cover a smaller surface area than those near the equator. Each cell is
    weighted by the cosine of its latitude, which is proportional to its area
    on a spherical Earth.

    NaN values in 't2m' are excluded from both the weighted sum and the
    sum of weights, so they do not bias the result.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
          - 'lat'  : latitude coordinate (degrees), named 'lat' or 'latitude'
          - 'lon'  : longitude coordinate (degrees), named 'lon' or 'longitude'

    var : string
        variable key

    plo : float (default = 0)
        lower percentile to mask out

    phi : float (default = 100)
        higher percentile to mask out

    Returns
    -------
    float
        The area-weighted mean of 't2m', ignoring NaNs.

    Raises
    ------
    ValueError
        If no latitude coordinate can be identified, or if all values are NaN.
    """

    # ------------------------------------------------------------------ #
    # 1. Identify the latitude coordinate (flexible naming)               #
    # ------------------------------------------------------------------ #
    lat_names = ["lat", "latitude", "LAT", "LATITUDE"]
    lat_coord = next((n for n in lat_names if n in ds.coords), None)
    if lat_coord is None:
        raise ValueError(
            f"Could not find a latitude coordinate. Expected one of {lat_names}. "
            f"Available coords: {list(ds.coords)}"
        )

    # ------------------------------------------------------------------ #
    # 2. Build cosine-of-latitude weights                                 #
    # ------------------------------------------------------------------ #
    lats_rad = np.deg2rad(ds[lat_coord])  # convert degrees → radians
    weights = np.cos(lats_rad)  # shape: (n_lat,)

    # Broadcast weights to the full shape of t2m so NaN masking is easy
    # xarray handles the broadcast automatically via `.weighted()`
    weights_da = xr.ones_like(ds[var]) * weights / sum(weights)  # shape: (n_lat, n_lon)

    # ------------------------------------------------------------------ #
    # 3a. Mask weights where t2m is NaN                                    #
    # ------------------------------------------------------------------ #
    valid_mask = ~np.isnan(ds[var])
    masked_weights = weights_da.where(valid_mask)  # NaN where t2m is NaN

    # ------------------------------------------------------------------ #
    # 3b. compute percentiles to mask values
    # ------------------------------------------------------------------ #
    var_plo = np.nanpercentile(ds[var], plo)
    var_phi = np.nanpercentile(ds[var], phi)

    # ------------------------------------------------------------------ #
    # 4. Compute weighted mean manually (works with any xarray version)   #
    # ------------------------------------------------------------------ #
    weighted_sum = (
        ds[var].where((ds[var] <= var_phi) & (ds[var] >= var_plo)) * masked_weights
    ).sum(skipna=True)
    weight_total = masked_weights.sum(skipna=True)

    if float(weight_total) == 0:
        raise ValueError("All 'var' values are NaN — cannot compute a weighted mean.")

    result = float(weighted_sum / weight_total)
    return result


def print_summary(ds, fit, anom_type):
    """
    Print a summary of GEV parameter statistics for the given dataset and anomaly type.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing GEV parameters and their trends
    anom_type : str
        The anomaly type (e.g., 'absolute', 'relative')
    """

    if anom_type == "raw":
        anom_var_name = anom_type
    else:
        anom_var_name = f"anom_{anom_type}"

    for param in MLE_FIT_ATTRS[fit]["param_names"]:
        print(
            f"The area weighted average for {param} is {area_weighted_average(ds, f'{param}_{anom_var_name}', 1, 99)} +/- {area_weighted_average(ds, f'se_{param}_{anom_var_name}', 1, 99)}"
        )


def print_trend_significance(ds, fit, anom_type, n_se=2):
    """
    Print the percentage of land grid cells with statistically confident
    positive and negative trends for each trend parameter, using a
    confidence interval of +/- n_se standard errors (default: 2 SE ~ 95%;
    pass n_se=1.645 for 90%).

    Reports:
      - % of all land cells with a confident positive / negative trend
      - % of positive-trend cells that are confidently positive
      - % of negative-trend cells that are confidently negative

    Parameters
    ----------
    ds : xr.Dataset
    fit : str
    anom_type : str
    n_se : float
        Number of standard errors for the CI (default 2; use 1.645 for 90%).
    """
    n_years = len(ds.year.values)

    for param in MLE_FIT_ATTRS[fit]["param_names"]:
        if param not in trend_params:
            continue

        da = ds[f"{param}_anom_{anom_type}"] / n_years * 10.0
        se = ds[f"se_{param}_anom_{anom_type}"] / n_years * 10.0

        ci_lower = da - n_se * se
        ci_upper = da + n_se * se

        valid = np.isfinite(da.values)
        pos = (da.values > 0) & valid
        neg = (da.values < 0) & valid
        sig_pos = (ci_lower.values > 0) & valid
        sig_neg = (ci_upper.values < 0) & valid

        n_total = valid.sum()
        n_pos = pos.sum()
        n_neg = neg.sum()
        n_sig_pos = sig_pos.sum()
        n_sig_neg = sig_neg.sum()

        pct_sig_pos_of_total = 100 * n_sig_pos / n_total
        pct_sig_neg_of_total = 100 * n_sig_neg / n_total
        pct_sig_pos_of_pos = 100 * n_sig_pos / n_pos if n_pos > 0 else np.nan
        pct_sig_neg_of_neg = 100 * n_sig_neg / n_neg if n_neg > 0 else np.nan

        print(f"\n{param} (CI: +/- {n_se} SE)")
        print(
            f"  Confident positive trend:  {pct_sig_pos_of_total:5.1f}% of all land  |  "
            f"{pct_sig_pos_of_pos:5.1f}% of positive-trend cells"
        )
        print(
            f"  Confident negative trend:  {pct_sig_neg_of_total:5.1f}% of all land  |  "
            f"{pct_sig_neg_of_neg:5.1f}% of negative-trend cells"
        )


if __name__ == "__main__":
    """Quick runnable test that mirrors the analysis in
    analysis/notebooks/era5_gev_fit_maps.ipynb. This will attempt to
    open the ERA5 GEV fit dataset and produce the parameter maps and
    summaries. Intended for developer smoke-testing only.
    """
    # Local imports to avoid adding heavy dependencies at module import time
    from evt_heat_waves.config import ERA5_PATH, FIGS_PATH
    from evt_heat_waves.plotting.plotting_presets import get_presets

    # apply plotting presets used in the notebook
    presets, _ = get_presets(markers=False)
    plt.rcParams.update(presets)

    # notebook defaults
    save_figs = True
    TMIN = 1979
    GRID = "1deg"
    ex_type = "min"
    nonstat_fit = "nonstat_gumbel_only_loc_trend"
    stat_fit = "stat_gumbel"
    anom_type = "annmean"

    try:
        ds_stat = xr.open_dataset(
            ERA5_PATH
            / "gev"
            / f"era5_t2m_annual_{ex_type}_{GRID}_landonly_gev_{stat_fit}_TMIN{TMIN}_{anom_type}.nc",
            engine="netcdf4",
        )
        ds_nonstat = xr.open_dataset(
            ERA5_PATH
            / "gev"
            / f"era5_t2m_annual_{ex_type}_{GRID}_landonly_gev_{nonstat_fit}_TMIN{TMIN}_{anom_type}.nc",
            engine="netcdf4",
        )
    except Exception as e:
        raise RuntimeError(f"Could not open ERA5 dataset: {e}")

    # run the plotting + summaries (mirrors the notebook)
    plot_gev_parameters(
        ds_stat,
        ds_nonstat,
        N_params=3,
        anom_type=anom_type,
        param_panel_order=["loc", "scale", "loc_t"],
        ex_type=ex_type,
        save_figs=save_figs,
        fname=f"era5-t2m-{ex_type}-{anom_type}-gev-parameters-{GRID}-{TMIN}-hybrid",
        output_dir=FIGS_PATH,
    )
