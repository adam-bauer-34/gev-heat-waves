import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
 
from pathlib import Path

from evt_heat_waves.plotting.utils import make_figure_filename
from evt_heat_waves.config import MLE_FIT_ATTRS
 
 
# set labels
colorbar_labels = {'loc': r'$\mu_O$ ($^\circ$C)', 'loc_t': r'$\mu_1$ ($^\circ$C/decade)', 
                   'scale': r'$\sigma_O$ ($^\circ$C)', 'scale_t': r'$\sigma_1$ ($^\circ$C/decade)',
                   'shape': r'$\xi_O$ ($-$)', 'shape_t': r'$\xi_1$ (decade$^{-1}$)'}
panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
 
always_negative = ['shape']  # shape parameter is always negative
always_positive = ['loc', 'scale']  # location and scale parameters are always positive
trend_params = ['loc_t', 'scale_t', 'shape_t']  # trend parameters with associated standard errors
 
# mapping for number of parameters to grid shape and figure size
param_num_to_grid = {
    2: {'grid': (1, 2), 'figsize': (12, 6)},
    3: {'grid': (1, 3), 'figsize': (18, 6)},
    4: {'grid': (2, 2), 'figsize': (12, 12)},
    5: {'grid': (2, 3), 'figsize': (18, 12)}
}
 
 
def plot_gev_parameters(ds, fit, anom_type,
                        suptitle=None, save_figs=True,
                        fname='gev_parameters', output_dir=Path("figs/analysis")):
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
        *param_num_to_grid[MLE_FIT_ATTRS[fit]['N_params']]['grid'],
        figsize=param_num_to_grid[MLE_FIT_ATTRS[fit]['N_params']]['figsize'],
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    if anom_type == 'raw':
        anom_var_name = anom_type
    else:
        anom_var_name = f'anom_{anom_type}'
 
    das = [
        ds[f'{param}_{anom_var_name}'] / len(ds.year.values) * 10.
        if param in trend_params
        else ds[f'{param}_{anom_var_name}']
        for param in MLE_FIT_ATTRS[fit]['param_names']
    ]
 
    for idx, (ax, da, param) in enumerate(zip(axes.flatten(), das, MLE_FIT_ATTRS[fit]['param_names'])):
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="white")
 
        # Determine colormap and vmin/vmax based on parameter type
        if param in always_positive:
            # Use sequential colormap (one half of PuOr) for always-positive parameters
            cmap = "Oranges"  # Or "YlOrBr" for consistency with orange side of PuOr
            vmin = np.nanpercentile(da.values, 5)
            vmax = np.nanpercentile(da.values, 95)
        elif param in always_negative:
            # Use sequential colormap (other half of PuOr) for always-negative parameters
            cmap = "Purples_r"  # Reversed to have darker colors for more negative values
            vmin = np.nanpercentile(da.values, 5)
            vmax = np.nanpercentile(da.values, 95)
        else:
            # Use diverging colormap centered on zero for parameters that can be +/-
            cmap = "PuOr_r"
            vmax = np.nanpercentile(np.abs(da.values), 95)
            vmin = -vmax
 
        # Plot the parameter
        im = da.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax
        )
 
        # Add stippling for trend parameters where sign is consistent within a 2 SE CI.
        # The SE is scaled by the same factor as the parameter (/ N_years * 10 for per-decade).
        if param in trend_params:
            n_years = len(ds.year.values)
            se = ds[f'se_{param}_{anom_var_name}'] / n_years * 10.
 
            ci_lower = da - 2 * se
            ci_upper = da + 2 * se
 
            # Sign is consistent if both CI bounds are strictly the same sign
            sign_consistent = (
                ((ci_lower > 0) & (ci_upper > 0)) |
                ((ci_lower < 0) & (ci_upper < 0))
            )
 
            # Build a lon/lat mesh matching the DataArray grid
            lons, lats = np.meshgrid(da.lon.values, da.lat.values)
            stipple_mask = sign_consistent.values  # boolean 2-D array
 
            ax.scatter(
                lons[stipple_mask],
                lats[stipple_mask],
                s=0.65,
                c='black',
                marker='.',
                transform=ccrs.PlateCarree(),
                zorder=5,
                linewidths=0,
                alpha=1.0,
                rasterized=True
            )
 
        # Add an aligned colorbar on the right of each plot
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label(
            colorbar_labels[param]
        )
 
    # add labels
    for idx, ax in enumerate(axes.flatten()):
        ax.text(0.02, 1.05, f'{panel_labels[idx]}', 
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='bottom', ha='right')
        ax.set_title('')
 
    if suptitle is not None:
        fig.suptitle(suptitle)
 
    fig.subplots_adjust(left=0.05, right=0.85, top=0.55, bottom=0.02)
 
    if save_figs:
        fname = make_figure_filename(name=fname, outdir=output_dir)
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f'Figure saved to: {fname}')
    plt.show()


def area_weighted_average(ds: xr.Dataset, var: str, plo: float = 0, phi: float = 100) -> float:
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
    lats_rad = np.deg2rad(ds[lat_coord])          # convert degrees → radians
    weights = np.cos(lats_rad)                    # shape: (n_lat,)
 
    # Broadcast weights to the full shape of t2m so NaN masking is easy
    # xarray handles the broadcast automatically via `.weighted()`
    weights_da = xr.ones_like(ds[var]) * weights / sum(weights) # shape: (n_lat, n_lon)
 
    # ------------------------------------------------------------------ #
    # 3a. Mask weights where t2m is NaN                                    #
    # ------------------------------------------------------------------ #
    valid_mask = ~np.isnan(ds[var])
    masked_weights = weights_da.where(valid_mask)   # NaN where t2m is NaN

    # ------------------------------------------------------------------ #
    # 3b. compute percentiles to mask values 
    # ------------------------------------------------------------------ #
    var_plo = np.nanpercentile(ds[var], plo)
    var_phi = np.nanpercentile(ds[var], phi)

    # ------------------------------------------------------------------ #
    # 4. Compute weighted mean manually (works with any xarray version)   #
    # ------------------------------------------------------------------ #
    weighted_sum = (
        ds[var].where(
            (ds[var] <= var_phi) & (ds[var] >= var_plo)
            ) * masked_weights
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

    if anom_type == 'raw':
        anom_var_name = anom_type
    else:
        anom_var_name = f'anom_{anom_type}'

    for param in MLE_FIT_ATTRS[fit]['param_names']:
        print(f"The area weighted average for {param} is {area_weighted_average(ds, f'{param}_{anom_var_name}', 1, 99)}")
