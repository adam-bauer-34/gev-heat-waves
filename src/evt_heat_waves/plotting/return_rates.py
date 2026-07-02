import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

from pathlib import Path

from evt_heat_waves.config import MIP_FIT_PATH_DICT
from evt_heat_waves.plotting.utils import make_figure_filename

# Set color and marker sets for CMIP models
# 10 High-Contrast, Colorblind-Friendly Hex Codes (Paul Tol / Okabe-Ito)
colors = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#EE7733",
    "#009988",
    "#332288",
    "#BBBBBB",
]

# Marker Set
markers = ["o", "s", "D"]  # Circle, Square, Diamond
linestyles = ["dashed", "dashdot", "dotted"]


# ---------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------
def plot_return_level_histogram(
    return_levels,
    return_period,
    event_idx=0,
    return_periods=None,
    event_names=None,
    bins=20,
    plot_type="hist",
    figsize=(8, 5),
    save_fig=False,
    fname=None,
    title=None,
):
    """Plot a histogram or KDE of model return levels at selected return period(s).

    Parameters
    ----------
    return_levels : array-like
        Array with shape (N_events, N_models, N_return_periods) in Kelvin.
    return_period : float or list/array of floats
        Return period(s) to visualize.
    event_idx : int
        Event index to use from the first axis of return_levels.
    return_periods : array-like
        Array of return periods corresponding to the last axis of return_levels.
    event_names : list[str] or None
        Optional event names for titles and labels.
    bins : int
        Number of histogram bins (only used if plot_type='hist').
    plot_type : str, default='hist'
        Type of plot: 'hist' for histogram or 'kde' for kernel density estimation.
    figsize : tuple
        Figure size.
    save_fig : bool
        Whether to save the figure.
    fname : str or None
        Output filename if save_fig is True.
    title : str or None
        Optional figure title.
    """

    if return_periods is None:
        raise ValueError("return_periods must be provided")

    if event_names is None:
        event_names = [f"event_{i}" for i in range(return_levels.shape[0])]

    if event_idx < 0 or event_idx >= return_levels.shape[0]:
        raise IndexError("event_idx is out of bounds for return_levels array")

    if plot_type not in ["hist", "kde"]:
        raise ValueError(f"plot_type must be 'hist' or 'kde', got {plot_type}")

    # Handle both scalar and array-like return_period
    return_period_list = np.atleast_1d(return_period)

    # Validate all return periods exist
    for rp in return_period_list:
        if rp not in return_periods:
            raise ValueError(f"Return period {rp} not found in return_periods.")

    # Get indices for all requested return periods
    rp_indices = [
        int(np.where(return_periods == rp)[0][0]) for rp in return_period_list
    ]

    # Color palette for multiple return periods
    color_palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "hist":
        # For histograms with multiple periods, use transparency to overlay
        for i, (rp, rp_idx) in enumerate(zip(return_period_list, rp_indices)):
            values = np.asarray(return_levels[event_idx, :, rp_idx])
            values = (
                values[np.isfinite(values)] - 273.15
            )  # Convert from Kelvin to Celsius

            if values.size == 0:
                raise ValueError(f"No finite return level values for {rp}-yr period.")

            color = color_palette[i % len(color_palette)]
            ax.hist(
                values,
                bins=bins,
                alpha=0.5,
                edgecolor="black",
                label=f"{rp:g}-yr",
                color=color,
            )

        ylabel = "Count"

    else:  # kde
        # Determine x-axis range across all data
        all_values = []
        for rp_idx in rp_indices:
            values = np.asarray(return_levels[event_idx, :, rp_idx])
            values = (
                values[np.isfinite(values)] - 273.15
            )  # Convert from Kelvin to Celsius
            if values.size > 0:
                all_values.extend(values)

        if not all_values:
            raise ValueError(
                "No finite return level values found for any selected return period."
            )

        x_min, x_max = np.min(all_values), np.max(all_values)
        x_range = np.linspace(x_min, x_max, 200)

        # Plot KDE for each return period
        for i, (rp, rp_idx) in enumerate(zip(return_period_list, rp_indices)):
            values = np.asarray(return_levels[event_idx, :, rp_idx])
            values = (
                values[np.isfinite(values)] - 273.15
            )  # Convert from Kelvin to Celsius

            if values.size == 0:
                continue

            kde = stats.gaussian_kde(values)
            color = color_palette[i % len(color_palette)]
            ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=f"{rp:g}-yr")
            ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

        ylabel = "Density"

    ax.set_xlabel("Return Level (°C)")
    ax.set_ylabel(ylabel)

    event_label = (
        event_names[event_idx] if event_idx < len(event_names) else f"event_{event_idx}"
    )
    plot_label = "Histogram" if plot_type == "hist" else "KDE"
    ax.set_title(title or f"{plot_label} of Return Levels for {event_label}")

    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    if save_fig:
        if fname is None:
            raise ValueError("fname must be provided when save_fig=True")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"✍️ Figure saved to: {fname}")

    return fig, ax


def plot_return_level_histogram_grid(
    return_levels,
    return_period_list,
    event_indices,
    return_periods=None,
    era5_return_levels=None,
    event_names=None,
    grid_shape=(3, 2),
    bins=20,
    plot_type="hist",
    figsize=(16, 12),
    save_fig=False,
    fname=None,
    outdir=Path("."),
    title=None,
    location_params=None,
    scale_params=None,
    era5_location_params=None,
    era5_scale_params=None,
    panel_xlims=None,  # list of (xlo, xhi) tuples, one per panel
):
    """Create a grid of histograms or KDE plots for multiple locations and return periods.

    Parameters
    ----------
    return_levels : array-like
        Array with shape (N_events, N_models, N_return_periods) in Kelvin.
    return_period_list : list/array of floats
        Return periods to visualize on each subplot.
    event_indices : list of ints
        Indices of events (locations) to plot. Length should equal grid_shape[0] * grid_shape[1].
    return_periods : array-like
        Array of return periods corresponding to the last axis of return_levels.
    era5_return_levels : array-like or None
        Array with shape (N_events, N_return_periods) in Kelvin. If provided, a dashed
        vertical line is drawn at the ERA5 return level for each return period.
    event_names : list[str] or None
        Names for each event/location.
    grid_shape : tuple, default=(3, 2)
        Grid shape (n_rows, n_cols).
    bins : int
        Number of histogram bins (only used if plot_type='hist').
    plot_type : str, default='hist'
        Type of plot: 'hist' for histogram or 'kde' for kernel density estimation.
    figsize : tuple
        Figure size.
    save_fig : bool
        Whether to save the figure.
    fname : str or None
        Output filename if save_fig is True.
    title : str or None
        Figure title.
    location_params : array-like or None
        Array with shape (N_events, N_models) of GEV location parameters (μ). If provided,
        a KDE inset is drawn in the upper-right of each panel.
    scale_params : array-like or None
        Array with shape (N_events, N_models) of GEV scale parameters (σ). If provided
        alongside location_params, both KDEs are shown in the inset.
    era5_location_params : array-like or None
        Array with shape (N_events,) of ERA5 location parameters. If provided, plotted
        as a dashed black vertical line in the location KDE inset.
    era5_scale_params : array-like or None
        Array with shape (N_events,) of ERA5 scale parameters. If provided, plotted
        as a dashed black vertical line in the scale KDE inset.
    panel_xlims : list of (float, float) or None
        Manual x-axis limits for each main panel, one tuple per panel in the same
        order as event_indices. E.g. [(20, 55), (25, 55), ...]. When None, all
        panels share the same global x_min → 57 range derived from the data.
    """

    if return_periods is None:
        raise ValueError("return_periods must be provided")

    if event_names is None:
        event_names = [f"event_{i}" for i in range(return_levels.shape[0])]

    if plot_type not in ["hist", "kde"]:
        raise ValueError(f"plot_type must be 'hist' or 'kde', got {plot_type}")

    n_rows, n_cols = grid_shape
    n_subplots = n_rows * n_cols

    if len(event_indices) != n_subplots:
        raise ValueError(
            f"event_indices length ({len(event_indices)}) must match grid_shape ({n_subplots})"
        )

    for event_idx in event_indices:
        if event_idx < 0 or event_idx >= return_levels.shape[0]:
            raise IndexError(
                f"event_idx {event_idx} is out of bounds for return_levels array"
            )

    if panel_xlims is not None and len(panel_xlims) != n_subplots:
        raise ValueError(
            f"panel_xlims length ({len(panel_xlims)}) must match number of panels ({n_subplots})"
        )

    has_insets = location_params is not None or scale_params is not None
    if has_insets:
        if location_params is not None:
            location_params = np.asarray(location_params)
        if scale_params is not None:
            scale_params = np.asarray(scale_params)
        if era5_location_params is not None:
            era5_location_params = np.asarray(era5_location_params)
        if era5_scale_params is not None:
            era5_scale_params = np.asarray(era5_scale_params)

    return_period_list = np.atleast_1d(return_period_list)
    for rp in return_period_list:
        if rp not in return_periods:
            raise ValueError(f"Return period {rp} not found in return_periods.")

    rp_indices = [
        int(np.where(return_periods == rp)[0][0]) for rp in return_period_list
    ]

    all_values_global = []
    for event_idx in event_indices:
        for rp_idx in rp_indices:
            values = np.asarray(return_levels[event_idx, :, rp_idx])
            values = values[np.isfinite(values)] - 273.15
            if values.size > 0:
                all_values_global.extend(values)

    global_x_min = np.min(all_values_global) if all_values_global else 0
    x_min_rounded = int(np.floor(global_x_min / 5) * 5)
    x_ticks_default = list(range(x_min_rounded, 58, 5))

    def format_rp_label(rp):
        rp_int = int(rp) if rp == int(rp) else rp
        return f"{rp_int:,} Year"

    color_palette = colors

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    panel_labels = [chr(65 + i) for i in range(n_subplots)]

    def _draw_param_inset(parent_ax, inset_bounds, values, era5_val, xlabel):
        ax_ins = parent_ax.inset_axes(inset_bounds)
        ax_ins.patch.set_facecolor("white")
        ax_ins.patch.set_alpha(0.7)

        values = np.asarray(values)
        values = values[np.isfinite(values)]

        if values.size > 1:
            kde = stats.gaussian_kde(values)
            x_lo, x_hi = values.min(), values.max()
            pad = 0.10 * (x_hi - x_lo) if x_hi > x_lo else 0.5
            x_grid = np.linspace(x_lo - pad, x_hi + pad, 200)
            y_kde = kde(x_grid)

            ax_ins.plot(x_grid, y_kde, color="grey", linewidth=1.2)
            ax_ins.fill_between(x_grid, y_kde, alpha=0.25, color="grey")

            median_val = np.median(values)
            ax_ins.axvline(median_val, color="black", linestyle="solid", linewidth=1.2)

        if era5_val is not None and np.isfinite(era5_val):
            ax_ins.axvline(era5_val, color="black", linestyle="dashed", linewidth=1.2)

        ax_ins.set_ylim(bottom=0)
        ax_ins.set_xlabel(xlabel, fontsize=14, labelpad=2)
        ax_ins.set_yticks([])
        ax_ins.tick_params(axis="x", labelsize=12, pad=1)
        ax_ins.spines[["top", "right", "left"]].set_visible(False)

        return ax_ins

    for subplot_idx, event_idx in enumerate(event_indices):
        ax = axes[subplot_idx]
        row_idx = subplot_idx // n_cols
        col_idx = subplot_idx % n_cols

        # Per-panel x limits and ticks for the main axes
        if panel_xlims is not None:
            xlo, xhi = panel_xlims[subplot_idx]
            x_ticks = list(range(int(np.floor(xlo / 5) * 5), int(xhi) + 1, 5))
        else:
            xlo, xhi = x_min_rounded, 57
            x_ticks = x_ticks_default

        if plot_type == "hist":
            for i, (rp, rp_idx) in enumerate(zip(return_period_list, rp_indices)):
                values = np.asarray(return_levels[event_idx, :, rp_idx])
                values = values[np.isfinite(values)] - 273.15
                if values.size == 0:
                    continue
                color = color_palette[i % len(color_palette)]
                ax.hist(
                    values,
                    bins=bins,
                    alpha=0.5,
                    edgecolor="black",
                    label=format_rp_label(rp),
                    color=color,
                )

                median_val = np.median(values)
                ax.axvline(median_val, color=color, linestyle="solid", linewidth=1.5)

                if era5_return_levels is not None:
                    era5_val = (
                        float(np.asarray(era5_return_levels)[event_idx, rp_idx])
                        - 273.15
                    )
                    if np.isfinite(era5_val):
                        ax.axvline(
                            era5_val, color=color, linestyle="dashed", linewidth=1.5
                        )

            ylabel = "Count"

        else:  # kde
            x_range = np.linspace(xlo, xhi, 200)
            for i, (rp, rp_idx) in enumerate(zip(return_period_list, rp_indices)):
                values = np.asarray(return_levels[event_idx, :, rp_idx])
                values = values[np.isfinite(values)] - 273.15
                if values.size == 0:
                    continue
                kde = stats.gaussian_kde(values)
                color = color_palette[i % len(color_palette)]
                ax.plot(
                    x_range,
                    kde(x_range),
                    color=color,
                    linewidth=2,
                    label=format_rp_label(rp),
                    linestyle="solid",
                )
                ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

                median_val = np.median(values)
                ax.axvline(median_val, color=color, linestyle="solid", linewidth=1.5)

                if era5_return_levels is not None:
                    era5_val = (
                        float(np.asarray(era5_return_levels)[event_idx, rp_idx])
                        - 273.15
                    )
                    if np.isfinite(era5_val):
                        ax.axvline(
                            era5_val, color=color, linestyle="dashed", linewidth=1.5
                        )

            ylabel = "Density"

        ax.set_xlim(xlo, xhi)
        ax.set_xticks(x_ticks)

        if row_idx == n_rows - 1:
            ax.set_xlabel("Return Level (°C)")
        # else:
        #    ax.tick_params(labelbottom=False)

        if col_idx == 0:
            ax.set_ylabel(ylabel, labelpad=10)
        else:
            ax.set_ylabel("")

        ax.yaxis.set_ticks([])

        event_label = event_names[event_idx]
        ax.set_title(event_label, fontsize=16, fontweight="bold")

        if row_idx == 0 and col_idx == 0:
            rp_handles = [
                mpatches.Patch(
                    color=color_palette[i % len(color_palette)],
                    alpha=0.5,
                    label=format_rp_label(rp),
                )
                for i, rp in enumerate(return_period_list)
            ]
            cmip6_handle = mlines.Line2D(
                [],
                [],
                color="grey",
                linestyle="solid",
                linewidth=1.5,
                label="CMIP6 Ensemble Median",
            )
            era5_handle = mlines.Line2D(
                [],
                [],
                color="grey",
                linestyle="dashed",
                linewidth=1.5,
                label="ERA5 Reanalysis",
            )
            all_handles = rp_handles + [cmip6_handle, era5_handle]

        ax.grid(False)

        ax.text(
            0.99,
            0.97,
            panel_labels[subplot_idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_ylim(bottom=0)

        if has_insets:
            both = location_params is not None and scale_params is not None

            if both:
                loc_bounds = [0.56, 0.62, 0.19, 0.32]
                scale_bounds = [0.77, 0.62, 0.19, 0.32]
            else:
                loc_bounds = [0.62, 0.62, 0.33, 0.32]
                scale_bounds = [0.62, 0.62, 0.33, 0.32]

            if location_params is not None:
                loc_vals = location_params[event_idx]
                if np.nanmedian(loc_vals) > 100:
                    loc_vals = loc_vals - 273.15
                era5_loc = (
                    float(era5_location_params[event_idx])
                    if era5_location_params is not None
                    else None
                )
                if era5_loc is not None and era5_loc > 100:
                    era5_loc -= 273.15
                _draw_param_inset(ax, loc_bounds, loc_vals, era5_loc, "μ (°C)")

            if scale_params is not None:
                scale_vals = np.asarray(scale_params[event_idx])
                era5_sc = (
                    float(era5_scale_params[event_idx])
                    if era5_scale_params is not None
                    else None
                )
                _draw_param_inset(ax, scale_bounds, scale_vals, era5_sc, "σ (°C)")

            if both:
                frame_x = loc_bounds[0]
                frame_y = loc_bounds[1]
                frame_w = scale_bounds[0] + scale_bounds[2] - loc_bounds[0]
                frame_h = max(loc_bounds[3], scale_bounds[3])
            elif location_params is not None:
                frame_x, frame_y, frame_w, frame_h = loc_bounds
            else:
                frame_x, frame_y, frame_w, frame_h = scale_bounds

            frame = mpatches.FancyBboxPatch(
                (frame_x, frame_y),
                frame_w,
                frame_h,
                boxstyle="square,pad=0",
                linewidth=0.8,
                edgecolor="black",
                facecolor="none",
                transform=ax.transAxes,
                zorder=10,
            )
            ax.add_patch(frame)

    fig.legend(
        handles=all_handles,
        loc="upper center",
        ncol=len(all_handles),
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    plt.tight_layout()

    if save_fig:
        if fname is None:
            raise ValueError("fname must be provided when save_fig=True")
        fname = make_figure_filename(fname, outdir)
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"✍️ Figure saved to: {fname}")

    return fig, axes


def plot_return_levels_obs(
    event_info,
    CMIPConfig,
    return_periods,
    obs_max,
    era5_ret_levels,
    return_levels_prim,
    return_levels_most,
    model_with_most,
    anom_type,
    cmip_variable,
    save_figs,
    fname,
    outdir,
):
    fig, ax = plt.subplots(3, 3, figsize=(24, 20), sharex=True, sharey=True)
    titles = {
        k: "".join([k, ", ", str(event_info[k]["year"])]) for k in event_info.keys()
    }

    models = list(CMIPConfig.iter_active_models(cmip_variable))

    for locx, (a, loc) in enumerate(zip(ax.flatten(), event_info.keys())):
        a.plot([0, 1], [0, 1])
        a.set_title(titles[loc])
        a.set_ylim((0, 60))
        a.set_xlim((2, max(return_periods)))

        a.axhline(
            obs_max[loc],
            linestyle="solid",
            linewidth=2.5,
            color="violet",
            label="Hottest Day on Record",
            zorder=100,
        )

        # plot era5
        a.semilogx(
            return_periods,
            era5_ret_levels[locx] - 274.15,
            label="ERA5",
            linestyle="solid",
            color="black",
            linewidth=4,
            zorder=100,
        )

        for idx, m in enumerate(models):
            color = colors[idx % 10]
            linestyle = linestyles[idx // 10]

            a.semilogx(
                return_periods,
                return_levels_prim[locx, idx] - 274.15,
                label=m.name if m.name != "MIROC6" else None,
                linestyle=linestyle,
                linewidth=1.75,
                color=color,
                zorder=99,
            )

        for idx in range(np.shape(return_levels_most)[1]):
            a.semilogx(
                return_periods,
                return_levels_most[locx, idx] - 274.15,
                label=f"{model_with_most} Ensemble Members" if idx == 0 else None,
                linestyle="solid",
                linewidth=1,
                color="grey",
                zorder=1,
            )

    ax[0, 0].set_ylabel(r"Return Level ($^\circ$ C)")
    ax[1, 0].set_ylabel(r"Return Level ($^\circ$ C)")
    ax[2, 0].set_ylabel(r"Return Level ($^\circ$ C)")

    ax[2, 0].set_xlabel("Return Period (Years)")
    ax[2, 1].set_xlabel("Return Period (Years)")
    ax[2, 2].set_xlabel("Return Period (Years)")

    ax[1, 2].legend(bbox_to_anchor=(1.1, 1.5), ncols=1, fontsize=14, frameon=True)

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    for a, label in zip(ax.flatten(), labels):
        a.text(
            0.025,
            0.97,
            label,
            transform=a.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )

    if save_figs:
        fname = make_figure_filename(fname, outdir)
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"✍️ Figure saved to: {fname}")


def make_event_location_checkplot(event_info, save_figs, fname, outdir):
    """Make checkplot of event locations."""
    # extract lat/lon and adjust longitudes >180
    lats = []
    lons = []
    names = []
    for name, info in event_info.items():
        lat = info["lat"]
        lon = info["lon"]
        if lon > 180:
            lon = lon - 360
        lats.append(lat)
        lons.append(lon)
        names.append(name)

    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.scatter(lons, lats, transform=ccrs.PlateCarree(), color="red", s=50, zorder=5)
    for name, lon, lat in zip(names, lons, lats):
        ax.text(lon + 1, lat + 1, name, transform=ccrs.PlateCarree(), fontsize=10)
    plt.title("Event locations from event_info")

    if save_figs:
        fname = make_figure_filename(fname, outdir)
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved event location checkplot to {fname}")


# ---------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------
def recenter_model_to_era5(
    loc_dict, ds_era5, cmip_model_name, member_id, mip, time_interval=slice(1979, 2000)
):
    """Recenter CMIP model to ERA5 temperature values."""

    # compute ERA5 climatological mean for this location
    era5_clim_mean = (
        ds_era5.sel(lat=loc_dict["lat"], lon=loc_dict["lon"], method="nearest")
        .sel(year=time_interval)
        .mean(dim="year")
        .t2m.values
    )

    # compute CMIP climatological mean for this location
    # set data path
    data_path = MIP_FIT_PATH_DICT[mip]["data"] / "tas_annual_mean" / "raw"

    # Make all landonly file names
    fnames = [
        f for f in data_path.glob(f"*{cmip_model_name}_*.nc")
    ]  # screen out allmems results

    # make sure we found a unique file, and then extract the name
    if len(fnames) == 1:
        cmip_filename = fnames[0]
        # print(f"        Model filename used for recentering: {cmip_filename}")

    else:
        raise ValueError(
            f"Multiple files found for unique CMIP model {cmip_model_name}. Check which is valid."
        )

    # import dataset, take mean
    ds_cmip = xr.open_dataset(cmip_filename)
    cmip_clim_mean = (
        ds_cmip.sel(lat=loc_dict["lat"], lon=loc_dict["lon"], method="nearest")
        .sel(
            year=time_interval,
            member_id=member_id,
        )
        .mean(dim="year")
        .tas.values
    )

    offset = era5_clim_mean - cmip_clim_mean

    # close dataset to save memory
    ds_cmip.close()

    return offset


def get_return_level(p, mu, sigma, xi):
    """Computes the return level for an event with probability P
    whose occurance is described by a GEV.

    Parameters
    ----------
    p: float
        probability

    mu: float
        mean-like GEV parameter

    sigma: float
        variance-like GEV parameter

    xi: float
        tail controlling GEV parameter

    Returns
    -------
    ret_level: float
        return level of event with probability p
    """

    inside = (-np.log(p)) ** (-xi) - 1
    ret_level = mu + (sigma / xi) * inside
    return ret_level


def array_max(arr):
    """
    Find the maximum value and its index in an array.

    Parameters:
        arr (array-like): Input array.

    Returns:
        tuple: (max_value, max_index)
    """
    arr = np.asarray(arr)
    max_index = np.argmax(arr)
    max_value = arr[max_index]
    return max_value, int(max_index)


def summarize_return_levels(
    return_levels,
    return_period_indices,
    return_periods,
    event_info,
    event_names=None,
    model_type="primary",
):
    """Print a summary of return level statistics for each location and return period.

    Parameters
    ----------
    return_levels : ndarray
        Array with shape (N_events, N_models_or_members, N_return_periods) in Kelvin.
    return_period_indices : list/array of ints
        Indices into return_periods to summarize (e.g., [2, 5, 9] for 10, 100, 1000-year).
    return_periods : ndarray
        Array of all return periods.
    event_info : dict
        Dictionary containing event metadata (location names, years, etc).
    event_names : list[str] or None
        Optional list of event names to display. If None, uses event_info keys.
    model_type : str, default='primary'
        Label for the type of model data ('primary', 'most members', etc).
    """
    if event_names is None:
        event_names = list(event_info.keys())

    rp_indices = np.atleast_1d(return_period_indices)
    rps = return_periods[rp_indices]

    print("\n" + "=" * 120)
    print(f"Return Level Summary ({model_type} ensemble)")
    print("=" * 120)

    for event_idx, event_name in enumerate(event_names):
        print(f"\n{event_name}")
        print("-" * 120)
        print(
            f"{'Return Period':>15} {'5th %ile':>15} {'Median':>15} {'Mean':>15} {'95th %ile':>15} {'5-95 Range':>15}"
        )
        print("-" * 120)

        for rp, rp_idx in zip(rps, rp_indices):
            # Extract model/member return levels for this location and return period
            values = return_levels[event_idx, :, int(rp_idx)]

            # Filter out NaN values
            valid_values = values[np.isfinite(values)] - 273.15  # Convert to Celsius

            if len(valid_values) > 0:
                p5 = np.percentile(valid_values, 5)
                median = np.percentile(valid_values, 50)
                mean = np.mean(valid_values)
                p95 = np.percentile(valid_values, 95)
                range_5_95 = p95 - p5

                # Format return period as integer if it's a whole number
                rp_str = f"{int(rp)}-year" if rp == int(rp) else f"{rp}-year"

                print(
                    f"{rp_str:>15} {p5:>15.2f} {median:>15.2f} {mean:>15.2f} {p95:>15.2f} {range_5_95:>15.2f}"
                )
            else:
                rp_str = f"{int(rp)}-year" if rp == int(rp) else f"{rp}-year"
                print(
                    f"{rp_str:>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}"
                )

    print("=" * 120)
