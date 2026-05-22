import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from scipy.stats import linregress
import pandas as pd

from evt_heat_waves.config import (
    MLE_FIT_ATTRS,
    CHECKS_PATH,
    FIGS_PATH,
    ERA5_PATH,
    MIP_FIT_PATH_DICT,
)
from evt_heat_waves.mip_fit.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.plotting.utils import make_figure_filename
from evt_heat_waves.utils import extract_model_name

# Set color and marker sets for CMIP models
# 10 High-Contrast, Colorblind-Friendly Hex Codes (Paul Tol / Okabe-Ito)
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', 
          '#AA3377', '#EE7733', '#009988', '#332288', '#BBBBBB']

# Marker Set
markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', 'p'] # Circle, Square, Diamond

title_map = {
    'loc': 'Location Parameter',
    'loc_t': 'Location Parameter Trend',
    'scale': 'Scale Parameter',
    'scale_t': 'Scale Parameter Trend',
    'shape': 'Shape Parameter',
    'shape_t': 'Shape Parameter Trend'
}

xlabel_map = {
    'loc': r'$\mu^{ERA5}_0$ ($^\circ$C)',
    'loc_t': r"$\mu^{ERA5}_1'$ ($^\circ$C/decade)",
    'scale': r'$\sigma^{ERA5}_0$ ($^\circ$C)',
    'scale_t': r"$\sigma^{ERA5}_1'$ ($^\circ$C/decade)",
    'shape': r'$\xi^{ERA5}_0$ ($-$)',
    'shape_t': r"$\xi^{ERA5}_1'$ (decade$^{-1}$)"
}

ylabel_maps = {
    'cmip': {
        'loc': r'$\mu^{CMIP}_0$ ($^\circ$C)',
        'loc_t': r"$\mu^{CMIP}_1'$ ($^\circ$C/decade)",
        'scale': r'$\sigma^{CMIP}_0$ ($^\circ$C)',
        'scale_t': r"$\sigma^{CMIP}_1'$ ($^\circ$C/decade)",
        'shape': r'$\xi^{CMIP}_0$ ($-$)',
        'shape_t': r"$\xi^{CMIP}_1'$ (decade$^{-1}$)"
    },
    'amip': {
        'loc': r'$\mu^{AMIP}_0$ ($^\circ$C)',
        'loc_t': r"$\mu^{AMIP}_1'$ ($^\circ$C/decade)",
        'scale': r'$\sigma^{AMIP}_0$ ($^\circ$C)',
        'scale_t': r"$\sigma^{AMIP}_1'$ ($^\circ$C/decade)",
        'shape': r'$\xi^{AMIP}_0$ ($-$)',
        'shape_t': r"$\xi^{AMIP}_1'$ (decade$^{-1}$)"   
    }
}

corr_title_map = {
    'loc': r'Location Parameter | $\mu$ $(^\circ$C$)$',
    'loc_t': r"Location Parameter Trend | $\mu_{trend}$ $(^\circ$C / dec$)$",
    'scale': r'Scale Parameter | $\sigma$ $(^\circ$C$)$',
    'scale_t': r"Scale Parameter Trend | $\sigma_{trend}$ $(^\circ$C / dec$)$",
    'shape': r'Shape Parameter | $\xi$ $(-)$',
    'shape_t': r"Shape Parameter Trend | $\xi_{trend}$ $($dec$^{-1})$"
}

panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

# mapping for number of parameters to grid shape and figure size
param_num_to_grid = {
    2: {'grid': (1, 2), 'figsize': (12, 6)},
    3: {'grid': (1, 3), 'figsize': (18, 6)},
    4: {'grid': (2, 2), 'figsize': (12, 12)},
    5: {'grid': (2, 3), 'figsize': (18, 12)}
}

xlabels = {
    'med': "Median absolute deviation: MODEL $-$ ERA5",
    'mean': "Mean absolute deviation: MODEL $-$ ERA5"
}

corr_plot_attrs = {
    2: {
        'grid': (1, 2),
        'figsize': (14, 6),
        'have_ylabels': [0],
        'have_xlabels': [0, 1],
        'legend_panel': 1,
        'legend': {
            'loc': 'center',
            'bbox_to_anchor': (0.5, 1.15),
            'ncol': 5,
        },
    },
    3: {
        'grid': (2, 2),
        'figsize': (25, 20),
        'have_ylabels': [0, 2],
        'have_xlabels': [0, 1, 2],
        'legend_panel': (1,0),
        'legend': {
            'loc': 'center',
            'bbox_to_anchor': (1.63, 0.6),
            'ncol': 3,
            'frameon': True
        },
    },
    4: {
        'have_ylabels': [0, 2],
        'have_xlabels': [2, 3],
        'legend_panel': 3,
        'legend': {
            'loc': 'center',
            'bbox_to_anchor': (0.5, 1.15),
            'ncol': 6,
            'fontsize': 12
        },
    }
}

trend_vars = ['loc_t', 'scale_t', 'shape_t']

def compute_primary_member_bias_vs_corr(
    fit,
    anom_type,
    ex_type,
    data_type,
    tmin,
    grid,
    mip,
    plo=5,
    phi=95,
):
    """Run the notebook bias-vs-corr analysis and return computed arrays.

    Parameters
    ----------
    fit : str
        Nonstationary fit name from MLE_FIT_ATTRS.
    anom_type : str
        Anomaly type: 'raw', 'trend', or 'annmean'.
    ex_type : str
        Extreme type: 'max' or 'min'.
    data_type : str
        The data field suffix used in the files, e.g. 'raw' or 'anom_annmean'.
    tmin : int
        Minimum year used in the ERA5 filename.
    grid : str
        Grid resolution string such as '1deg'.
    mip : str
        Dataset group name, e.g. 'cmip' or 'amip'.
    plo : float
        Lower percentile for masking.
    phi : float
        Upper percentile for masking.

    Returns
    -------
    dict
        Dictionary containing the generated metrics and masks:
        - era5_data
        - modelname_filepath_matcher
        - med_abs_dev_prim, mean_abs_dev_prim, slopes_prim,
          intercepts_prim, r2s_prim
        - med_abs_dev_most, mean_abs_dev_most, slopes_most,
          intercepts_most, r2s_most
        - era5_masked_prim, mip_masked_prim, era5_masked_most,
          mip_masked_most
        - model_with_most, allmems_filepaths
    """

    # Determine data suffix when not explicitly provided
    if data_type is None:
        # For raw anomalies the files use 'raw', otherwise use 'anom_<type>' convention
        data_type = 'raw' if anom_type == 'raw' else f'anom_{anom_type}'

    # Variable naming conventions for ERA5 vs CMIP in filepaths
    era5_variable = f't2m_annual_{ex_type}'
    cmip_variable = f'tas_annual_{ex_type}'

    # Construct ERA5 file path and open dataset
    era5_path = (
        ERA5_PATH / 'gev' /
        f'era5_{era5_variable}_{grid}_landonly_gev_{fit}_TMIN{tmin}_{anom_type}.nc'
    )
    ds_era5 = xr.open_dataset(era5_path, engine='netcdf4')
    # convert trend parameters to per-decade units (dataset stored per-year)
    per_decade_conversion_factor = 10 / len(ds_era5.year.values)

    # Extract ERA5 fit parameter arrays into a simple dict of flattened numpy arrays
    # For trend parameters apply the per-decade conversion factor
    era5_data = {
        p: ds_era5[f'{p}_{data_type}'].values.flatten() * per_decade_conversion_factor
        if p in trend_vars
        else ds_era5[f'{p}_{data_type}'].values.flatten()
        for p in MLE_FIT_ATTRS[fit]['param_names']
    }

    # Load CMIP6 ensemble configuration (metadata + QC) for the requested MIP
    CMIPConfig = CMIP6EnsembleConfig.from_yaml(
        MIP_FIT_PATH_DICT[mip]['config']['meta'],
        MIP_FIT_PATH_DICT[mip]['config']['qc']
    )

    # Build list of primary model files (exclude multi-member 'allmems' files)
    data_path = MIP_FIT_PATH_DICT[mip]['data'] / cmip_variable / 'gev'
    fnames = [
        f for f in data_path.glob(f'*_{fit}_*{anom_type}*.nc')
        if 'allmems' not in f.name
    ]
    # Map model short name -> filepath for quick lookup
    modelname_filepath_matcher = {
        extract_model_name(f): f for f in fnames
    }

    # Prepare arrays to collect metrics for the primary (one-file-per-model) analysis
    N_active_models = len(list(CMIPConfig.iter_active_models(cmip_variable)))
    med_abs_dev_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
    mean_abs_dev_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
    slopes_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
    intercepts_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
    r2s_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}

    # Determine which model has the most ensemble members so we can analyze its members
    Nens_for_active_models = np.array([
        len(m.all_members) for m in CMIPConfig.iter_active_models(cmip_variable)
    ])
    max_inds = np.where(Nens_for_active_models == np.max(Nens_for_active_models))[0]
    ind_ = max_inds[0]
    # choose a default for 'amip' where ensembles are treated specially
    model_with_most = ([m.name for m in CMIPConfig.iter_active_models(cmip_variable)][ind_]
                       if mip != 'amip'
                       else 'MIROC6')  # deals with tie for ensemble members
    N_members = int(np.max(Nens_for_active_models))

    # Prepare arrays to collect metrics for the multi-member (largest ensemble) analysis
    med_abs_dev_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
    mean_abs_dev_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
    slopes_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
    intercepts_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
    r2s_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}

    # Find the all-members file for the chosen model and open it
    gev_dir = data_path
    pattern = f'*{model_with_most}*_{fit}_*allmems*{anom_type}.nc'
    allmems_filepaths = sorted(gev_dir.glob(pattern))
    if not allmems_filepaths:
        raise FileNotFoundError(f'No allmems file found for pattern: {pattern}')

    ds_most = xr.open_dataset(allmems_filepaths[0], engine='netcdf4')

    # Loop over each ensemble member in the largest ensemble, computing masks and metrics
    era5_masked_most = []
    mip_masked_most = []
    for idx, mem in enumerate(ds_most.member_id.values):
        tmp_ds = ds_most.sel(member_id=mem)
        for param in MLE_FIT_ATTRS[fit]['param_names']:
            # trend parameters need per-decade scaling
            if param in trend_vars:
                mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten() * per_decade_conversion_factor
            else:
                mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten()

            # Apply mutual percentile mask between ERA5 and model values
            era5_masked, mip_masked = mutual_mask_perc(era5_data[param], mip_vals, plo, phi)
            era5_masked_most.append(era5_masked)
            mip_masked_most.append(mip_masked)

            # Absolute deviation metrics: median and mean of (model - ERA5)
            abs_dev = mip_masked - era5_masked
            med_abs_dev_most[param][idx] = np.nanmedian(abs_dev)
            mean_abs_dev_most[param][idx] = np.nanmean(abs_dev)

            # Linear regression between ERA5 and model for each member
            reg = linregress(era5_masked, mip_masked)
            slopes_most[param][idx] = reg.slope
            intercepts_most[param][idx] = reg.intercept
            r2s_most[param][idx] = reg.rvalue**2

        tmp_ds.close()
    ds_most.close()

    # Repeat the same analysis but using the single-file primary member for each active model
    era5_masked_prim = []
    mip_masked_prim = []
    for idx, m in enumerate(list(CMIPConfig.iter_active_models(cmip_variable))):
        tmp_ds = xr.open_dataset(modelname_filepath_matcher[m.name], engine='netcdf4')
        for param in MLE_FIT_ATTRS[fit]['param_names']:
            if param in trend_vars:
                mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten() * per_decade_conversion_factor
            else:
                mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten()

            era5_masked, mip_masked = mutual_mask_perc(era5_data[param], mip_vals, plo, phi)
            era5_masked_prim.append(era5_masked)
            mip_masked_prim.append(mip_masked)

            abs_dev = mip_masked - era5_masked
            med_abs_dev_prim[param][idx] = np.nanmedian(abs_dev)
            mean_abs_dev_prim[param][idx] = np.nanmean(abs_dev)

            reg = linregress(era5_masked, mip_masked)
            slopes_prim[param][idx] = reg.slope
            intercepts_prim[param][idx] = reg.intercept
            r2s_prim[param][idx] = reg.rvalue**2

        tmp_ds.close()

    # Return all computed arrays and metadata needed by downstream plotting routines
    return {
        'era5_data': era5_data,
        'modelname_filepath_matcher': modelname_filepath_matcher,
        'med_abs_dev_prim': med_abs_dev_prim,
        'mean_abs_dev_prim': mean_abs_dev_prim,
        'slopes_prim': slopes_prim,
        'intercepts_prim': intercepts_prim,
        'r2s_prim': r2s_prim,
        'med_abs_dev_most': med_abs_dev_most,
        'mean_abs_dev_most': mean_abs_dev_most,
        'slopes_most': slopes_most,
        'intercepts_most': intercepts_most,
        'r2s_most': r2s_most,
        'era5_masked_prim': era5_masked_prim,
        'mip_masked_prim': mip_masked_prim,
        'era5_masked_most': era5_masked_most,
        'mip_masked_most': mip_masked_most,
        'model_with_most': model_with_most,
        'allmems_filepaths': allmems_filepaths,
        'cmip_config': CMIPConfig,
        'ds_era5': ds_era5,
    }


def plot_bias_vs_corr(abs_dev_prim, r2s_prim, 
                      abs_dev_most, r2s_most,
                      param_order, N_params,
                      models, model_with_most, med_or_mean,
                      fname: str, save_figs: bool = True,
                      xlim_percentiles: tuple = (1, 99)):
    
    fig, ax = plt.subplots(
        *corr_plot_attrs[N_params]['grid'],
        figsize=corr_plot_attrs[N_params]['figsize']
    )

    for (idx, (a, var, param)) in enumerate(zip(ax.flatten(), r2s_prim.keys(), param_order)):
        a.axvline(0, 0, 1, linestyle='solid', color='k')
        a.axhline(0, -1, 1, linestyle='solid', color='k')
        for mdx, m in enumerate(models):
            marker = markers[mdx // 10]
            color = colors[mdx % 10]
            a.scatter(abs_dev_prim[param][mdx], r2s_prim[param][mdx], s=90, marker=marker, color=color, label=m.name, zorder=100)
            a.set_title(corr_title_map[param])
            if idx in corr_plot_attrs[N_params]['have_ylabels']:
                a.set_ylabel("$r^2$")
            if idx in corr_plot_attrs[N_params]['have_xlabels']:
                a.set_xlabel(xlabels[med_or_mean].replace('Absolute ', '').replace('absolute ', ''))

    for (idx, (a, var, param)) in enumerate(zip(ax.flatten(), r2s_most.keys(), param_order)):
        a.axvline(0, 0, 1, linestyle='solid', color='k')
        a.axhline(0, -1, 1, linestyle='solid', color='k')
        for mdx in range(len(abs_dev_most[var])):
            a.scatter(abs_dev_most[param][mdx], r2s_most[param][mdx],
                      s=60, marker='.', color='grey', zorder=1,
                      label=f'{model_with_most} Ensemble Members' if mdx == 0 else None)
            a.set_title(corr_title_map[param])
            if idx in corr_plot_attrs[N_params]['have_ylabels']:
                a.set_ylabel("r$^2$", fontsize=18)
            if idx in corr_plot_attrs[N_params]['have_xlabels']:
                a.set_xlabel(xlabels[med_or_mean].replace('Absolute ', '').replace('absolute ', ''),
                             fontsize=18)

    # Set axis limits per panel
    for idx, (a, var) in enumerate(zip(ax.flatten(), param_order)):
        all_x = np.concatenate([
            [abs_dev_prim[var][mdx] for mdx in range(len(models))],
            [abs_dev_most[var][mdx] for mdx in range(len(abs_dev_most[var]))]
        ])
        all_r2 = np.concatenate([
            [r2s_prim[var][mdx] for mdx in range(len(models))],
            [r2s_most[var][mdx] for mdx in range(len(r2s_most[var]))]
        ])

        #x_mean, x_std = np.mean(all_x), np.std(all_x)
        #x_lo, x_hi = x_mean - 2.5 * x_std, x_mean + 3 * x_std

        x_lo = all_x[all_x >= np.percentile(all_x, xlim_percentiles[0])].min() * 1.1
        x_hi = all_x[all_x <= np.percentile(all_x, xlim_percentiles[1])].max() * 1.05

        a.set_xlim(x_lo, x_hi)

        n_clipped = np.sum((all_x < x_lo) | (all_x > x_hi))
        if n_clipped > 0:
            clipped_models = [m.name for mdx, m in enumerate(models) 
                              if abs_dev_prim[var][mdx] < x_lo or abs_dev_prim[var][mdx] > x_hi]
            # also check ensemble members (no names, just indices)
            n_clipped_most = np.sum(
                (np.array([abs_dev_most[var][mdx] for mdx in range(len(abs_dev_most[var]))]) < x_lo) |
                (np.array([abs_dev_most[var][mdx] for mdx in range(len(abs_dev_most[var]))]) > x_hi)
            )
            msg = f"Warning [{var}]: x-axis limits [{x_lo:.3f}, {x_hi:.3f}] clips"
            if clipped_models:
                msg += f" models: {', '.join(clipped_models)}"
            if n_clipped_most > 0:
                msg += f" + {n_clipped_most} {model_with_most} ensemble member(s)"
            print(msg)

        if np.nanmax(all_r2) > 0.2:
            a.set_ylim(0, 1)

    if N_params == 3:
        ax[1, 1].set_visible(False)

    ax[corr_plot_attrs[N_params]['legend_panel']].legend(
        **corr_plot_attrs[N_params]['legend']
    )

    for a, label in zip(ax.flatten(), panel_labels):
        a.text(0.025, 0.97, label, transform=a.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')

    if save_figs:
        fname = make_figure_filename(fname, outdir=FIGS_PATH)
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fname}")


def plot_scatter_regression(x_data, y_data, slopes, intercepts, r2s,
                            fit, mip, model_name,
                            fname: str):
    ylabel_map = ylabel_maps[mip]  # map for y labels

    # Set up the plotting grid
    fig, ax = plt.subplots(
        *param_num_to_grid[MLE_FIT_ATTRS[fit]['N_params']]['grid'],
        figsize=param_num_to_grid[MLE_FIT_ATTRS[fit]['N_params']]['figsize']
    )

    for (a, param, x, y, slope, intercept, r2) in zip(ax.flatten(), MLE_FIT_ATTRS[fit]['param_names'], x_data, y_data, slopes, intercepts, r2s):
        # data and regression lines
        a.scatter(x, y, s=1, marker='.', c='grey')  # data
        a.plot(np.arange(min(x), max(x), (max(x) - min(x))/1000),
                np.arange(min(x), max(x), (max(x) - min(x))/1000),
                linewidth=2.5, linestyle='dashed', color='r')  # one to one line
        a.plot(x, slope * x + intercept, linestyle='solid', color='b', linewidth=2.5, label=f"r$^2$={r2:.2f}")  # regression line

        # aesthetics
        a.set_title(title_map[param])
        a.set_xlabel(xlabel_map[param])
        a.set_ylabel(ylabel_map[param])
        a.legend()
    
    fig.suptitle(f"Model: {model_name}")
    fig.tight_layout()

    fname = make_figure_filename(fname, outdir=CHECKS_PATH)
    fig.savefig(fname, dpi=300)
    print(f"     Figure saved to: {fname}")

def mutual_mask_perc(x, y, p_lo, p_hi):
    xnew = x[
        (x >= np.nanpercentile(x, p_lo)) & (x <= np.nanpercentile(x, p_hi))
    ]
    ynew = y[
        (x >= np.nanpercentile(x, p_lo)) & (x <= np.nanpercentile(x, p_hi))
    ]

    yfinal = ynew[
        (ynew >= np.nanpercentile(y, p_lo)) & (ynew <= np.nanpercentile(y, p_hi))
    ]

    xfinal = xnew[
        (ynew >= np.nanpercentile(y, p_lo)) & (ynew <= np.nanpercentile(y, p_hi))
    ]

    return xfinal, yfinal

def print_summary(r2s_prim, r2s_most, cmip_config, cmip_variable, model_with_most, N_members, fit):
    """Compute summary statistics and print R^2 comparisons for primary vs merged members."""
    merged_dict = {var: np.hstack([r2s_prim[var], r2s_most[var]]) for var in r2s_prim.keys()}

    models = list(cmip_config.iter_active_models(cmip_variable))
    prim_index = [m.name for m in models]
    merged_index = prim_index + [f'{model_with_most}_member_{i}' for i in range(N_members)]

    df_prim = pd.DataFrame(r2s_prim, index=prim_index)
    df_merged = pd.DataFrame(merged_dict, index=merged_index)

    prim_means = df_prim.mean()
    merged_means = df_merged.mean()
    prim_meds = df_prim.median()
    merged_meds = df_merged.median()
    prim_maxs = df_prim.max()
    merged_maxs = df_merged.max()

    print("SUMMARY STATISTICS")
    print("-" * 80)
    for p in MLE_FIT_ATTRS[fit]['param_names']:
        print(f"Parameter: {p}")
        print(f"    Primary members only - mean R^2: {prim_means[p]:.3f}, median R^2: {prim_meds[p]:.3f}, max R^2: {prim_maxs[p]:.3f}")
        print(f"    Primary + all members - mean R^2: {merged_means[p]:.3f}, median R^2: {merged_meds[p]:.3f}, max R^2: {merged_maxs[p]:.3f}\n")
