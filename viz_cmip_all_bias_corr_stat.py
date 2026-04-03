"""
Convert viz_cmip_all_bias_corr_stat notebook to a script that loops over fit types and anomaly types.
Generates figures and collects summary statistics in a DataFrame saved to CSV.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from pathlib import Path

from config import DATA_ROOT
from src.utils import extract_model_name
from src.cmip_dataclass import CMIP6EnsembleConfig
from src.plotting_presets import get_presets
from ambpy.plotutils import make_figure_filename

# Setup plotting
presets, _ = get_presets(markers=False)
plt.rcParams.update(presets)

# Set color and marker sets for CMIP models
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', 
          '#AA3377', '#EE7733', '#009988', '#332288', '#BBBBBB']
markers = ['o', 's', 'D']

# Constants
TMIN = 1979
era5_variable = 't2m_annual_max'
cmip_variable = 'tas_annual_max'
plo = 5
phi = 95
save_figs = True

# Set random seed
np.random.seed(4)

# Load CMIP config
CMIPConfig = CMIP6EnsembleConfig.from_yaml(
    'config/meta.yaml',
    'config/qc.yaml'
)


def mutual_mask_perc(x, y, p_lo, p_hi):
    """Mutually mask percentiles between two arrays."""
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


def plot_mean_bias_figure(mean_abs_dev_prim, mean_abs_dev_most, r2s_prim, r2s_most, 
                          models, model_with_most, ds, fit, anom_type):
    """Generate figure showing mean absolute deviation vs r2."""
    if fit == 'nonstat':
        fig, ax = plt.subplots(3, 2, figsize=(16, 20))
        titles = [r'Location Parameter | $\mu_0$ $(^\circ$C$)$', 
                  r"Location Parameter Trend | $\mu_1$ $(^\circ$C / dec$)$",
                  r'Scale Parameter | $\sigma_0$ $(^\circ$C$)$', 
                  r"Scale Parameter Trend | $\sigma_1$ $(^\circ$C / dec$)$",
                  r'Shape Parameter | $\xi_0$ $(-)$', 
                  r"Shape Parameter Trend | $\xi_1$ $($dec$^{-1})$"]
        have_ylabels = [0, 2, 4]
        have_xlabels = [4, 5]
    else:  # stat
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        titles = [r'Location Parameter | $\mu_0$ $(^\circ$C$)$',
                  r'Scale Parameter | $\sigma_0$ $(^\circ$C$)$',
                  r'Shape Parameter | $\xi_0$ $(-)$']
        have_ylabels = [0]
        have_xlabels = [0, 1, 2]

    for (idx, (a, var, title)) in enumerate(zip(ax.flatten(), r2s_prim.keys(), titles)):
        a.axvline(0, 0, 1, linestyle='solid', color='k')
        a.axhline(0, -1, 1, linestyle='solid', color='k')
        
        for mdx, m in enumerate(models):
            marker = markers[mdx // 10]
            color = colors[mdx % 10]
            a.scatter(mean_abs_dev_prim[var][mdx], r2s_prim[var][mdx], 
                     s=90, marker=marker, color=color, label=m.name, zorder=100)
            a.set_title(title)
            if idx in have_ylabels:
                a.set_ylabel("$r^2$")
            if idx in have_xlabels:
                a.set_xlabel("Mean absolute deviation: MODEL $-$ ERA5")

        for mdx, m in zip(range(len(ds.member_id.values)), ds.member_id.values):
            a.scatter(mean_abs_dev_most[var][mdx], r2s_most[var][mdx], 
                     s=60, marker='.', color='grey', zorder=1,
                     label=f'{model_with_most} Ensemble Members' if mdx == 0 else None)

    if fit == 'nonstat':
        ax[1,1].legend(bbox_to_anchor=(1.05, 1.37), frameon=True)
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
    else:
        ax[1].legend(bbox_to_anchor=(1.63, -0.2), frameon=True, ncol=4)
        labels = ['A', 'B', 'C']

    for a, label in zip(ax.flatten(), labels):
        a.text(0.025, 0.97, label, transform=a.transAxes,
              fontsize=16, fontweight='bold', va='top', ha='left')

    if save_figs:
        fname = make_figure_filename(
            f'all_cmip_era5_r2_avgbias_{cmip_variable}_{anom_type}_{fit}', 
            'png', 'figs/analysis'
        )
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✍️ Figure (mean bias) saved to: {fname}")
    
    plt.close(fig)


def plot_median_bias_figure(med_abs_dev_prim, med_abs_dev_most, r2s_prim, r2s_most, 
                            models, model_with_most, ds, fit, anom_type):
    """Generate figure showing median absolute deviation vs r2."""
    if fit == 'nonstat':
        fig, ax = plt.subplots(3, 2, figsize=(16, 20))
        titles = [r'Location Parameter | $\mu_0$ $(^\circ$C$)$', 
                  r"Location Parameter Trend | $\mu_1$ $(^\circ$C / dec$)$",
                  r'Scale Parameter | $\sigma_0$ $(^\circ$C$)$', 
                  r"Scale Parameter Trend | $\sigma_1$ $(^\circ$C / dec$)$",
                  r'Shape Parameter | $\xi_0$ $(-)$', 
                  r"Shape Parameter Trend | $\xi_1$ $($dec$^{-1})$"]
        have_ylabels = [0, 2, 4]
        have_xlabels = [4, 5]
    else:  # stat
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        titles = [r'Location Parameter | $\mu_0$ $(^\circ$C$)$',
                  r'Scale Parameter | $\sigma_0$ $(^\circ$C$)$',
                  r'Shape Parameter | $\xi_0$ $(-)$']
        have_ylabels = [0]
        have_xlabels = [0, 1, 2]

    for (idx, (a, var, title)) in enumerate(zip(ax.flatten(), r2s_prim.keys(), titles)):
        a.axvline(0, 0, 1, linestyle='solid', color='k')
        a.axhline(0, -1, 1, linestyle='solid', color='k')
        
        for mdx, m in enumerate(models):
            marker = markers[mdx // 10]
            color = colors[mdx % 10]
            a.scatter(med_abs_dev_prim[var][mdx], r2s_prim[var][mdx], 
                     s=90, marker=marker, color=color, label=m.name, zorder=100)
            a.set_title(title)
            if idx in have_ylabels:
                a.set_ylabel("$r^2$")
            if idx in have_xlabels:
                a.set_xlabel("Median absolute deviation: MODEL $-$ ERA5")

        for mdx, m in zip(range(len(ds.member_id.values)), ds.member_id.values):
            a.scatter(med_abs_dev_most[var][mdx], r2s_most[var][mdx], 
                     s=60, marker='.', color='grey', zorder=1,
                     label=f'{model_with_most} Ensemble Members' if mdx == 0 else None)

    if fit == 'nonstat':
        ax[1,1].legend(bbox_to_anchor=(1.05, 1.37), frameon=True)
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
    else:
        ax[1].legend(bbox_to_anchor=(1.63, -0.2), frameon=True, ncol=4)
        labels = ['A', 'B', 'C']

    for a, label in zip(ax.flatten(), labels):
        a.text(0.025, 0.97, label, transform=a.transAxes,
              fontsize=16, fontweight='bold', va='top', ha='left')

    if save_figs:
        fname = make_figure_filename(
            f'all_cmip_era5_r2_medbias_{cmip_variable}_{anom_type}_{fit}', 
            'png', 'figs/analysis'
        )
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✍️ Figure (median bias) saved to: {fname}")
    
    plt.close(fig)


def process_fit_and_anom(fit, anom_type):
    """Process a single fit type and anomaly type combination."""
    print(f"\n{'='*60}")
    print(f"Processing: fit={fit}, anom_type={anom_type}")
    print(f"{'='*60}\n")

    # Set data type
    if anom_type == 'raw':
        data_type = 'raw'
    else:
        data_type = f"anom_{anom_type}"

    # Load ERA5 data
    print(f"Loading ERA5 data...")
    ds_era5 = xr.open_dataset(
        DATA_ROOT / 'ERA5' / 'gev' / f'era5_{era5_variable}_1deg_landonly_gev_{fit}_TMIN{TMIN}.nc',
        engine='netcdf4'
    )
    PER_DECADE_CONVERSTION_FACTOR = 10 / len(ds_era5.year.values)

    # Extract ERA5 data
    if fit == 'nonstat':
        era5_loc = ds_era5[f'loc_{data_type}'].values.flatten()
        era5_loc_trend = ds_era5[f'loc_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
        era5_scale = ds_era5[f'scale_{data_type}'].values.flatten()
        era5_scale_trend = ds_era5[f'scale_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
        era5_shape = ds_era5[f'shape_{data_type}'].values.flatten()
        era5_shape_trend = ds_era5[f'shape_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
    else:  # stat
        era5_loc = ds_era5[f'loc_{data_type}'].values.flatten()
        era5_scale = ds_era5[f'scale_{data_type}'].values.flatten()
        era5_shape = ds_era5[f'shape_{data_type}'].values.flatten()

    ds_era5.close()

    # Get model data paths
    data_path = DATA_ROOT / 'CMIP6' / cmip_variable / 'gev'
    fnames = [f for f in data_path.glob(f"*{fit}*{anom_type}*.nc") if "allmems" not in f.name]
    modelname_filepath_matcher = {extract_model_name(f): f for f in fnames}

    # Initialize models
    models = list(CMIPConfig.iter_active_models(cmip_variable))
    N_active_models = len(models)

    # Initialize statistics containers
    if fit == 'nonstat':
        med_abs_dev_prim = {
            'loc': np.zeros(N_active_models),
            'loc_trend': np.zeros(N_active_models),
            'scale': np.zeros(N_active_models),
            'scale_trend': np.zeros(N_active_models),
            'shape': np.zeros(N_active_models),
            'shape_trend': np.zeros(N_active_models)
        }
        mean_abs_dev_prim = {k: np.zeros(N_active_models) for k in med_abs_dev_prim.keys()}
        r2s_prim = {k: np.zeros(N_active_models) for k in med_abs_dev_prim.keys()}
    else:  # stat
        med_abs_dev_prim = {
            'loc': np.zeros(N_active_models),
            'scale': np.zeros(N_active_models),
            'shape': np.zeros(N_active_models),
        }
        mean_abs_dev_prim = {k: np.zeros(N_active_models) for k in med_abs_dev_prim.keys()}
        r2s_prim = {k: np.zeros(N_active_models) for k in med_abs_dev_prim.keys()}

    # Find model with most ensemble members
    Nens_for_active_models = np.array([len(m.all_members) for m in models])
    max_inds = np.where(Nens_for_active_models == np.max(Nens_for_active_models))[0]
    model_with_most = models[max_inds[0]].name
    N_members = np.max(Nens_for_active_models)
    
    print(f"Model with most members: {model_with_most} ({N_members} members)\n")

    # Initialize containers for model with most members
    med_abs_dev_most = {k: np.zeros(N_members) for k in med_abs_dev_prim.keys()}
    mean_abs_dev_most = {k: np.zeros(N_members) for k in med_abs_dev_prim.keys()}
    r2s_most = {k: np.zeros(N_members) for k in med_abs_dev_prim.keys()}

    # Load ensemble members for model with most members
    print(f"Loading ensemble data for {model_with_most}...")
    gev_dir = DATA_ROOT / 'CMIP6' / cmip_variable / 'gev'
    pattern = f"*{model_with_most}*_{fit}*allmems*{anom_type}.nc"
    allmems_files = sorted(gev_dir.glob(pattern))
    
    if len(allmems_files) == 0:
        print(f"ERROR: No allmems files found for pattern: {pattern}")
        return None
    
    ds = xr.open_dataset(allmems_files[0], engine='netcdf4')

    # Process members for model with most ensemble members
    print(f"Processing {len(ds.member_id.values)} ensemble members...")
    for idx, mem in enumerate(ds.member_id.values):
        tmp_ds = ds.sel(member_id=mem)

        if fit == 'nonstat':
            m_loc = tmp_ds[f'loc_{data_type}'].values.flatten()
            m_loc_trend = tmp_ds[f'loc_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
            m_scale = tmp_ds[f'scale_{data_type}'].values.flatten()
            m_scale_trend = tmp_ds[f'scale_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
            m_shape = tmp_ds[f'shape_{data_type}'].values.flatten()
            m_shape_trend = tmp_ds[f'shape_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR

            era5_loc_masked, m_loc_masked = mutual_mask_perc(era5_loc, m_loc, plo, phi)
            era5_loc_trend_masked, m_loc_trend_masked = mutual_mask_perc(era5_loc_trend, m_loc_trend, plo, phi)
            era5_scale_masked, m_scale_masked = mutual_mask_perc(era5_scale, m_scale, plo, phi)
            era5_scale_trend_masked, m_scale_trend_masked = mutual_mask_perc(era5_scale_trend, m_scale_trend, plo, phi)
            era5_shape_masked, m_shape_masked = mutual_mask_perc(era5_shape, m_shape, plo, phi)
            era5_shape_trend_masked, m_shape_trend_masked = mutual_mask_perc(era5_shape_trend, m_shape_trend, plo, phi)

            tmp_abs_dev_loc = m_loc_masked - era5_loc_masked
            tmp_abs_dev_loc_trend = m_loc_trend_masked - era5_loc_trend_masked
            tmp_abs_dev_scale = m_scale_masked - era5_scale_masked
            tmp_abs_dev_scale_trend = m_scale_trend_masked - era5_scale_trend_masked
            tmp_abs_dev_shape = m_shape_masked - era5_shape_masked
            tmp_abs_dev_shape_trend = m_shape_trend_masked - era5_shape_trend_masked

            med_abs_dev_most['loc'][idx] = np.nanmedian(tmp_abs_dev_loc)
            med_abs_dev_most['loc_trend'][idx] = np.nanmedian(tmp_abs_dev_loc_trend)
            med_abs_dev_most['scale'][idx] = np.nanmedian(tmp_abs_dev_scale)
            med_abs_dev_most['scale_trend'][idx] = np.nanmedian(tmp_abs_dev_scale_trend)
            med_abs_dev_most['shape'][idx] = np.nanmedian(tmp_abs_dev_shape)
            med_abs_dev_most['shape_trend'][idx] = np.nanmedian(tmp_abs_dev_shape_trend)

            mean_abs_dev_most['loc'][idx] = np.nanmean(tmp_abs_dev_loc)
            mean_abs_dev_most['loc_trend'][idx] = np.nanmean(tmp_abs_dev_loc_trend)
            mean_abs_dev_most['scale'][idx] = np.nanmean(tmp_abs_dev_scale)
            mean_abs_dev_most['scale_trend'][idx] = np.nanmean(tmp_abs_dev_scale_trend)
            mean_abs_dev_most['shape'][idx] = np.nanmean(tmp_abs_dev_shape)
            mean_abs_dev_most['shape_trend'][idx] = np.nanmean(tmp_abs_dev_shape_trend)

            r2s_most['loc'][idx] = linregress(era5_loc_masked, m_loc_masked).rvalue**2
            r2s_most['loc_trend'][idx] = linregress(era5_loc_trend_masked, m_loc_trend_masked).rvalue**2
            r2s_most['scale'][idx] = linregress(era5_scale_masked, m_scale_masked).rvalue**2
            r2s_most['scale_trend'][idx] = linregress(era5_scale_trend_masked, m_scale_trend_masked).rvalue**2
            r2s_most['shape'][idx] = linregress(era5_shape_masked, m_shape_masked).rvalue**2
            r2s_most['shape_trend'][idx] = linregress(era5_shape_trend_masked, m_shape_trend_masked).rvalue**2

        else:  # stat
            m_loc = tmp_ds[f'loc_{data_type}'].values.flatten()
            m_scale = tmp_ds[f'scale_{data_type}'].values.flatten()
            m_shape = tmp_ds[f'shape_{data_type}'].values.flatten()

            era5_loc_masked, m_loc_masked = mutual_mask_perc(era5_loc, m_loc, plo, phi)
            era5_scale_masked, m_scale_masked = mutual_mask_perc(era5_scale, m_scale, plo, phi)
            era5_shape_masked, m_shape_masked = mutual_mask_perc(era5_shape, m_shape, plo, phi)

            tmp_abs_dev_loc = m_loc_masked - era5_loc_masked
            tmp_abs_dev_scale = m_scale_masked - era5_scale_masked
            tmp_abs_dev_shape = m_shape_masked - era5_shape_masked

            med_abs_dev_most['loc'][idx] = np.nanmedian(tmp_abs_dev_loc)
            med_abs_dev_most['scale'][idx] = np.nanmedian(tmp_abs_dev_scale)
            med_abs_dev_most['shape'][idx] = np.nanmedian(tmp_abs_dev_shape)

            mean_abs_dev_most['loc'][idx] = np.nanmean(tmp_abs_dev_loc)
            mean_abs_dev_most['scale'][idx] = np.nanmean(tmp_abs_dev_scale)
            mean_abs_dev_most['shape'][idx] = np.nanmean(tmp_abs_dev_shape)

            r2s_most['loc'][idx] = linregress(era5_loc_masked, m_loc_masked).rvalue**2
            r2s_most['scale'][idx] = linregress(era5_scale_masked, m_scale_masked).rvalue**2
            r2s_most['shape'][idx] = linregress(era5_shape_masked, m_shape_masked).rvalue**2

    ds.close()

    # Process primary members for all models
    print(f"Processing {N_active_models} models' primary members...")
    for (i, m) in enumerate(models):
        if m.name not in modelname_filepath_matcher:
            print(f"  Skipping {m.name} (not found in file matcher)")
            continue
        
        tmp_ds = xr.open_dataset(modelname_filepath_matcher[m.name])

        if fit == 'nonstat':
            m_loc = tmp_ds[f'loc_{data_type}'].values.flatten()
            m_loc_trend = tmp_ds[f'loc_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
            m_scale = tmp_ds[f'scale_{data_type}'].values.flatten()
            m_scale_trend = tmp_ds[f'scale_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
            m_shape = tmp_ds[f'shape_{data_type}'].values.flatten()
            m_shape_trend = tmp_ds[f'shape_t_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR

            era5_loc_masked, m_loc_masked = mutual_mask_perc(era5_loc, m_loc, plo, phi)
            era5_loc_trend_masked, m_loc_trend_masked = mutual_mask_perc(era5_loc_trend, m_loc_trend, plo, phi)
            era5_scale_masked, m_scale_masked = mutual_mask_perc(era5_scale, m_scale, plo, phi)
            era5_scale_trend_masked, m_scale_trend_masked = mutual_mask_perc(era5_scale_trend, m_scale_trend, plo, phi)
            era5_shape_masked, m_shape_masked = mutual_mask_perc(era5_shape, m_shape, plo, phi)
            era5_shape_trend_masked, m_shape_trend_masked = mutual_mask_perc(era5_shape_trend, m_shape_trend, plo, phi)

            tmp_abs_dev_loc = m_loc_masked - era5_loc_masked
            tmp_abs_dev_loc_trend = m_loc_trend_masked - era5_loc_trend_masked
            tmp_abs_dev_scale = m_scale_masked - era5_scale_masked
            tmp_abs_dev_scale_trend = m_scale_trend_masked - era5_scale_trend_masked
            tmp_abs_dev_shape = m_shape_masked - era5_shape_masked
            tmp_abs_dev_shape_trend = m_shape_trend_masked - era5_shape_trend_masked

            med_abs_dev_prim['loc'][i] = np.nanmedian(tmp_abs_dev_loc)
            med_abs_dev_prim['loc_trend'][i] = np.nanmedian(tmp_abs_dev_loc_trend)
            med_abs_dev_prim['scale'][i] = np.nanmedian(tmp_abs_dev_scale)
            med_abs_dev_prim['scale_trend'][i] = np.nanmedian(tmp_abs_dev_scale_trend)
            med_abs_dev_prim['shape'][i] = np.nanmedian(tmp_abs_dev_shape)
            med_abs_dev_prim['shape_trend'][i] = np.nanmedian(tmp_abs_dev_shape_trend)

            mean_abs_dev_prim['loc'][i] = np.nanmean(tmp_abs_dev_loc)
            mean_abs_dev_prim['loc_trend'][i] = np.nanmean(tmp_abs_dev_loc_trend)
            mean_abs_dev_prim['scale'][i] = np.nanmean(tmp_abs_dev_scale)
            mean_abs_dev_prim['scale_trend'][i] = np.nanmean(tmp_abs_dev_scale_trend)
            mean_abs_dev_prim['shape'][i] = np.nanmean(tmp_abs_dev_shape)
            mean_abs_dev_prim['shape_trend'][i] = np.nanmean(tmp_abs_dev_shape_trend)

            r2s_prim['loc'][i] = linregress(era5_loc_masked, m_loc_masked).rvalue**2
            r2s_prim['loc_trend'][i] = linregress(era5_loc_trend_masked, m_loc_trend_masked).rvalue**2
            r2s_prim['scale'][i] = linregress(era5_scale_masked, m_scale_masked).rvalue**2
            r2s_prim['scale_trend'][i] = linregress(era5_scale_trend_masked, m_scale_trend_masked).rvalue**2
            r2s_prim['shape'][i] = linregress(era5_shape_masked, m_shape_masked).rvalue**2
            r2s_prim['shape_trend'][i] = linregress(era5_shape_trend_masked, m_shape_trend_masked).rvalue**2

        else:  # stat
            m_loc = tmp_ds[f'loc_{data_type}'].values.flatten()
            m_scale = tmp_ds[f'scale_{data_type}'].values.flatten()
            m_shape = tmp_ds[f'shape_{data_type}'].values.flatten()

            era5_loc_masked, m_loc_masked = mutual_mask_perc(era5_loc, m_loc, plo, phi)
            era5_scale_masked, m_scale_masked = mutual_mask_perc(era5_scale, m_scale, plo, phi)
            era5_shape_masked, m_shape_masked = mutual_mask_perc(era5_shape, m_shape, plo, phi)

            tmp_abs_dev_loc = m_loc_masked - era5_loc_masked
            tmp_abs_dev_scale = m_scale_masked - era5_scale_masked
            tmp_abs_dev_shape = m_shape_masked - era5_shape_masked

            med_abs_dev_prim['loc'][i] = np.nanmedian(tmp_abs_dev_loc)
            med_abs_dev_prim['scale'][i] = np.nanmedian(tmp_abs_dev_scale)
            med_abs_dev_prim['shape'][i] = np.nanmedian(tmp_abs_dev_shape)

            mean_abs_dev_prim['loc'][i] = np.nanmean(tmp_abs_dev_loc)
            mean_abs_dev_prim['scale'][i] = np.nanmean(tmp_abs_dev_scale)
            mean_abs_dev_prim['shape'][i] = np.nanmean(tmp_abs_dev_shape)

            r2s_prim['loc'][i] = linregress(era5_loc_masked, m_loc_masked).rvalue**2
            r2s_prim['scale'][i] = linregress(era5_scale_masked, m_scale_masked).rvalue**2
            r2s_prim['shape'][i] = linregress(era5_shape_masked, m_shape_masked).rvalue**2

        tmp_ds.close()

    # Generate figures
    print(f"Generating figures...")
    plot_mean_bias_figure(mean_abs_dev_prim, mean_abs_dev_most, r2s_prim, r2s_most, 
                         models, model_with_most, ds, fit, anom_type)
    plot_median_bias_figure(med_abs_dev_prim, med_abs_dev_most, r2s_prim, r2s_most, 
                           models, model_with_most, ds, fit, anom_type)

    # Compute summary statistics
    print(f"Computing summary statistics...")
    merged_dict = {var: np.hstack([r2s_prim[var], r2s_most[var]]) for var in r2s_prim.keys()}
    df_prim = pd.DataFrame(r2s_prim, index=[m.name for m in models])
    df_merged = pd.DataFrame(merged_dict, index=[m.name for m in models] + 
                            [f'{model_with_most}_member_{i}' for i in range(N_members)])

    # Create results dictionary
    results = {
        'fit': fit,
        'anom_type': anom_type,
        'dataset': 'primary_members',
        'mean': df_prim.mean().to_dict(),
        'median': df_prim.median().to_dict(),
        'max': df_prim.max().to_dict(),
    }
    
    results_merged = {
        'fit': fit,
        'anom_type': anom_type,
        'dataset': 'primary_plus_ensemble',
        'mean': df_merged.mean().to_dict(),
        'median': df_merged.median().to_dict(),
        'max': df_merged.max().to_dict(),
    }

    return [results, results_merged]


def main():
    """Main execution function."""
    fit_types = ['stat', 'nonstat']
    anom_types = ['annmean', 'trend', 'raw']

    all_results = []

    for fit in fit_types:
        for anom_type in anom_types:
            try:
                results = process_fit_and_anom(fit, anom_type)
                if results is not None:
                    all_results.extend(results)
            except Exception as e:
                print(f"ERROR processing fit={fit}, anom_type={anom_type}: {e}")
                import traceback
                traceback.print_exc()

    # Create DataFrame from all results
    print(f"\n{'='*60}")
    print("Creating summary statistics DataFrame...")
    print(f"{'='*60}\n")

    df_stats = pd.DataFrame(all_results)
    
    # Save to CSV
    stats_dir = DATA_ROOT / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)
    output_file = stats_dir / 'gev_param_summary_stats.csv'
    
    df_stats.to_csv(output_file, index=False)
    print(f"✅ Summary statistics saved to: {output_file}")
    print(f"\nDataFrame shape: {df_stats.shape}")
    print(f"\nFirst few rows:")
    print(df_stats.head())


if __name__ == '__main__':
    main()
