"""
Script to generate bias vs correlation plots for CMIP models vs ERA5.

This script makes bias vs correlation plots for CMIP models compared to ERA5 data.
It loops through fits, anomaly types, extreme types, and MIP types.
"""

# Import base packages for data analysis
import matplotlib.pyplot as plt

# Import configuration paths
from evt_heat_waves.plotting.plotting_presets import get_presets
from evt_heat_waves.plotting.bias_vs_corr import compute_primary_member_bias_vs_corr, plot_bias_vs_corr

# Set up plotting presets
presets, _ = get_presets(markers=False)
plt.rcParams.update(presets)

# Configuration flags
SAVE_FIGS = True  # Whether to save the generated figures

fit_configs = {
    'fig_3': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t'],
        'GRID': '1deg',
        'data_type': 'anom_annmean',
        'mip': 'cmip',
        'primary_vars': ['loc', 'scale'],
        'secondary_vars': ['loc_t']
    },
    'si_raw': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'raw',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t'],
        'GRID': '1deg',
        'data_type': 'raw',
        'mip': 'cmip',
        'primary_vars': ['loc', 'scale'],
        'secondary_vars': ['loc_t']
    },
    'si_trend': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'trend',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t'],
        'GRID': '1deg',
        'data_type': 'anom_trend',
        'mip': 'cmip',
        'primary_vars': ['loc', 'scale'],
        'secondary_vars': ['loc_t']
    },
    'si_max_amip': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t'],
        'GRID': '1deg',
        'data_type': 'anom_annmean',
        'mip': 'amip',
        'primary_vars': ['loc', 'scale'],
        'secondary_vars': ['loc_t']
    },
    'si_min_amip': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'min',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t'],
        'GRID': '1deg',
        'data_type': 'anom_annmean',
        'mip': 'amip',
        'primary_vars': ['loc', 'scale'],
        'secondary_vars': ['loc_t']
    },
    'si_min_cmip': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'min',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t'],
        'GRID': '1deg',
        'data_type': 'anom_annmean',
        'mip': 'cmip',
        'primary_vars': ['loc', 'scale'],
        'secondary_vars': ['loc_t']
    },
}

# loop through configs, do analysis, and make plots.
for fit, config in fit_configs.items():
    # unpack config
    primary_fit = config['primary_fit']
    secondary_fit = config['secondary_fit']
    ex_type = config['ex_type']
    anom_type = config['anom_type']
    TMIN = config['TMIN']
    N_PARAMS = config['N_params']
    param_panel_order = config['param_panel_order']
    GRID = config['GRID']
    data_type = config['data_type']
    mip = config['mip']
    primary_vars = config['primary_vars']
    secondary_vars = config['secondary_vars']

    print("-" * 90)
    print(f"[{fit}] {mip} | {ex_type} | {anom_type} | {data_type} | params={N_PARAMS}")

    # analyze primary and secondary fit
    primary_fit_output = compute_primary_member_bias_vs_corr(primary_fit,
                                                            anom_type,
                                                            ex_type,
                                                            data_type,
                                                            TMIN,
                                                            GRID,
                                                            mip)

    secondary_fit_output = compute_primary_member_bias_vs_corr(secondary_fit,
                                                            anom_type,
                                                            ex_type,
                                                            data_type,
                                                            TMIN,
                                                            GRID,
                                                            mip)

    # group into one final output
    final_output = {}
    for var_key in ['med_abs_dev_prim', 'med_abs_dev_most', 'mean_abs_dev_prim', 'mean_abs_dev_most', 'r2s_prim', 'r2s_most']:
        merged = {}
        merged.update({k: primary_fit_output[var_key][k] for k in primary_vars})
        merged.update({k: secondary_fit_output[var_key][k] for k in secondary_vars})
        final_output[var_key] = merged

    # plot
    plot_bias_vs_corr(
        abs_dev_prim=final_output['med_abs_dev_prim'], r2s_prim=final_output['r2s_prim'],
        abs_dev_most=final_output['med_abs_dev_most'], r2s_most=final_output['r2s_most'],
        param_order=param_panel_order, N_params=N_PARAMS,
        models=list(primary_fit_output['cmip_config'].iter_active_models(f'tas_annual_{ex_type}')),
        model_with_most=primary_fit_output['model_with_most'], med_or_mean='med',
        fname=f'{mip}_era5_r2_bias_{ex_type}_{data_type}_{primary_fit}_{secondary_fit}_hybrid',
        save_figs=SAVE_FIGS
    )