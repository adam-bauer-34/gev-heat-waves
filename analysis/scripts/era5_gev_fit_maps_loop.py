"""
Script to generate GEV parameter maps for ERA5 data.

This script makes all GEV parameter maps for ERA5 data. It loops
through TMINs, fit types, anomaly types, and extreme types.
"""

# Import base packages for data analysis
import xarray as xr
import matplotlib.pyplot as plt

# Import configuration paths
from evt_heat_waves.config import ERA5_PATH, FIGS_PATH

# Import custom plotting utilities
from evt_heat_waves.plotting.plotting_presets import get_presets
from evt_heat_waves.plotting.gev_param_plots import plot_gev_parameters, print_trend_significance

# Set up plotting presets
presets, _ = get_presets(markers=False)
plt.rcParams.update(presets)

# Configuration flags
save_figs = True  # Whether to save the generated figures
GRID = '1deg'     # Grid resolution for the data

# Parameter configs to loop over, specifically selected for paper SI
fit_configs = {
    'fig_2': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_raw': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'raw',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_trend': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'trend',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_min': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'min',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_min_1950': {
        'primary_fit': 'stat_gumbel',
        'secondary_fit': 'nonstat_gumbel_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1950,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_full_gumbel': {
        'primary_fit': 'nonstat_gumbel_only_loc_trend',
        'secondary_fit': None,
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_minxi': {
        'primary_fit': 'stat_gumbel_minxi',
        'secondary_fit': 'nonstat_minxi_only_loc_trend',
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 3,
        'param_panel_order': ['loc', 'scale', 'loc_t']
    },
    'si_nonstat': {
        'primary_fit': 'nonstat',
        'secondary_fit': None,
        'ex_type': 'max',
        'anom_type': 'annmean',
        'TMIN': 1979,
        'N_params': 6,
        'param_panel_order': ['loc', 'loc_t', 'scale', 'scale_t', 'shape', 'shape_t']
    }
}

# Loop through all combinations of parameters
for fit, config in fit_configs.items():
    primary_fit = config['primary_fit']
    secondary_fit = config['secondary_fit']
    ex_type = config['ex_type']
    anom_type = config['anom_type']
    TMIN = config['TMIN']
    N_PARAMS = config['N_params']
    param_panel_order = config['param_panel_order']

    print("-" * 90)
    print(f"Working on {fit}")
    print(f"Config: primary_fit={primary_fit}, secondary_fit={secondary_fit}, ex_type={ex_type}, anom_type={anom_type}, TMIN={TMIN}")

    # Load the GEV-fitted dataset for the current parameter combination
    try:
        ds_prim = xr.open_dataset(ERA5_PATH / 'gev' / f'era5_t2m_annual_{ex_type}_{GRID}_landonly_gev_{primary_fit}_TMIN{TMIN}_{anom_type}.nc', engine='netcdf4')

        ds_sec = (xr.open_dataset(ERA5_PATH / 'gev' / f'era5_t2m_annual_{ex_type}_{GRID}_landonly_gev_{secondary_fit}_TMIN{TMIN}_{anom_type}.nc', engine='netcdf4')
                    if secondary_fit is not None
                    else None)
        
    except FileNotFoundError:
        print(f"File not found for primary_fit={primary_fit}, secondary_fit={secondary_fit}, ex_type={ex_type}, anom_type={anom_type}, TMIN={TMIN}")
        continue

    fname = (f'era5-t2m-{ex_type}-{anom_type}-gev-parameters-{GRID}-{TMIN}-{primary_fit}-{secondary_fit}-hybrid'
                if secondary_fit is not None
                else f'era5-t2m-{ex_type}-{anom_type}-gev-parameters-{GRID}-{TMIN}-{primary_fit}')

    # Generate and save the GEV parameter plots
    plot_gev_parameters(ds_prim, ds_sec, N_params=N_PARAMS, anom_type=anom_type,
                        param_panel_order=param_panel_order, ex_type=ex_type,
                        save_figs=save_figs,
                        fname=fname, 
                        output_dir=FIGS_PATH)
    
    if secondary_fit is not None and fit == 'fig_2':
        print_trend_significance(ds_sec, fit=secondary_fit, anom_type=anom_type)
    
    # Close the dataset to free memory
    ds_prim.close()

    if secondary_fit is not None:
        ds_sec.close()