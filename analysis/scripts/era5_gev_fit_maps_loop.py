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
from evt_heat_waves.plotting.gev_param_plots import plot_gev_parameters, print_summary

# Set up plotting presets
presets, _ = get_presets(markers=False)
plt.rcParams.update(presets)

# Configuration flags
save_figs = True  # Whether to save the generated figures
GRID = '1deg'     # Grid resolution for the data

# Parameter lists to loop over
fits = ['stat_new', 'nonstat_only_loc_trend']  # GEV fit types: stationary and non-stationary
ex_types = ['max', 'min']                       # Extreme types: maximum and minimum temperatures
anom_types = ['raw', 'trend', 'annmean']        # Anomaly types: raw data, trend anomalies, annual mean anomalies
tmins = [1979, 1950]                           # Minimum years for data inclusion

# Loop through all combinations of parameters
for fit in fits:
    for ex_type in ex_types:
        for anom_type in anom_types:
            for TMIN in tmins:
                print("-" * 80)
                print(f"Working on fit={fit}, ex_type={ex_type}, anom_type={anom_type}, TMIN={TMIN}")
                # Load the GEV-fitted dataset for the current parameter combination
                ds = xr.open_dataset(ERA5_PATH / 'gev' / f'era5_t2m_annual_{ex_type}_{GRID}_landonly_gev_{fit}_TMIN{TMIN}_{anom_type}.nc', engine='netcdf4')
                
                # Generate and save the GEV parameter plots
                plot_gev_parameters(ds, fit=fit, anom_type=anom_type, save_figs=save_figs,
                                    fname=f'era5-t2m-{ex_type}-{anom_type}-gev-{fit}-parameters-{GRID}-{TMIN}', 
                                    output_dir=FIGS_PATH)
                
                # Print summary statistics for the dataset
                print_summary(ds, fit=fit, anom_type=anom_type)

                # Close the dataset to free memory
                ds.close()