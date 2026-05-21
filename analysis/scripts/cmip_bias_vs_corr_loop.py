"""
Script to generate bias vs correlation plots for CMIP models vs ERA5.

This script makes bias vs correlation plots for CMIP models compared to ERA5 data.
It loops through fits, anomaly types, extreme types, and MIP types.
"""

# Import base packages for data analysis
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import linregress

# Import configuration paths
from evt_heat_waves.config import ERA5_PATH, MIP_FIT_PATH_DICT, MLE_FIT_ATTRS, FIGS_PATH
from evt_heat_waves.utils import extract_model_name
from evt_heat_waves.mip_fit.cmip_dataclass import CMIP6EnsembleConfig
from evt_heat_waves.plotting.plotting_presets import get_presets
from evt_heat_waves.plotting.bias_vs_corr import mutual_mask_perc, plot_scatter_regression, plot_bias_vs_corr

# Set up plotting presets
presets, _ = get_presets(markers=False)
plt.rcParams.update(presets)

# Configuration flags
SAVE_FIGS = True  # Whether to save the generated figures
GRID = '1deg'     # Grid resolution for the data
MAKE_CHECKS = False  # Whether to make individual check plots

# Parameter lists to loop over
fits = ['stat', 'nonstat_only_loc_trend']  # GEV fit types: stationary and non-stationary
ex_types = ['max', 'min']                   # Extreme types: maximum and minimum temperatures
anom_types = ['raw', 'trend', 'annmean']   # Anomaly types: raw data, trend anomalies, annual mean anomalies
mips = ['cmip', 'amip']                     # MIP types: CMIP6 and AMIP
TMIN = 1979                                 # Minimum year for data inclusion

# Loop through all combinations of parameters
for mip in mips:
    for fit in fits:
        for ex_type in ex_types:
            for anom_type in anom_types:
                print("-" * 80)
                print(f"Working on mip={mip}, fit={fit}, ex_type={ex_type}, anom_type={anom_type}")

                # Set data attributes
                if anom_type == 'raw':
                    data_type = 'raw'
                else:
                    data_type = f"anom_{anom_type}"

                # Attributes
                era5_variable = f't2m_annual_{ex_type}'
                cmip_variable = f'tas_annual_{ex_type}'
                trend_vars = ['loc_t', 'scale_t', 'shape_t']

                # Load CMIP config
                CMIPConfig = CMIP6EnsembleConfig.from_yaml(
                    MIP_FIT_PATH_DICT[mip]['config']['meta'],
                    MIP_FIT_PATH_DICT[mip]['config']['qc']
                )

                # Make file/model matcher
                data_path = MIP_FIT_PATH_DICT[mip]['data'] / cmip_variable / 'gev'
                fnames = [f for f in data_path.glob(f"*{fit}*{anom_type}*.nc") if "allmems" not in f.name]
                modelname_filepath_matcher = {
                    extract_model_name(f): f for f in fnames
                }

                # Import ERA5 data
                ds_era5 = xr.open_dataset(ERA5_PATH / 'gev' / f'era5_{era5_variable}_{GRID}_landonly_gev_{fit}_TMIN{TMIN}.nc', engine='netcdf4')
                PER_DECADE_CONVERSTION_FACTOR = 10 / len(ds_era5.year.values)

                era5_data = {p: ds_era5[f'{p}_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
                             if p in trend_vars
                             else ds_era5[f'{p}_{data_type}'].values.flatten()
                             for p in MLE_FIT_ATTRS[fit]['param_names']
                }

                # Get number of models
                N_active_models = len(list(CMIPConfig.iter_active_models(cmip_variable)))

                # Initialize arrays for primary members
                med_abs_dev_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
                mean_abs_dev_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
                slopes_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
                intercepts_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}
                r2s_prim = {p: np.zeros(N_active_models) for p in MLE_FIT_ATTRS[fit]['param_names']}

                # Find model with most members
                Nens_for_active_models = np.array([
                    len(m.all_members) for m in CMIPConfig.iter_active_models(cmip_variable)
                ])
                max_inds = np.where(Nens_for_active_models == np.max(Nens_for_active_models))[0]
                ind_ = max_inds[0]
                model_with_most = [m.name for m in CMIPConfig.iter_active_models(cmip_variable)][ind_]
                N_members = np.max(Nens_for_active_models)

                # Initialize arrays for most members model
                med_abs_dev_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
                mean_abs_dev_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
                slopes_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
                intercepts_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}
                r2s_most = {p: np.zeros(N_members) for p in MLE_FIT_ATTRS[fit]['param_names']}

                print(f"Analyzing {model_with_most} with {N_members} members")

                # Load allmems files for most members model
                gev_dir = MIP_FIT_PATH_DICT[mip]['data'] / cmip_variable / 'gev'
                pattern = f"*{model_with_most}*_{fit}_*allmems*{anom_type}.nc"
                allmems_files = sorted(gev_dir.glob(pattern))
                print(f"Found {len(allmems_files)} allmems files")

                # Process most members model
                plo = 5
                phi = 95
                era5_masked_most = []
                mip_masked_most = []

                if allmems_files:
                    ds = xr.open_dataset(allmems_files[0], engine='netcdf4')

                    for idx, mem in enumerate(ds.member_id.values):
                        tmp_ds = ds.sel(member_id=mem)

                        for param in MLE_FIT_ATTRS[fit]['param_names']:
                            if param in trend_vars:
                                mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
                            else:
                                mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten()

                            era5_masked, mip_masked = mutual_mask_perc(era5_data[param], mip_vals, plo, phi)
                            era5_masked_most.append(era5_masked)
                            mip_masked_most.append(mip_masked)

                            abs_dev = mip_masked - era5_masked

                            med_abs_dev_most[param][idx] = np.nanmedian(abs_dev)
                            mean_abs_dev_most[param][idx] = np.nanmean(abs_dev)

                            reg = linregress(era5_masked, mip_masked)
                            slopes_most[param][idx] = reg.slope
                            intercepts_most[param][idx] = reg.intercept
                            r2s_most[param][idx] = reg.rvalue**2

                        if MAKE_CHECKS:
                            plot_scatter_regression(
                                x_data=era5_masked_most,
                                y_data=mip_masked_most,
                                slopes=[slopes_most[param][idx] for param in MLE_FIT_ATTRS[fit]['param_names']],
                                intercepts=[intercepts_most[param][idx] for param in MLE_FIT_ATTRS[fit]['param_names']],
                                r2s=[r2s_most[param][idx] for param in MLE_FIT_ATTRS[fit]['param_names']],
                                fit=fit, mip=mip,
                                model_name=f"{model_with_most}_{mem}",
                                fname=f'{mip}_most_mems_{ex_type}_{anom_type}_{model_with_most}_{mem}_bias_vs_corr.png'
                            )
                            plt.close()

                        tmp_ds.close()

                    ds.close()

                # Process primary members
                era5_masked_prim = []
                mip_masked_prim = []

                for idx, m in enumerate(list(CMIPConfig.iter_active_models(cmip_variable))):
                    tmp_ds = xr.open_dataset(modelname_filepath_matcher[m.name])

                    for param in MLE_FIT_ATTRS[fit]['param_names']:
                        if param in trend_vars:
                            mip_vals = tmp_ds[f'{param}_{data_type}'].values.flatten() * PER_DECADE_CONVERSTION_FACTOR
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

                    if MAKE_CHECKS:
                        plot_scatter_regression(
                            x_data=era5_masked_prim,
                            y_data=mip_masked_prim,
                            slopes=[slopes_prim[param][idx] for param in MLE_FIT_ATTRS[fit]['param_names']],
                            intercepts=[intercepts_prim[param][idx] for param in MLE_FIT_ATTRS[fit]['param_names']],
                            r2s=[r2s_prim[param][idx] for param in MLE_FIT_ATTRS[fit]['param_names']],
                            fit=fit, mip=mip,
                            model_name=f"{m.name}",
                            fname=f'{mip}_prim_{ex_type}_{anom_type}_{m.name}_bias_vs_corr.png'
                        )
                        plt.close()

                    tmp_ds.close()

                # Generate plots
                plot_bias_vs_corr(
                    abs_dev_prim=med_abs_dev_prim, r2s_prim=r2s_prim,
                    abs_dev_most=med_abs_dev_most, r2s_most=r2s_most,
                    models=list(CMIPConfig.iter_active_models(cmip_variable)),
                    model_with_most=model_with_most, fit=fit, med_or_mean='med',
                    fname=f'all_{mip}_era5_r2_medbias_{cmip_variable}_{anom_type}_{fit}',
                    save_figs=SAVE_FIGS
                )

                plot_bias_vs_corr(
                    abs_dev_prim=mean_abs_dev_prim, r2s_prim=r2s_prim,
                    abs_dev_most=mean_abs_dev_most, r2s_most=r2s_most,
                    models=list(CMIPConfig.iter_active_models(cmip_variable)),
                    model_with_most=model_with_most, fit=fit, med_or_mean='mean',
                    fname=f'all_{mip}_era5_r2_meanbias_{cmip_variable}_{anom_type}_{fit}',
                    save_figs=SAVE_FIGS
                )

                # Close ERA5 dataset
                ds_era5.close()

                print(f"Completed {mip}, {fit}, {ex_type}, {anom_type}")