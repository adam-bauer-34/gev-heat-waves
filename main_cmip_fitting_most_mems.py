"""Main file for GEV fitting of CMIP model that has the most members.

Adam Michael Bauer
UChicago
Jan 2026

To run: main_cmip_fitting.py STAT

Last edited: 1/29/2026, 12:03 PM CST
"""

import os
import sys
import shutil
import pprint

import xarray as xr
import numpy as np

from config import DATA_ROOT
from mle_claude import ds_mle_fit, reset_mle_stats
from src.cmip_dataclass import CMIP6EnsembleConfig
from src.utils import extract_model_name

# import command line grid
STAT, = sys.argv[1:]
width = shutil.get_terminal_size(fallback=(80, 20)).columns

# setup CMIP config object
CMIPConfig = CMIP6EnsembleConfig.from_yaml("config/meta.yaml", 
                                            "config/qc.yaml")

# define variables and open datasets
vars = ['tas_annual_max']

for var in vars:
    print('='*width)
    print("🏋🏼‍♀️ Carrying out GEV fits for: ", var)
    print('='*width)
    
    # make data directory if they don't exist
    print("🧐 Making a {} directory if it doesn't exist...".format(DATA_ROOT / 'CMIP6' / var / 'gev'))
    os.makedirs(DATA_ROOT / 'CMIP6' / var / 'gev', exist_ok=True)
    data_path = DATA_ROOT / 'CMIP6' / var / 'landonly'

    # make all landonly file names
    fnames = [f for f in data_path.glob("*_landonly.nc")]

    # make matcher for model names to paths
    modelname_filepath_matcher = {
        extract_model_name(f): f for f in fnames
    }

    # calculate the number of ensemble members for each active model for this variable
    Nens_for_active_models = np.array([
        len(m.all_members) for m in CMIPConfig.iter_active_models(var)
    ])

    # compute the index of the model with the most members
    # NOTE: we default to the first listed if there is a tie!
    max_inds = np.where(Nens_for_active_models == np.max(Nens_for_active_models))[0]

    # set tie variable if there is a tie
    if len(max_inds) > 1:
        tie = True
    else:
        tie = False

    # select first model with most members
    ind_ = max_inds[0]

    # extract model name with most ensemble members
    model_with_most = [
        m.name for m in CMIPConfig.iter_active_models(var)
    ][ind_]

    # no loop necessary, just call the ds_mle_fit function with the full dataset
    print('-'*width)
    message =  (f"🪛 Identified {model_with_most} as model with most ensemble"
                f" members (has {np.max(Nens_for_active_models)} members).")
    print(message)
    
    # report a tie if necessary
    if tie:
        all_model_names = np.array(
            [m.name for m in CMIPConfig.iter_active_models(var)]
        )[max_inds[1:]]
        print(f"Note this model was tied with {all_model_names}!")

    #double_check = {m.name: len(m.all_members) for m in CMIPConfig.iter_active_models(var)}
    #pprint.pprint(double_check)

    # open dataset for model with the most ensemble members
    ds = xr.open_dataset(modelname_filepath_matcher[model_with_most])

    #print(modelname_filepath_matcher[model_with_most])

    # carry out GEV fits
    print('-'*width)
    print(f"🧮 Doing GEV fitting on each member for {model_with_most}...")
    print('-'*width)

    if STAT == 'stat':
        print("🥩 Doing stationary GEV fits on raw temperature data...")
        ds_raw_fit = ds_mle_fit(ds, 
                                var_name='tas', fit_dim='year',
                                non_stat=False, all_mems=True)
        reset_mle_stats()  # reset mle progress bar
        
        print("\n⚡️ Doing stationary GEV fits on temperature anomalies relative to annual mean...")
        ds_raw_annmean_fit = ds_mle_fit(ds_raw_fit, 
                                        var_name='t2m_anom_annmean', fit_dim='year',
                                        non_stat=False, all_mems=True)
        reset_mle_stats()  # reset mle progress bar
        
        print("\n⚡️ Doing stationary GEV fits on temperature anomalies relative to long term trend...")
        ds_raw_annmean_trend_fit = ds_mle_fit(ds_raw_annmean_fit, 
                                                var_name='t2m_anom_trend', fit_dim='year',
                                                non_stat=False, all_mems=True)
        reset_mle_stats()  # reset mle progress bar

    elif STAT == 'nonstat':
        print("🥩 Doing nonstationary GEV fits on raw temperature data...")
        ds_raw_fit = ds_mle_fit(ds, 
                                var_name='tas', fit_dim='year',
                                non_stat=True, all_mems=True)
        reset_mle_stats()  # reset mle progress bar
        
        print("\n⚡️ Doing nonstationary GEV fits on temperature anomalies relative to annual mean...")
        ds_raw_annmean_fit = ds_mle_fit(ds_raw_fit, 
                                        var_name='t2m_anom_annmean', fit_dim='year',
                                        non_stat=True, all_mems=True)
        reset_mle_stats()  # reset mle progress bar
        
        print("\n⚡️ Doing nonstationary GEV fits on temperature anomalies relative to long term trend...")
        ds_raw_annmean_trend_fit = ds_mle_fit(ds_raw_annmean_fit, 
                                                var_name='t2m_anom_trend', fit_dim='year',
                                                non_stat=True, all_mems=True)
        reset_mle_stats()  # reset mle progress bar
        
    elif STAT != 'nonstat' and STAT != 'stat':
        raise ValueError("⚠️ Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")

    print("\n✅ GEV fitting complete.")

    # save datasets
    fpath = modelname_filepath_matcher[model_with_most]
    gev_dir = fpath.parent.parent / 'gev'
    gev_name = fpath.with_name(
        fpath.stem + f"_gev_{STAT}_allmems" + fpath.suffix
    ).name

    # print(f"GEV directory: {gev_dir}\nGEV file name: {gev_name}\nSaved file location: {gev_dir / gev_name}")
    ds_raw_annmean_trend_fit.to_netcdf(
        gev_dir / gev_name
    )

    print(f"✍️ Dataset successfully saved to:\n{gev_dir / gev_name}.")

print('='*width)
print("🥳 All done! 🥳")
print('='*width)