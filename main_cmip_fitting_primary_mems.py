"""Main file for GEV fitting of CMIP data.

Adam Michael Bauer
UChicago
Jan 2026

To run: main_cmip_fitting.py STAT

Last edited: 1/28/2026, 5:34 PM CST
"""

import os
import sys
import shutil
import pprint

import xarray as xr

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
vars = ['tas_annual_max', 'tas_annual_min']

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
    # model_names = [extract_model_name(f) for f in fnames]
    modelname_filepath_matcher = {
        extract_model_name(f): f for f in fnames
    }

    # pprint.pprint(modelname_filepath_matcher)

    # iterate through active models and perform fitting
    #tester = [m for m in CMIPConfig.iter_active_models(var)][:1]

    for m in CMIPConfig.iter_active_models(var):
    #for m in tester:
        print('-'*width)
        print("🪛 Working on ", m.name)
        fpath = modelname_filepath_matcher[m.name]  # get filepath for current active model

        ds = xr.open_dataset(fpath)  # open dataset

        # carry out GEV fits
        print('-'*width)
        print("🧮 Doing GEV fitting on primary ensemble member only...")
        print('-'*width)

        if STAT == 'stat':
            print("🥩 Doing stationary GEV fits on raw temperature data...")
            ds_raw_fit = ds_mle_fit(ds.sel(member_id=m.primary_member), 
                                    var_name='tas', fit_dim='year',
                                    non_stat=False)
            reset_mle_stats()  # reset mle progress bar
            
            print("\n⚡️ Doing stationary GEV fits on temperature anomalies relative to annual mean...")
            ds_raw_annmean_fit = ds_mle_fit(ds_raw_fit, 
                                            var_name='t2m_anom_annmean', fit_dim='year',
                                            non_stat=False)
            reset_mle_stats()  # reset mle progress bar
            
            print("\n⚡️ Doing stationary GEV fits on temperature anomalies relative to long term trend...")
            ds_raw_annmean_trend_fit = ds_mle_fit(ds_raw_annmean_fit, 
                                                  var_name='t2m_anom_trend', fit_dim='year',
                                                  non_stat=False)
            reset_mle_stats()  # reset mle progress bar

        elif STAT == 'nonstat':
            print("🥩 Doing nonstationary GEV fits on raw temperature data...")
            ds_raw_fit = ds_mle_fit(ds.sel(member_id=m.primary_member), 
                                    var_name='tas', fit_dim='year',
                                    non_stat=True)
            reset_mle_stats()  # reset mle progress bar
            
            print("\n⚡️ Doing nonstationary GEV fits on temperature anomalies relative to annual mean...")
            ds_raw_annmean_fit = ds_mle_fit(ds_raw_fit, 
                                            var_name='t2m_anom_annmean', fit_dim='year',
                                            non_stat=True)
            reset_mle_stats()  # reset mle progress bar
            
            print("\n⚡️ Doing nonstationary GEV fits on temperature anomalies relative to long term trend...")
            ds_raw_annmean_trend_fit = ds_mle_fit(ds_raw_annmean_fit, 
                                                  var_name='t2m_anom_trend', fit_dim='year',
                                                  non_stat=True)
            reset_mle_stats()  # reset mle progress bar
            
        else:
            raise ValueError("⚠️ Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
    
        print("\n✅ GEV fitting complete.")

        # save datasets
        gev_dir = fpath.parent.parent / 'gev'
        gev_name = fpath.with_name(
            fpath.stem + f"_gev_{STAT}" + fpath.suffix
        ).name

        # print(f"GEV directory: {gev_dir}\nGEV file name: {gev_name}\nSaved file location: {gev_dir / gev_name}")
        ds_raw_annmean_trend_fit.to_netcdf(
            gev_dir / gev_name
        )

        print(f"✍️ Dataset successfully saved to:\n{gev_dir / gev_name}.")

        # close datasets to save RAM
        ds_raw_fit.close()
        ds_raw_annmean_fit.close()
        ds_raw_annmean_trend_fit.close()
        ds.close()

print('='*width)
print("🥳 All done! 🥳")
print('='*width)