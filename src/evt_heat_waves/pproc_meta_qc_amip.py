"""Quality controlling AMIP data

Adam Michael Bauer
UChicago
1.15.2026

To run: python meta_qc_amip.py
"""

import sys
import shutil
import yaml

import numpy as np
import xarray as xr

from src.utils import yaml_safe
from datetime import datetime
from pathlib import Path
from config import DATA_ROOT

width = shutil.get_terminal_size(fallback=(80, 20)).columns

# set variables to perform qc on, should be directory names in DATA_ROOT/data_type
vars = ['tas_annual_max', 'tas_annual_mean', 'tas_annual_min']

# metadata and quality control
meta = {}
qc = {}

# required years
required_years = np.arange(1979, 2015, 1)  # shorter interval than CMIP

# loop through variables for QC
for var in vars:
    print('='*width)
    print("Performing QC on {} data...".format(var))
    print('='*width)

    # set data path, making sure to QC on raw data
    data_path = DATA_ROOT / "AMIP" / var / 'raw'

    # grab files
    files = [f for f in data_path.glob('*.nc')]
    var_qc =  {}

    for f in files:
        # define empty lists for each file / model
        model_qc = {}
        model_ac = {}
        model_meta = {}
        tmp_failure_mode = None
        tmp_active = True

        # get model name
        fparts = f.stem.split('_')
        model = '_'.join(fparts[2:3])

        print("-"*width)
        print("Working on {}...".format(model))

        # open dataset
        ds = xr.open_dataset(f)

        # --------------------------------
        # Step 1: Meta-analysis extraction
        # --------------------------------
        model_meta['ensemble_members'] = [str(m) for m in ds.member_id.values]
        model_meta['N_members'] = int(len(model_meta['ensemble_members']))
        model_meta['primary_member'] = str(ds.member_id.values[0])  # for now

        meta[model] = model_meta

        # --------------------------------
        # Step 2: Data Quality Controling 
        # --------------------------------
        model_yrs = ds.year.values

        # empty lists to populate later
        valid_yrs = []
        invalid_yrs = []
        valid_mems = []
        invalid_mems = []

        # loop through the years i need. if i find any nans,
        # i add that year to the invalid years and i turn off the
        # model
        for yr in required_years:
            yr_invalid_mems = []
            yr_valid_mems = []

            for mem in ds.member_id.values:
                try:
                    data = ds.tas.sel(year=yr,
                            member_id=mem).values
                    if np.isnan(data).all():
                        yr_invalid_mems.append(mem)
                        tmp_active = False
                        tmp_failure_mode = 'nan_data'
                    else:
                        yr_valid_mems.append(mem)

                except Exception as e:
                    yr_invalid_mems.append(mem)
                    tmp_active = False
                    tmp_failure_mode = 'missing_data'

            if len(yr_invalid_mems) == 0:
                valid_yrs.append(yr)
                valid_mems.extend(yr_valid_mems)

            else:
                invalid_yrs.append(yr)
                invalid_mems.extend(yr_invalid_mems)

        # populate this model's qc data
        model_qc['valid_years'] = valid_yrs
        model_qc['invalid_years'] = invalid_yrs
        model_qc['failure_mode'] = tmp_failure_mode
        model_qc['active'] = tmp_active

        # add model to variable qc
        var_qc[model] = model_qc

        # close dataset to save memory
        ds.close()

    # add variable to qc
    qc[var] = var_qc

# make final dictionaries
metas = {}
qcs = {}

# adding this to remember last time i did all this
metas['generated_on'] = datetime.now().isoformat()
qcs['generated_on'] = datetime.now().isoformat()

# populate
metas['models'] = meta
qcs['models'] = qc

# save dictionaries as .yamls 
outpath_meta = Path('config/meta_amip.generated.yaml')
with open(outpath_meta, 'w') as f:
    yaml.safe_dump(
        yaml_safe(metas),
        f,
        sort_keys=True,
        default_flow_style=False,
        indent=2
    )

outpath_qc = Path('config/qc_amip.generated.yaml')
with open(outpath_qc, 'w') as f:
    yaml.safe_dump(
        yaml_safe(qcs),
        f,
        sort_keys=True,
        default_flow_style=False,
        indent=2
    )

print('*'*width)
print("QC and metadata successfully saved!")
