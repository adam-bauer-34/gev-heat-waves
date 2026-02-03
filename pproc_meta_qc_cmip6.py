"""Quality controlling CMIP6 data

Adam Michael Bauer
UChicago
1.15.2026

To run: python meta_qc_cmip6.py [data-type]
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

# parse command line
data_type = sys.argv[1]  # ai or cmip
if data_type != 'AI' and data_type != 'CMIP6':
    raise ValueError("Invalid data type; only 'CMIP6' and 'AI' supported.")

width = shutil.get_terminal_size(fallback=(80, 20)).columns

# set variables to perform qc on, should be directory names in DATA_ROOT/data_type
vars = ['tas_annual_max', 'tas_annual_mean', 'tas_annual_min']

# metadata and quality control
meta = {}
qc = {}

# required years
required_years = np.arange(1979, 2025, 1)  # same as ERA5 data

# loop through variables for QC
for var in vars:
    print('='*width)
    print("Performing QC on {} data...".format(var))
    print('='*width)

    # set data path, making sure to QC on raw data
    data_path = DATA_ROOT / data_type / var / 'raw'

    # grab files
    files = [f for f in data_path.glob('*.nc')]
    var_qc =  {}

    for f in files:
        # define empty lists for each file / model
        model_qc = {}
        model_ac = {}
        tmp_failure_mode = None
        tmp_active = True

        # get model name
        fparts = f.stem.split('_')
        model = '_'.join(fparts[2:3])

        print("-"*width)
        print("Working on {}...".format(model))

        # open dataset
        ds = xr.open_dataset(f)

        # extract relevant info
        model_yrs = ds.year.values

        # empty lists to populate later
        valid_yrs = []
        invalid_yrs = []

        # loop through the years i need. if i find any nans,
        # i add that year to the invalid years and i turn off the
        # model
        for yr in required_years:
            try:
                data = ds.tas.sel(year=yr).values
                if np.isnan(data).all():
                    invalid_yrs.append(yr)
                    tmp_active = False
                    tmp_failure_mode = 'nan_data'
                else:
                    valid_yrs.append(yr)

            except Exception as e:
                invalid_yrs.append(yr)
                tmp_active = False
                tmp_failure_mode = 'missing_data'

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

# fill in metadata
# NOTE: I'm assuming each quantity i consider has the same metadata, which should be fine,
# but i didn't check it!
meta_path = DATA_ROOT / data_type / 'tas_annual_max' / 'raw'
meta_files = [f for f in meta_path.glob('*.nc')]

for f in meta_files:
    # model-specific metadata dict
    model_meta = {}
   
    # get model name
    fparts = f.stem.split('_')
    model = '_'.join(fparts[2:3])

    # open dataset
    ds = xr.open_dataset(f)

    # extract relevant info
    model_meta['ensemble_members'] = [str(m) for m in ds.member_id.values]
    model_meta['N_members'] = int(len(model_meta['ensemble_members']))
    model_meta['primary_member'] = str(ds.member_id.values[0])  # for now
    # model_meta['active'] = qc['tas_annual_max'][model]['active']  # from qc

    # close dataset to save memory
    ds.close()

    # store model metadata in the dict
    meta[model] = model_meta

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
outpath_meta = Path('config/meta.generated.yaml')
with open(outpath_meta, 'w') as f:
    yaml.safe_dump(
        yaml_safe(metas),
        f,
        sort_keys=True,
        default_flow_style=False,
        indent=2
    )

outpath_qc = Path('config/qc.generated.yaml')
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