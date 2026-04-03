import xarray as xr
import numpy as np

from config import DATA_ROOT

example_path = DATA_ROOT / "AMIP" / "tas_annual_max" / "tas_CMIP6_BCC-CSM2-MR_day_hist+ssp585_1850-2015.nc"

ds = xr.open_dataset(example_path)

print(ds)
