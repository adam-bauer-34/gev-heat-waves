import numpy as np
import xarray as xr
from config import DATA_ROOT

def compute_ecdf(values, extend_lower=True,
                extend_upper=False, ub=None):
    """
    Compute the empirical cumulative distribution function (ECDF)
    of an xarray DataArray.

    Parameters
    ----------
    values : numpy.ndarray
        Input data. Will be flattened before computing ECDF.

    Returns
    -------
    x : numpy.ndarray
        Sorted data values.
    y : numpy.ndarray
        ECDF probabilities corresponding to x.
    """
    # Extract and flatten values
    values = values[~np.isnan(values)]

    # Sort values
    x = np.sort(values)

    # Compute ECDF probabilities
    #y = np.arange(1, len(x) + 1) / len(x)
    n = len(x)
    y = (np.arange(1, n + 1) - 0.5) / n

    if extend_lower and x[0] > 0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([0.0], y))

    if extend_upper and x[1] < ub:
        x = np.concatenate((x, [ub]))
        y = np.concatenate((y, [1.0]))    
    
    return x, y


def check_lat_lon_grids_consistent():
    """
    Check if all CMIP6 tas_annual_max landonly datasets have consistent lat/lon grids.

    Returns
    -------
    bool
        True if all grids are identical, False otherwise.
    """
    data_folder = DATA_ROOT / 'CMIP6' / 'tas_annual_max'
    fnames = list(data_folder.glob("*_landonly.nc"))
    
    if not fnames:
        print("No *_landonly.nc files found in the data folder.")
        return False
    
    # Open the first dataset to get reference lat/lon
    ds_first = xr.open_dataset(fnames[0])
    lat_ref = ds_first.lat
    lon_ref = ds_first.lon
    ds_first.close()  # Close to save memory
    
    consistent = True
    for fname in fnames[1:]:
        ds = xr.open_dataset(fname)
        if not ds.lat.equals(lat_ref) or not ds.lon.equals(lon_ref):
            print(f"Grid mismatch in {fname.name}")
            consistent = False
        ds.close()
    
    if consistent:
        print("All datasets have consistent lat/lon grids.")
    else:
        print("Some datasets have mismatched lat/lon grids.")
    
    return consistent