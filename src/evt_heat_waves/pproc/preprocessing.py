"""Preprocessing utility functions.

Adam Bauer
UChicago
1.12.2026
"""

import shutil

import xesmf as xe
import xarray as xr

from evt_heat_waves.config import ERA5_PATH

# terminal length
width = shutil.get_terminal_size(fallback=(80, 20)).columns


def make_regridded_land_mask(GRID='1deg'):
    print('        Creating new land mask...')

    # import high-res land mask
    land_mask = xr.open_dataset(ERA5_PATH / 'era5_land_mask.nc')

    # import temporary dataset to get lat/lon ranges
    tmp_ds = xr.open_dataset(ERA5_PATH / ('era5_t2m_annual_max_' + GRID + '.nc'))

    # make lat/lon masks
    lat_mask = (land_mask.latitude >= min(tmp_ds.lat)) & (land_mask.latitude <= max(tmp_ds.lat))
    lon_mask = (land_mask.longitude >= min(tmp_ds.lon)) & (land_mask.longitude <= max(tmp_ds.lon))

    # select land mask within the bounds of the temporary dataset
    land_mask = land_mask.sel(longitude=lon_mask, latitude=lat_mask).copy()
    ds_output_grid = xr.Dataset(
        {
            'lat': (['lat'], tmp_ds.lat.values),
            'lon': (['lon'], tmp_ds.lon.values)
        }
    )

    # initialize the regridder and regrid the land mask
    regridder = xe.Regridder(land_mask, ds_output_grid, 'conservative')
    land_mask_regridded = regridder(land_mask, keep_attrs=True)

    land_mask_regridded.to_netcdf(ERA5_PATH / ('era5_land_mask_' + GRID + '.nc'))
    print(f"        Finished, saved regridded land mask to: {ERA5_PATH / ('era5_land_mask_' + GRID + '.nc')}")