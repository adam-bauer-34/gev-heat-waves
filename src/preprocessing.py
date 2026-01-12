"""Preprocessing utility functions.

Adam Bauer
UChicago
1.12.2026
"""

import shutil

import xesmf as xe
import xarray as xr

from ..config.paths import DATA_ROOT

# terminal length
width = shutil.get_terminal_size(fallback=(80, 20)).columns


def make_regridded_land_mask(GRID='1deg'):
    print('-'*width)
    print('⛰️ Regridding land mask...')

    # import high-res land mask
    land_mask = xr.open_dataset(DATA_ROOT / 'ERA5' / 'era5_land_mask.nc')

    # import temporary dataset to get lat/lon ranges
    tmp_ds = xr.open_dataset(DATA_ROOT / 'ERA5' / ('era5_t2m_annual_max_' + GRID + '.nc'))

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

    print('-'*width)
    print("⚒️ Making regridding object... (this could take a second)")
    # initialize the regridder and regrid the land mask
    regridder = xe.Regridder(land_mask, ds_output_grid, 'conservative')
    land_mask_regridded = regridder(land_mask, keep_attrs=True)

    land_mask_regridded.to_netcdf(DATA_ROOT / 'ERA5' / ('era5_land_mask_' + GRID + '.nc'))
    print('-'*width)
    print("✅ Regridded land mask saved to: ", DATA_ROOT / 'ERA5' / ('era5_land_mask_' + GRID + '.nc'))