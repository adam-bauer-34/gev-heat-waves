"""
Convert a GPWv4 ESRI ASCII raster (.asc) to a NetCDF file.

Dependencies
------------
pip install rasterio xarray netCDF4 numpy

Fill in the input and output paths below and run:
    python convert_gpw_to_netcdf.py
"""

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
import xesmf as xe

from evt_heat_waves.config import DATA_ROOT, POP_PATH, ERA5_PATH


def pproc_pop(logger, args):
    # ============================================================================
    # User inputs
    # ============================================================================

    INPUT_ASC = POP_PATH / "gpw_v4_population_count_rev11_2020_1_deg.asc"
    OUTPUT_NC = POP_PATH / "gpw_v4_population_count_rev11_2020_1_deg.nc"

    VARIABLE_NAME = "population"
    LONG_NAME = "GPWv4 Population Count"
    UNITS = "persons"

    # Set to True if your climate model expects longitudes from 0 to 360.
    CONVERT_TO_360 = True

    # ============================================================================
    # Read raster
    # ============================================================================

    logger.info("> Importing world population data...")
    with rasterio.open(INPUT_ASC) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata

    # Replace NoData values with NaN
    if nodata is not None:
        data[data == nodata] = np.nan
    logger.info("> Import successful.")

    # ============================================================================
    # Construct latitude and longitude coordinates
    # ============================================================================

    nlat, nlon = data.shape

    lon = np.array(
        [
            rasterio.transform.xy(transform, 0, j, offset="center")[0]
            for j in range(nlon)
        ]
    )

    lat = np.array(
        [
            rasterio.transform.xy(transform, i, 0, offset="center")[1]
            for i in range(nlat)
        ]
    )

    # ============================================================================
    # Create xarray Dataset
    # ============================================================================

    ds = xr.Dataset(
        data_vars={
            VARIABLE_NAME: (
                ("lat", "lon"),
                data,
                {
                    "long_name": LONG_NAME,
                    "units": UNITS,
                },
            )
        },
        coords={
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )

    # Ensure latitude is increasing (-90 -> 90), which is common for climate data
    ds = ds.sortby("lat")

    # Optionally convert longitude from [-180, 180] to [0, 360]
    if CONVERT_TO_360:
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360)).sortby("lon")

    # ============================================================================
    # Add some global metadata
    # ============================================================================

    ds.attrs = {
        "title": "GPWv4 Population Count",
        "source": "Gridded Population of the World Version 4",
        "Conventions": "CF-1.8",
    }

    # ============================================================================
    # Create metadata
    # ============================================================================

    encoding = {
        VARIABLE_NAME: {
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.nan,
        }
    }

    # ============================================================================
    # Regrid to ERA5
    # ============================================================================

    logger.info("> Regridding population data to ERA5 grid...")
    ds_era5 = xr.open_dataset(ERA5_PATH / "era5_t2m_annual_max_1deg.nc")

    regridder = xe.Regridder(ds, ds_era5[["lat", "lon"]], method="conservative")

    ds_regrid = regridder(ds)

    ds_regrid.to_netcdf(
        OUTPUT_NC,
        encoding=encoding,
    )

    # ============================================================================
    # Write NetCDF
    # ============================================================================

    logger.info(f"Successfully wrote:\n{OUTPUT_NC}")
